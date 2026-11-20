"""Channels: the building block for every way in and out of the agent.

A channel owns one or more **artifact types** on the event log (say `thread`,
`file`, `email`, `ticket`). It decides these things, each optional:

    gate_rules     which of its events wake the agent (`should_respond` rules)
    tools()        what the agent calls to act there (e.g. post a reply)
    reply_hint()   the `→ Reply in:` line telling the agent where to answer
    tail_parts     bounded extra context an event drags into the wake
    guidance       a short prompt on how to behave in this channel
    context()      a section shown on *every* wake (e.g. the recent conversation)
    notify()       how the harness reaches the person, if this channel can
    answer_only_when_asked
                   hold back this channel's actions on a wake where nothing
                   arrived through it, so an answer lands where the question was
                   asked (a prompt alone doesn't reliably prevent a stray reply)

The gate routes an event to the channel named by its rule's `answer`, or else to
the channel owning the event's artifact type.

Nothing here knows about chat, comments, email or any other product concept.
Chat is built in (`plato/chat.py`) but uses only this interface; the rest are
built on top (see `examples/`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plato.events import Event, EventLog
from plato.should_respond import Rule
from plato.tools import ToolRegistry
from plato.workspace import InboxItem, Workspace

if TYPE_CHECKING:
    from plato.agent import Agent
    from plato.tails import TailPart


class Channel:
    name: str = ""
    artifact_types: tuple[str, ...] = ()
    gate_rules: tuple[Rule, ...] = ()
    tail_parts: tuple[TailPart, ...] = ()
    guidance: str = ""
    answer_only_when_asked: bool = False

    def bind(self, agent: Agent) -> None:
        """Called once when the agent is built."""
        self.agent = agent
        self.ws: Workspace = agent.ws
        self.log: EventLog = agent.log

    # ── writing events ───────────────────────────────────────────────────
    def receive(
        self, artifact_id: str, artifact_type: str, surface: str, text: str, *,
        actor_id: str | None = None, actor_type: str = "human", op: str | None = None,
        parent_id: str | None = None, **payload,
    ) -> Event:
        """Record something that happened in this channel (a person's action, a
        webhook…). The agent follows the artifact so the gate sees the event."""
        self.agent.follow(artifact_id, artifact_type)
        return self.log.submit(
            artifact_id, actor_type=actor_type, actor_id=actor_id or self.agent.human_id,
            surface=surface, op=op, parent_id=parent_id, payload={"text": text, **payload})

    def act(
        self, artifact_id: str, artifact_type: str, surface: str, text: str, *,
        parent_id: str | None = None, **payload,
    ) -> Event:
        """Record the agent acting in this channel (e.g. its reply). Counts as
        the agent answering; the gate's self-trigger check stops it re-waking.
        Returns None, recording nothing, if this channel is held back on this
        wake (see `answer_only_when_asked`); tools should then return WITHHELD."""
        if self.withheld():
            self.agent._emit("withheld", channel=self.name, artifact=artifact_id, text=text)
            return None
        self.agent.follow(artifact_id, artifact_type)
        ev = self.log.submit(
            artifact_id, actor_type="agent", actor_id=self.agent.agent_id, surface=surface,
            parent_id=parent_id, payload={"text": text, **payload})
        self.agent._emit("channel_reply", channel=self.name, artifact=artifact_id, text=text)
        return ev

    # ── what the agent sees ──────────────────────────────────────────────
    def tools(self) -> ToolRegistry | None:
        return None

    def label(self, event: Event) -> str:
        """How the event is introduced in the inbox."""
        return f"{event.surface} on {event.artifact_id}"

    def meta(self, event: Event) -> dict:
        """What reply hints and tail parts can use: the payload plus ids."""
        return {
            **event.payload,
            "event_id": event.event_id,
            "artifact_id": event.artifact_id,
            "surface": event.surface,
            "actor_id": event.actor_id,
            "parent_id": event.parent_id,
            "thread_root": event.parent_id or event.event_id,
        }

    def reply_hint(self, item: InboxItem) -> str:
        return ""

    def tail(self, item: InboxItem) -> str:
        parts = (part.render(self, item) for part in self.tail_parts)
        return "\n\n".join(p for p in parts if p)

    def guide(self) -> str:
        return self.guidance.strip()

    def context(self, exclude: set[str]) -> str:
        """A section for every wake, whatever the inbox holds. `exclude` is the
        ids of the events being delivered now (so they aren't shown twice)."""
        return ""

    def notify(self, text: str) -> bool:
        """Tell the person something from the harness itself. Return True if
        delivered. Channels that can't reach a person return False."""
        return False

    def withheld(self) -> bool:
        return self.agent._withheld(self)


WITHHELD = "[not delivered: nothing in this wake arrived through this channel; answer where you were asked, on the `→ Reply in:` line]"
