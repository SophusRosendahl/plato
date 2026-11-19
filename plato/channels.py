"""Channels: the building block for every way into the agent beyond its own chat.

A channel owns one or more **artifact types** on the event log (say `file`,
`email`, `ticket`). It decides five things, each optional:

    gate_rules   which of its events wake the agent (`should_respond` rules)
    tools()      what the agent calls to act there (e.g. post a reply)
    reply_hint() the `→ Reply in:` line telling the agent where to answer
    tail_parts   bounded extra context an event drags into the wake
    guidance     a short prompt on how to behave in this channel

When the gate decides an event should be answered "on the artifact", the event
is routed to the channel that owns that artifact type. If every item in a wake
is routed to channels, the harness withholds chat `reply()`: the answer belongs
where the question was asked.

Nothing here knows about comments, email or any other product concept: those
are built on top (see `examples/`).
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
        the agent answering; the gate's self-trigger check stops it re-waking."""
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
