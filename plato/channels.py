"""Channels: where events come from, what context they drag in, where answers go.

Chat is only one way in. A comment on a document, an email, a file landing in
a folder, a webhook: each is a channel. An event that arrives on a channel
carries three things into the agent's wake, beyond its own text:

1. **A reply route.** A `→ Reply in:` line naming the call that answers in that
   channel. When every item in a wake is routed to a channel, the harness also
   stops chat `reply()` from going out (a prompt alone does not reliably prevent
   a stray chat message).
2. **A tail.** Bounded extra context the event drags in at render time: for a
   comment, the quoted passage, the recent thread, and a reference to the
   document (path + size, the body is read on demand, not inlined).
3. **Rules.** A short channel-specific prompt, rendered once per wake.

A channel is a small class: subclass `Channel`, give it a `name`, and override
what you need. `CommentsChannel` is a complete worked example.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from plato.context import render
from plato.events import Event, EventLog
from plato.tools import ToolRegistry
from plato.workspace import InboxItem, PathEscape, Workspace

if TYPE_CHECKING:
    from plato.agent import Agent


class Channel:
    """Base class. `artifact_type` is the artifact type this channel answers
    for: when the gate decides an event should be answered on an artifact of
    that type, the event is routed here."""

    name: str = ""
    artifact_type: str = ""

    def bind(self, agent: Agent) -> None:
        """Called once when the agent is built."""
        self.agent = agent
        self.ws: Workspace = agent.ws
        self.log: EventLog = agent.log

    def tools(self) -> ToolRegistry | None:
        """Tools the agent uses to act in this channel (e.g. `comment()`)."""
        return None

    def label(self, event: Event) -> str:
        """How the event is introduced in the inbox."""
        return ""

    def meta(self, event: Event) -> dict:
        """Data the channel needs later (reply hint, tail) pulled off the event."""
        return dict(event.payload)

    def reply_hint(self, item: InboxItem) -> str:
        """The call that answers this event, shown on a `→ Reply in:` line."""
        return ""

    def tail(self, item: InboxItem) -> str:
        """Extra context this event drags into the wake. Keep it bounded."""
        return ""

    def rules(self) -> str:
        """Channel-specific guidance, rendered once per wake."""
        return ""


class CommentsChannel(Channel):
    """Comment threads on documents. Each document is an artifact `doc:<path>`
    on the event log; a comment is a `comment` event on it, and a thread is
    just the comment events on that artifact.

    Your app calls `post(entity, text, quote=...)` when a person comments. The
    agent answers with `comment(entity, text, parent_id=...)`; that is an event
    by the agent, so the gate's self-trigger rule keeps it from re-waking.
    """

    name = "comments"
    artifact_type = "doc"

    def __init__(self, max_thread: int = 6, max_comment_chars: int = 300) -> None:
        self.max_thread = max_thread
        self.max_comment_chars = max_comment_chars

    @staticmethod
    def artifact(entity: str) -> str:
        return f"doc:{entity}"

    def thread(self, entity: str) -> list[Event]:
        return self.log.recent(self.artifact(entity), limit=10**6, surfaces=("comment",))

    def post(
        self,
        entity: str,
        text: str,
        *,
        author: str | None = None,
        quote: str | None = None,
        parent_id: str | None = None,
    ) -> Event:
        """A person comments on `entity`. The agent follows the document, so the
        dispatcher will route the comment to it."""
        self.agent.follow(self.artifact(entity), self.artifact_type)
        return self.log.submit(
            self.artifact(entity), actor_type="human", actor_id=author or self.agent.human_id,
            surface="comment", parent_id=parent_id,
            payload={"text": text, "entity": entity, "quote": quote},
        )

    # ── agent side ───────────────────────────────────────────────────────
    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Comments")
        def comment(entity: str, text: str, parent_id: str | None = None) -> str:
            """Post a comment on a file; pass parent_id to reply inside a thread."""
            self.agent.follow(self.artifact(entity), self.artifact_type)
            ev = self.log.submit(
                self.artifact(entity), actor_type="agent", actor_id=self.agent.agent_id,
                surface="comment", parent_id=parent_id, payload={"text": str(text), "entity": entity})
            self.agent._emit("channel_reply", channel=self.name, entity=entity, text=ev.text)
            return f"commented on {entity} ({ev.event_id})"

        @reg.tool(group="Comments")
        def list_comments(entity: str) -> str:
            """Show the comments on a file, oldest first."""
            rows = [f"{c.event_id} {c.actor_type}: {c.text}" for c in self.thread(entity)]
            return "\n".join(rows) or "(no comments)"

        return reg

    def label(self, event: Event) -> str:
        return f"Comment on {event.payload.get('entity')}"

    def meta(self, event: Event) -> dict:
        return {"entity": event.payload.get("entity"), "quote": event.payload.get("quote"),
                "comment_id": event.event_id, "thread_root": event.parent_id or event.event_id}

    def reply_hint(self, item: InboxItem) -> str:
        m = item.meta
        return f"comment(entity={m['entity']!r}, text='...', parent_id={m['thread_root']!r})"

    def tail(self, item: InboxItem) -> str:
        m = item.meta
        parts: list[str] = []
        if m.get("quote"):
            parts.append(render("tail_quote", QUOTE=m["quote"]))
        artifact, skip = self.artifact(m["entity"]), {m["comment_id"]}
        recent = self.log.recent(artifact, limit=self.max_thread, surfaces=("comment",), exclude=skip)
        if recent:
            total = self.log.count(artifact, surfaces=("comment",), exclude=skip)
            header = (render("tail_thread_partial", RECENT=len(recent), TOTAL=total)
                      if len(recent) < total else render("tail_thread_all"))
            lines = []
            for c in recent:
                who = "You" if c.actor_type == "agent" and c.actor_id == self.agent.agent_id else "User"
                body = " ".join(c.text.split())
                if len(body) > self.max_comment_chars:
                    body = body[: self.max_comment_chars - 1] + "…"
                lines.append(f"- {who} ({c.event_id}): {body}")
            parts.append(header + "\n" + "\n".join(lines))
        try:
            doc = self.ws.resolve(m["entity"])
        except PathEscape:
            doc = None
        if doc is not None and doc.is_file():
            body = doc.read_text(encoding="utf-8", errors="replace")
            parts.append(render("tail_doc_reference", PATH=m["entity"], CHARS=len(body),
                                LINES=body.count("\n") + 1))
        return "\n\n".join(parts)

    def rules(self) -> str:
        return render("channel_comments_rules")
