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

import json
import threading
import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from plato.context import render
from plato.tools import ToolRegistry
from plato.workspace import InboxItem, PathEscape, Workspace

if TYPE_CHECKING:
    from plato.agent import Agent


class Channel:
    name: str = ""

    def bind(self, agent: Agent) -> None:
        """Called once when the agent is built."""
        self.agent = agent
        self.ws: Workspace = agent.ws

    def tools(self) -> ToolRegistry | None:
        """Tools the agent uses to act in this channel (e.g. `comment()`)."""
        return None

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
    """Comment threads on files in the workspace (or any named entity).

    Your app calls `post(entity, text, quote=...)` when a person comments. The
    agent answers with `comment(entity, text, parent_id=...)`.
    """

    name = "comments"

    def __init__(self, max_thread: int = 6, max_comment_chars: int = 300) -> None:
        self.max_thread = max_thread
        self.max_comment_chars = max_comment_chars
        self._lock = threading.Lock()

    def bind(self, agent: Agent) -> None:
        super().bind(agent)
        self.path = self.ws.root / "comments.jsonl"
        self.path.touch(exist_ok=True)

    # ── store ────────────────────────────────────────────────────────────
    def all(self) -> list[dict]:
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(ln) for ln in lines if ln.strip()]

    def thread(self, entity: str) -> list[dict]:
        return [c for c in self.all() if c["entity"] == entity]

    def post(
        self,
        entity: str,
        text: str,
        *,
        author: str = "user",
        quote: str | None = None,
        parent_id: str | None = None,
        wake: bool = True,
    ) -> dict:
        """Add a comment. A person's comment lands in the agent's inbox."""
        c = {
            "id": "c" + uuid.uuid4().hex[:6],
            "entity": entity,
            "author": author,
            "text": text,
            "quote": quote,
            "parent_id": parent_id,
            "ts": datetime.now().isoformat(timespec="seconds"),
        }
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(c) + "\n")
        if author != "agent" and wake:
            self.ws.inbox_append(
                text,
                label=f"Comment on {entity}",
                channel=self.name,
                meta={"comment_id": c["id"], "entity": entity, "quote": quote,
                      "thread_root": parent_id or c["id"]},
            )
        return c

    # ── agent side ───────────────────────────────────────────────────────
    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Comments")
        def comment(entity: str, text: str, parent_id: str | None = None) -> str:
            """Post a comment on a file; pass parent_id to reply inside a thread."""
            c = self.post(entity, str(text), author="agent", parent_id=parent_id, wake=False)
            self.agent._emit("channel_reply", channel=self.name, entity=entity, text=c["text"])
            return f"commented on {entity} ({c['id']})"

        @reg.tool(group="Comments")
        def list_comments(entity: str) -> str:
            """Show the comments on a file, oldest first."""
            rows = [f"{c['id']} {c['author']}: {c['text']}" for c in self.thread(entity)]
            return "\n".join(rows) or "(no comments)"

        return reg

    def reply_hint(self, item: InboxItem) -> str:
        m = item.meta
        return f"comment(entity={m['entity']!r}, text='...', parent_id={m['thread_root']!r})"

    def tail(self, item: InboxItem) -> str:
        m = item.meta
        parts: list[str] = []
        if m.get("quote"):
            parts.append(render("tail_quote", QUOTE=m["quote"]))
        prior = [c for c in self.thread(m["entity"]) if c["id"] != m["comment_id"]]
        if prior:
            recent = prior[-self.max_thread:]
            header = (
                render("tail_thread_partial", RECENT=len(recent), TOTAL=len(prior))
                if len(recent) < len(prior) else render("tail_thread_all")
            )
            lines = []
            for c in recent:
                who = "You" if c["author"] == "agent" else "User"
                body = " ".join(c["text"].split())
                if len(body) > self.max_comment_chars:
                    body = body[: self.max_comment_chars - 1] + "…"
                lines.append(f"- {who} ({c['id']}): {body}")
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
