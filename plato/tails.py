"""Tail parts: composable, bounded context an event drags into the wake.

A channel lists the parts it wants; each renders a block (or nothing) for an
inbox item. Compose them to get, for example, "the quoted passage + the last
six messages in this thread + a pointer to the file".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from plato.context import render
from plato.workspace import InboxItem, PathEscape

if TYPE_CHECKING:
    from plato.channels import Channel


class TailPart(Protocol):
    def render(self, channel: Channel, item: InboxItem) -> str: ...


@dataclass(frozen=True)
class PayloadField:
    """Render one field of the event's payload, if present."""

    key: str
    template: str = "{value}"

    def render(self, channel: Channel, item: InboxItem) -> str:
        value = item.meta.get(self.key)
        return self.template.format(value=value) if value else ""


@dataclass(frozen=True)
class RecentEvents:
    """The recent tail of the event's artifact, oldest first, bounded in count
    and per-event length. The triggering event itself is excluded."""

    surfaces: tuple[str, ...] | None = None
    limit: int = 6
    max_chars: int = 300

    def render(self, channel: Channel, item: InboxItem) -> str:
        artifact, skip = item.meta["artifact_id"], {item.id}
        log = channel.log
        recent = log.recent(artifact, limit=self.limit, surfaces=self.surfaces, exclude=skip)
        if not recent:
            return ""
        total = log.count(artifact, surfaces=self.surfaces, exclude=skip)
        header = (render("tail_recent_partial", RECENT=len(recent), TOTAL=total)
                  if len(recent) < total else render("tail_recent_all"))
        lines = []
        for ev in recent:
            me = ev.actor_type == "agent" and ev.actor_id == channel.agent.agent_id
            who = "You" if me else ev.actor_id
            body = " ".join(ev.text.split())
            if len(body) > self.max_chars:
                body = body[: self.max_chars - 1] + "…"
            lines.append(f"- {who} ({ev.event_id}): {body}")
        return header + "\n" + "\n".join(lines)


@dataclass(frozen=True)
class FileReference:
    """Point at a workspace file named in the payload: path and size only. The
    body is read on demand, so a big file never bloats every wake."""

    key: str = "path"

    def render(self, channel: Channel, item: InboxItem) -> str:
        path = item.meta.get(self.key)
        if not path:
            return ""
        try:
            f = channel.ws.resolve(path)
        except PathEscape:
            return ""
        if not f.is_file():
            return ""
        body = f.read_text(encoding="utf-8", errors="replace")
        return render("tail_file_reference", PATH=path, CHARS=len(body), LINES=body.count("\n") + 1)
