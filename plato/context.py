"""Prompt templates and the wake context — what the model sees when it wakes."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from plato.workspace import InboxItem, Workspace

PROMPTS = Path(__file__).parent / "prompts"


def render(name: str, **values: str) -> str:
    """Load prompts/<name>.md and fill {{VAR}} placeholders in one pass."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    for key, val in values.items():
        text = text.replace("{{" + key + "}}", str(val))
    return text.strip()


def build_system_prompt(identity: str = "") -> str:
    return render("system", IDENTITY=identity.strip())


def build_tool_schema(tools_catalogue: str) -> dict:
    return {
        "type": "function",
        "function": {
            "name": "run",
            "description": render("tool_description", TOOLS=tools_catalogue),
            "parameters": {
                "type": "object",
                "properties": {
                    "thoughts": {
                        "type": "string",
                        "description": "Brief plan for THIS code block, 1-3 short sentences.",
                    },
                    "code": {"type": "string", "description": "Python code to execute"},
                },
                "required": ["thoughts", "code"],
            },
        },
    }


def render_inbox(items: list[InboxItem], channels: dict | None = None) -> str:
    """Render inbox items. An item on a known channel gets its `→ Reply in:`
    route and its tail; each channel's guidance is appended once."""
    channels = channels or {}
    blocks: list[str] = []
    used: list = []
    for item in items:
        ch = channels.get(item.channel) if item.channel else None
        if ch is None:
            blocks.append(item.render())
            continue
        parts = []
        hint = ch.reply_hint(item)
        if hint:
            parts.append(f"→ Reply in: {hint}")
        parts.append(item.render())
        tail = ch.tail(item)
        if tail:
            parts.append(tail)
        blocks.append("\n".join(parts))
        if ch not in used:
            used.append(ch)
    rules = [g for g in (ch.guide() for ch in used) if g]
    return "\n\n".join(blocks + rules) or "(empty)"


def routed_to_channel(items: list[InboxItem], channels: dict | None = None) -> bool:
    """True when every item belongs to a known non-chat channel."""
    channels = channels or {}
    return bool(items) and all(i.channel in channels for i in items)


def build_wake_context(
    ws: Workspace,
    inbox: list[InboxItem],
    now: datetime | None = None,
    channels: dict | None = None,
    recall: str = "",
) -> str:
    now = now or datetime.now()
    active = ws.active_todos()
    todo_header = "--- todo ---" + ("  ⚠ PENDING ITEMS" if active and not inbox else "")
    sections = {
        "memory": f"--- memory ---\n{ws.memory() or '(empty)'}",
        "todo": f"{todo_header}\n{ws.render_todos()}",
        "log": f"--- daily log (recent) ---\n{ws.log_tail() or '(empty)'}",
        "recall": f"--- recent conversation ---\n{recall}" if recall else "",
        "inbox": f"--- inbox ---\n{render_inbox(inbox, channels)}",
    }
    if inbox:
        # Inbox last: it is closest to where the model starts writing.
        footer = render("footer_reply_channel" if routed_to_channel(inbox, channels) else "footer_new_items")
        order = ["memory", "todo", "log", "recall", "inbox"]
    elif active:
        order, footer = ["inbox", "todo", "memory", "log", "recall"], render("footer_pending_todos")
    else:
        order, footer = ["inbox", "todo", "memory", "log", "recall"], ""
    head = f"Now: {now.strftime('%A %Y-%m-%d %H:%M')}"
    body = "\n\n".join(sections[k] for k in order if sections[k])
    return "\n\n".join(p for p in (head, body, footer) if p)
