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


def render_inbox(items: list[InboxItem]) -> str:
    return "\n".join(i.render() for i in items) or "(empty)"


def build_wake_context(ws: Workspace, inbox: list[InboxItem], now: datetime | None = None) -> str:
    now = now or datetime.now()
    active = ws.active_todos()
    todo_header = "--- todo ---" + ("  ⚠ PENDING ITEMS" if active and not inbox else "")
    sections = {
        "memory": f"--- memory ---\n{ws.memory() or '(empty)'}",
        "todo": f"{todo_header}\n{ws.render_todos()}",
        "log": f"--- daily log (recent) ---\n{ws.log_tail() or '(empty)'}",
        "inbox": f"--- inbox ---\n{render_inbox(inbox)}",
    }
    if inbox:
        # Inbox last: it is closest to where the model starts writing.
        order, footer = ["memory", "todo", "log", "inbox"], render("footer_new_items")
    elif active:
        order, footer = ["inbox", "todo", "memory", "log"], render("footer_pending_todos")
    else:
        order, footer = ["inbox", "todo", "memory", "log"], ""
    head = f"Now: {now.strftime('%A %Y-%m-%d %H:%M')}"
    body = "\n\n".join(sections[k] for k in order)
    return "\n\n".join(p for p in (head, body, footer) if p)
