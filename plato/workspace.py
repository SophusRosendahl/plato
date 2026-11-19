"""The agent's state, as plain files in one folder.

    memory.md      durable facts the agent keeps about its user/world
    todo.md        the plan: `N. [ ] title` (active), [x] done, [~] blocked, [-] cancelled
    daily_log.md   one line per finished run (written by the harness on stop)
    events.db      the event log (SQLite): every event, subscriptions, the inbox
    files/         the agent's working files — the only place file tools reach
    traces/        JSONL trace per wake

Everything is human-readable on purpose: you can open the folder and watch the
agent think.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

STATUS_CHARS = {"active": " ", "done": "x", "blocked": "~", "cancelled": "-"}
CHAR_STATUS = {v: k for k, v in STATUS_CHARS.items()}
TERMINAL = {"done", "cancelled"}

_TODO_RE = re.compile(r"^(\d+)\.\s*\[(.)\]\s*(.*?)(?:\s+—\s+(.*))?$")


@dataclass
class Todo:
    position: int
    status: str
    title: str
    reason: str = ""

    def render(self) -> str:
        line = f"{self.position}. [{STATUS_CHARS[self.status]}] {self.title}"
        return f"{line} — {self.reason}" if self.reason else line


@dataclass
class InboxItem:
    id: str
    ts: str
    label: str
    text: str
    # Where the event came from and where an answer belongs. None = chat.
    channel: str | None = None
    # Channel-specific data pulled off the event (ids, payload fields).
    meta: dict = field(default_factory=dict)

    def render(self) -> str:
        return f"[{self.ts}] {self.label}: {self.text}"


class PathEscape(ValueError):
    pass


class Workspace:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.files = self.root / "files"
        self.traces = self.root / "traces"
        self.memory_path = self.root / "memory.md"
        self.todo_path = self.root / "todo.md"
        self.log_path = self.root / "daily_log.md"
        self._lock = threading.RLock()
        # Monotonic count of "durable progress" (todo resolved, file written).
        # The circuit breaker compares it before/after a run.
        self.progress = 0
        for d in (self.root, self.files, self.traces):
            d.mkdir(parents=True, exist_ok=True)
        for p in (self.memory_path, self.todo_path, self.log_path):
            p.touch(exist_ok=True)

    # ── todos ────────────────────────────────────────────────────────────
    def todos(self) -> list[Todo]:
        out = []
        for ln in self.todo_path.read_text(encoding="utf-8").splitlines():
            m = _TODO_RE.match(ln.strip())
            if m and m.group(2) in CHAR_STATUS:
                out.append(Todo(int(m.group(1)), CHAR_STATUS[m.group(2)], m.group(3), m.group(4) or ""))
        return out

    def _save_todos(self, todos: list[Todo]) -> None:
        self.todo_path.write_text("".join(t.render() + "\n" for t in todos), encoding="utf-8")

    def add_todo(self, title: str) -> int:
        with self._lock:
            todos = self.todos()
            pos = max((t.position for t in todos), default=0) + 1
            todos.append(Todo(pos, "active", " ".join(title.split())))
            self._save_todos(todos)
            return pos

    def set_todo_status(self, position: int, status: str, reason: str = "") -> Todo:
        with self._lock:
            todos = self.todos()
            for t in todos:
                if t.position == position:
                    if status in TERMINAL and t.status not in TERMINAL:
                        self.progress += 1
                    t.status, t.reason = status, " ".join(reason.split())
                    self._save_todos(todos)
                    return t
        raise KeyError(f"no todo #{position}")

    def active_todos(self) -> list[Todo]:
        return [t for t in self.todos() if t.status == "active"]

    def non_terminal_todos(self) -> list[Todo]:
        return [t for t in self.todos() if t.status not in TERMINAL]

    def clear_resolved_todos(self) -> None:
        with self._lock:
            keep = [t for t in self.todos() if t.status not in TERMINAL]
            for i, t in enumerate(keep, 1):
                t.position = i
            self._save_todos(keep)

    def render_todos(self) -> str:
        return "\n".join(t.render() for t in self.todos()) or "(empty)"

    # ── memory + log ─────────────────────────────────────────────────────
    def memory(self) -> str:
        return self.memory_path.read_text(encoding="utf-8").strip()

    def remember(self, fact: str) -> None:
        with self.memory_path.open("a", encoding="utf-8") as f:
            f.write(f"- {' '.join(fact.split())}\n")

    def log(self, line: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M")
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"- [{ts}] {' '.join(line.split())}\n")

    def log_tail(self, n: int = 20) -> str:
        lines = self.log_path.read_text(encoding="utf-8").splitlines()
        return "\n".join(lines[-n:])

    # ── files (sandboxed to files/) ──────────────────────────────────────
    def resolve(self, path: str) -> Path:
        p = (self.files / path).resolve()
        if p != self.files and self.files not in p.parents:
            raise PathEscape(f"path {path!r} is outside the workspace files/ folder")
        return p

    def note_write(self) -> None:
        self.progress += 1
