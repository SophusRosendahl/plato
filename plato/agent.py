"""The harness: wake → run code step by step → stop → dormant.

Smart model, dumb harness. The loop does lifecycle plumbing and a few hard
guarantees; every judgment (what to do, whether to plan, what to say) is the
model's. The guarantees, each of which exists because of a real failure:

- **Open-todo stop guard.** `stop()` is refused while any todo is still active,
  so work can't be silently abandoned. Bounded by `max_stop_rejections` so a
  confused model ends gracefully instead of thrashing.
- **Silent-stop guard.** Stopping on the first step with new inbox items, no
  reply and no plan is refused once — the user would otherwise get nothing.
- **Budget notice.** The step limit is invisible to a model, so it is told
  shortly before it runs out and can land the work instead of being cut off.
- **Circuit breaker.** A run that hits the step limit with no durable progress
  (no todo resolved, no file written) counts as a no-progress wake. After K in a
  row on the same todos, the items are parked as blocked, the user is told once,
  and later wakes on that unchanged set get a small step budget.
"""

from __future__ import annotations

import json
import re
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

from plato.context import (
    build_system_prompt,
    build_tool_schema,
    build_wake_context,
    render,
    render_inbox,
    routed_to_channel,
)
from plato.events import Dispatched, Event, EventLog
from plato.executor import Executor, StopRun
from plato.model import Model
from plato.should_respond import Policy
from plato.tools import ToolRegistry, default_registry
from plato.workspace import InboxItem, Workspace


@dataclass
class Config:
    max_steps: int = 30
    max_stop_rejections: int = 5
    breaker_max_wakes: int = 3
    breaker_paused_steps: int = 8
    wrap_up_steps: int = 5
    history_max_messages: int = 40
    recall_messages: int = 12  # recent chat turns (both sides) shown on every wake
    output_cap: int = 8_000


@dataclass
class WakeResult:
    outcome: str  # stopped | stop_guard_exhausted | budget_exhausted | stuck
    steps: int
    replies: list[str] = field(default_factory=list)
    summary: str = ""
    trace_path: Path | None = None


def trim_history(messages: list[dict], keep: int) -> list[dict]:
    """Keep the system message + roughly the last `keep` messages, never starting
    on an orphaned tool result and preferring to start on a user turn."""
    if keep < 1 or len(messages) - 1 <= keep:
        return messages
    cut = len(messages) - keep
    while cut > 1 and messages[cut].get("role") == "tool":
        cut -= 1
    if cut > 1 and messages[cut].get("role") == "assistant" and messages[cut - 1].get("role") == "user":
        cut -= 1
    return [messages[0]] + messages[cut:]


def _default_label(ev: Event) -> str:
    if ev.surface == "chat":
        return "User says"
    what = ev.op or ev.surface
    return f"{what} on {ev.artifact_id}"


_TAGS = re.compile(r"<(script|style)[^>]*>.*?</\1>|<[^>]+>", re.DOTALL | re.IGNORECASE)


class Agent:
    def __init__(
        self,
        workspace: Workspace | str | Path,
        model: Model,
        *,
        config: Config | None = None,
        tools: ToolRegistry | None = None,
        identity: str = "",
        channels: list | None = None,
        log: EventLog | None = None,
        policy: Policy | None = None,
        agent_id: str = "agent",
        human_id: str = "user",
        on_reply: Callable[[str], None] | None = None,
        on_event: Callable[[str, dict], None] | None = None,
    ) -> None:
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace(workspace)
        self.model = model
        self.cfg = config or Config()
        self.on_reply = on_reply
        self.on_event = on_event
        self._trace_file: Path | None = None
        # The event log is the source of truth for everything that happens; the
        # inbox is derived from it. Pass a shared log to run several agents on it.
        self.log = log or EventLog(self.ws.root / "events.db")
        self.agent_id, self.human_id = agent_id, human_id
        self.thread = f"thread:{agent_id}"
        self.log.ensure_artifact(self.thread, "thread", owner_id=human_id)
        self.log.subscribe(agent_id, self.thread)
        self._dispatch_log: list[Dispatched] = []
        self.registry = self._builtin_tools()
        self.registry.extend(default_registry)
        if tools is not None:
            self.registry.extend(tools)
        # Channels: other ways in besides chat (comments, email, webhooks…).
        self.channels: dict = {}
        for ch in channels or []:
            ch.bind(self)
            self.channels[ch.name] = ch
            ch_tools = ch.tools()
            if ch_tools is not None:
                self.registry.extend(ch_tools)
        # The gate's rules: the base policy (chat by default), then each
        # channel's rules for its own events. First match wins.
        self.policy = (policy or Policy()).with_rules(
            *(r for ch in self.channels.values() for r in ch.gate_rules))
        self.log.set_policy(agent_id, self.policy)
        self.system_prompt = build_system_prompt(identity)
        self.tool_schema = build_tool_schema(self.registry.describe())
        # circuit-breaker state (in memory; a restart is a natural reset)
        self._breaker_sig: tuple | None = None
        self._no_progress_wakes = 0
        self._breaker_tripped = False
        # per-wake
        self._replies: list[str] = []
        self._channel_replies = 0
        self._suppress_chat = False

    # ── public API ───────────────────────────────────────────────────────
    def send(self, text: str) -> Event:
        """A person's chat message: appended to the log. It does not run the
        agent by itself; `wake()` does."""
        return self.log.submit(self.thread, actor_type="human", actor_id=self.human_id,
                               surface="chat", payload={"text": text})

    def follow(self, artifact_id: str, type: str) -> None:  # noqa: A002
        """Register an artifact and subscribe this agent to its events."""
        self.log.ensure_artifact(artifact_id, type)
        self.log.subscribe(self.agent_id, artifact_id)

    def pending_inbox(self) -> list[InboxItem]:
        """Run the dispatcher, then return what this agent hasn't handled yet."""
        self._dispatch_log += self.log.dispatch_once()
        return [self._inbox_item(ev) for ev in self.log.pending_inbox(self.agent_id)]

    def wake(self, force: bool = False) -> WakeResult | None:
        """Run one wake if there is anything in the inbox (or `force` and there
        are active todos). Returns None if the agent stays dormant."""
        inbox = self.pending_inbox()
        if not inbox and not (force and self.ws.active_todos()):
            return None
        return self._run(inbox)

    # ── the loop ─────────────────────────────────────────────────────────
    def _run(self, inbox) -> WakeResult:
        cfg, ws = self.cfg, self.ws
        self._replies, self._channel_replies = [], 0
        self._trace_file = ws.traces / f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}.jsonl"

        sig = self._todo_signature()
        if sig != self._breaker_sig:
            self._breaker_sig, self._no_progress_wakes, self._breaker_tripped = sig, 0, False
        budget = min(cfg.breaker_paused_steps, cfg.max_steps) if self._breaker_tripped else cfg.max_steps

        recall = self._render_recall(exclude={i.id for i in inbox})
        context = build_wake_context(ws, inbox, channels=self.channels, recall=recall)
        # Every item arrived on a channel → answers belong there, not in chat.
        # The prompt says so too, but a prompt alone doesn't reliably stop a
        # stray chat message, so the harness withholds chat reply() as well.
        self._suppress_chat = routed_to_channel(inbox, self.channels)
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": context}]
        if self._no_progress_wakes:
            messages.append({"role": "user", "content": render(
                "correction_circuit_breaker", WAKES=self._no_progress_wakes)})
        self.log.mark_done(self.agent_id, {i.id for i in inbox})
        for d in self._dispatch_log:
            self._emit("gate", event=d.event.event_id, surface=d.event.surface, artifact=d.event.artifact_id,
                       participant=d.participant, respond=d.decision.respond, reason=d.decision.reason)
        self._dispatch_log = []
        self._emit("wake", context=context, budget=budget, inbox=len(inbox))

        executor = Executor(self.registry.namespace(), cfg.output_cap)
        progress_start = ws.progress
        had_inbox = bool(inbox)
        wrap_at = min(cfg.wrap_up_steps, budget // 3)
        step = used = rejections = 0
        warned = False

        while used < budget:
            step += 1
            used += 1
            if step > 1:
                arrived = self.pending_inbox()
                if arrived:
                    self.log.mark_done(self.agent_id, {i.id for i in arrived})
                    if not routed_to_channel(arrived, self.channels):
                        self._suppress_chat = False  # a chat message arrived: chat is live again
                    items = render_inbox(arrived, self.channels)
                    messages.append({"role": "user", "content": render("mid_run_message", ITEMS=items)})
                    self._emit("mid_run_inbox", items=len(arrived))
                    used, warned, had_inbox = 1, False, True  # new input buys a fresh budget
            if not warned and wrap_at > 0 and budget - used < wrap_at:
                messages.append({"role": "user", "content": render("budget_wrap_up", STEPS_LEFT=budget - used + 1)})
                warned = True
                self._emit("guard", name="budget_wrap_up", steps_left=budget - used + 1)

            messages = trim_history(messages, cfg.history_max_messages)
            turn = self.model.complete(messages, self.tool_schema)
            if turn.code is None:
                messages.append({"role": "assistant", "content": turn.text or ""})
                messages.append({"role": "user", "content": render("nudge_no_tool_call")})
                self._emit("no_tool_call", text=turn.text)
                continue

            replies_before = len(self._replies)
            result = executor.run(turn.code)
            sent = len(self._replies) - replies_before
            out = result.output or "(no output)"
            if sent:
                out += f"\n[{sent} reply(s) delivered to the user]"
            messages.append({"role": "assistant", "content": turn.text or None, "tool_calls": [{
                "id": turn.call_id, "type": "function",
                "function": {"name": "run", "arguments": json.dumps({"thoughts": turn.thoughts, "code": turn.code})},
            }]})
            messages.append({"role": "tool", "tool_call_id": turn.call_id, "content": out})
            self._emit("step", step=step, thoughts=turn.thoughts, code=turn.code,
                       status=result.status, output=result.output, usage=turn.usage)

            if result.status != "stop":
                continue

            # ── stop requested: apply the guards ──
            answered = self._replies or self._channel_replies
            if step == 1 and had_inbox and not answered and not ws.non_terminal_todos():
                rejections += 1
                messages.append({"role": "user", "content": render("correction_planning_guard")})
                self._emit("guard", name="silent_stop")
                continue
            active = ws.active_todos()
            if active:
                rejections += 1
                if rejections > cfg.max_stop_rejections:
                    ws.log(f"(ended: stop guard exhausted with {len(active)} active todo(s)) {result.summary}")
                    self._emit("end", outcome="stop_guard_exhausted")
                    return self._result("stop_guard_exhausted", step, result.summary)
                lines = "\n".join(t.render() for t in active)
                messages.append({"role": "user", "content": render("correction_stop_guard", TODO_LINES=lines)})
                self._emit("guard", name="open_todo_stop", active=len(active), rejection=rejections)
                continue

            ws.log(result.summary or "(stopped without a summary)")
            ws.clear_resolved_todos()
            self._breaker_sig, self._no_progress_wakes, self._breaker_tripped = self._todo_signature(), 0, False
            self._emit("end", outcome="stopped", summary=result.summary)
            return self._result("stopped", step, result.summary)

        # ── budget ran out ──
        if ws.progress > progress_start:
            self._no_progress_wakes, self._breaker_tripped = 0, False
            self._breaker_sig = self._todo_signature()
            ws.log("(ran out of steps; work carried over to the next wake)")
            self._emit("end", outcome="budget_exhausted")
            return self._result("budget_exhausted", step)

        sig = self._todo_signature()
        if sig != self._breaker_sig:
            self._breaker_sig, self._no_progress_wakes = sig, 0
        self._no_progress_wakes += 1
        if self._no_progress_wakes >= cfg.breaker_max_wakes and not self._breaker_tripped:
            self._breaker_tripped = True
            stuck = ws.active_todos()
            for t in stuck:
                ws.set_todo_status(t.position, "blocked", "paused: no progress after repeated attempts")
            self._breaker_sig = self._todo_signature()
            self._deliver(
                f"I've paused: I made no progress on {len(stuck)} item(s) after "
                f"{self._no_progress_wakes} attempts. Tell me what to change and I'll pick it back up."
            )
            ws.log(f"(circuit breaker tripped after {self._no_progress_wakes} no-progress wakes)")
            self._emit("guard", name="circuit_breaker_tripped", wakes=self._no_progress_wakes)
            self._emit("end", outcome="stuck")
            return self._result("stuck", step)
        ws.log("(ran out of steps with no progress)")
        self._emit("end", outcome="budget_exhausted", no_progress_wakes=self._no_progress_wakes)
        return self._result("budget_exhausted", step)

    # ── helpers ──────────────────────────────────────────────────────────
    def _result(self, outcome: str, steps: int, summary: str = "") -> WakeResult:
        return WakeResult(outcome, steps, list(self._replies), summary, self._trace_file)

    def _inbox_item(self, ev: Event) -> InboxItem:
        """Turn a log event into what the model sees. The gate's reply surface
        picks the channel: 'chat', or the artifact type a channel handles."""
        channel = None
        if ev.reply_surface and ev.reply_surface != "chat":
            channel = next((c for c in self.channels.values() if ev.reply_surface in c.artifact_types), None)
        label = (channel.label(ev) if channel else None) or ev.payload.get("label") or _default_label(ev)
        meta = channel.meta(ev) if channel else {}
        ts = datetime.fromisoformat(ev.created_at).astimezone().strftime("%H:%M")
        return InboxItem(ev.event_id, ts, label, ev.text, channel.name if channel else None, meta)

    def _record_edit(self, path: str) -> None:
        """The agent's file writes go on the log as `edit` events on `file:<path>`."""
        artifact = f"file:{path}"
        self.log.ensure_artifact(artifact, "file")
        self.log.submit(artifact, actor_type="agent", actor_id=self.agent_id, surface="edit",
                        payload={"path": path})

    def _render_recall(self, exclude: set[str]) -> str:
        turns = self.log.recent(self.thread, limit=self.cfg.recall_messages, surfaces=("chat",), exclude=exclude)
        lines = []
        for ev in turns:
            who = "You" if ev.actor_type == "agent" and ev.actor_id == self.agent_id else "User"
            lines.append(f"{who}: {' '.join(ev.text.split())[:500]}")
        return "\n".join(lines)

    def _todo_signature(self) -> tuple:
        return tuple(t.title for t in self.ws.non_terminal_todos())

    def _deliver(self, message: str) -> None:
        self._replies.append(message)
        # The agent's own words go on the log too, so the conversation is
        # two-sided and recallable on later wakes. The gate's self-trigger rule
        # keeps it from waking the agent.
        self.log.submit(self.thread, actor_type="agent", actor_id=self.agent_id,
                        surface="chat", payload={"text": message})
        self._emit("reply", text=message)
        if self.on_reply:
            self.on_reply(message)

    def _emit(self, kind: str, **data) -> None:
        if kind == "channel_reply":
            self._channel_replies += 1
        if self._trace_file is not None:
            rec = {"ts": datetime.now().isoformat(timespec="seconds"), "kind": kind, **data}
            with self._trace_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec, default=str) + "\n")
        if self.on_event:
            self.on_event(kind, data)

    def _builtin_tools(self) -> ToolRegistry:
        reg, ws = ToolRegistry(), self.ws

        @reg.tool(group="Talk & lifecycle")
        def reply(message: str) -> str | None:
            """Send a message to the user (non-blocking)."""
            if self._suppress_chat:
                self._emit("reply_suppressed", text=str(message))
                return "[not delivered: this wake is routed to a channel; answer on the `→ Reply in:` line]"
            self._deliver(str(message))
            return None

        @reg.tool(group="Talk & lifecycle")
        def ask_user(question: str) -> str:
            """Ask the user a question; then mark_todo_blocked() the item that waits on it."""
            if self._suppress_chat:
                self._emit("reply_suppressed", text=str(question))
                return "[not delivered: this wake is routed to a channel; ask there instead]"
            self._deliver(str(question))
            return "Question sent. Park the item that depends on it with mark_todo_blocked(N, reason)."

        @reg.tool(group="Talk & lifecycle")
        def stop(summary: str = "") -> None:
            """Go dormant. Pass a one-sentence summary for your log."""
            raise StopRun(str(summary))

        @reg.tool(group="Todo list (your plan)")
        def add_todo(title: str) -> int:
            """Queue an active todo; returns its position number."""
            return ws.add_todo(title)

        @reg.tool(group="Todo list (your plan)")
        def mark_todo_done(position: int) -> str:
            """Mark a todo done."""
            return ws.set_todo_status(position, "done").render()

        @reg.tool(group="Todo list (your plan)")
        def mark_todo_blocked(position: int, reason: str = "") -> str:
            """Park a todo as waiting on the user."""
            return ws.set_todo_status(position, "blocked", reason).render()

        @reg.tool(group="Todo list (your plan)")
        def mark_todo_active(position: int) -> str:
            """Re-activate a blocked todo once the user has answered."""
            return ws.set_todo_status(position, "active").render()

        @reg.tool(group="Todo list (your plan)")
        def mark_todo_cancelled(position: int, reason: str = "") -> str:
            """Drop a todo as won't-do (only with the user's OK)."""
            return ws.set_todo_status(position, "cancelled", reason).render()

        @reg.tool(group="Todo list (your plan)")
        def list_todos() -> str:
            """Render the todo list."""
            return ws.render_todos()

        @reg.tool(group="Files (your files/ folder)")
        def read_file(path: str) -> str:
            """Read a file."""
            return ws.resolve(path).read_text(encoding="utf-8")

        @reg.tool(group="Files (your files/ folder)")
        def write_file(path: str, content: str) -> str:
            """Create or overwrite a file."""
            p = ws.resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            ws.note_write()
            self._record_edit(path)
            return f"wrote {path} ({len(content)} chars)"

        @reg.tool(group="Files (your files/ folder)")
        def edit_file(path: str, old: str, new: str) -> str:
            """Replace one exact occurrence of `old` with `new`."""
            p = ws.resolve(path)
            text = p.read_text(encoding="utf-8")
            n = text.count(old)
            if n != 1:
                return f"[edit refused] `old` occurs {n} times in {path}; it must occur exactly once"
            p.write_text(text.replace(old, new, 1), encoding="utf-8")
            ws.note_write()
            self._record_edit(path)
            return f"edited {path}"

        @reg.tool(group="Files (your files/ folder)")
        def append_file(path: str, content: str) -> str:
            """Append to a file (creates it if missing)."""
            p = ws.resolve(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a", encoding="utf-8") as f:
                f.write(content)
            ws.note_write()
            self._record_edit(path)
            return f"appended to {path}"

        @reg.tool(group="Files (your files/ folder)")
        def list_files(path: str = ".") -> str:
            """List a folder."""
            p = ws.resolve(path)
            if not p.exists():
                return f"(no such folder: {path})"
            entries = sorted(p.iterdir())
            return "\n".join(e.name + ("/" if e.is_dir() else "") for e in entries) or "(empty folder)"

        @reg.tool(group="Memory & web")
        def remember(fact: str) -> str:
            """Save a durable fact to memory (shown on every wake)."""
            ws.remember(fact)
            return "remembered"

        @reg.tool(group="Memory & web")
        def web_fetch(url: str, max_chars: int = 8000) -> str:
            """Fetch a URL and return its text (HTML stripped)."""
            req = urllib.request.Request(url, headers={"User-Agent": "plato/0.1"})
            with urllib.request.urlopen(req, timeout=20) as r:
                raw = r.read(2_000_000).decode(r.headers.get_content_charset() or "utf-8", "replace")
            text = re.sub(r"\s+", " ", _TAGS.sub(" ", raw)).strip()
            return text[:max_chars]

        return reg
