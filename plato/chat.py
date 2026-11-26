"""Chat: the agent's built-in conversation, implemented as an ordinary Channel.

Nothing in the harness special-cases chat. It is installed by default
(`Agent(chat=True)`) and uses exactly the interface any other channel uses, which
also makes it the reference example of writing one:

    artifact     one thread per agent, `thread:<agent_id>`
    events       `chat` messages (a reply carries `parent_id`), and `reaction`
                 events (`op` is the emoji, `parent_id` the message reacted to)
    gate rule    a `chat` message wakes the agent and is answered in chat,
                 whatever artifact it was typed on. Reactions have no rule, so
                 they never wake it; pass `wake_on=("❓",)` to let some do.
    tools        reply(message, reply_to=None), ask_user(question), react(...)
    tail         a reply arrives with the message it replies to (InReplyTo)
    context      the recent conversation, both sides, with reactions, on every
                 wake: the agent's own replies are events on the log, so nothing
                 depends on in-memory history surviving a restart
    notify       the harness can reach the person here (e.g. "I'm stuck")
    answer_only_when_asked
                 on a wake where nothing arrived by chat (say, only a comment),
                 reply() is held back: the answer belongs where it was asked
"""

from __future__ import annotations

from collections.abc import Callable

from plato.channels import WITHHELD, Channel
from plato.events import Event
from plato.executor import uncaptured
from plato.should_respond import Rule
from plato.tails import InReplyTo
from plato.tools import ToolRegistry
from plato.workspace import InboxItem


class Chat(Channel):
    name = "chat"
    artifact_types = ("thread",)
    tail_parts = (InReplyTo(),)
    answer_only_when_asked = True

    def __init__(
        self,
        recall: int = 12,
        on_message: Callable[[str], None] | None = None,
        wake_on: tuple[str, ...] = (),
    ) -> None:
        self.recall = recall
        self.on_message = on_message
        # Reactions that should wake the agent (e.g. "❓" = "explain more").
        self.gate_rules = (
            Rule("chat", True, answer="chat", reason="chat-message"),
            *(Rule("reaction", True, op=emoji, answer="chat", reason=f"reaction-{emoji}") for emoji in wake_on),
        )

    def bind(self, agent) -> None:
        super().bind(agent)
        self.thread = f"thread:{agent.agent_id}"
        agent.follow(self.thread, "thread")

    # ── the person's side (call from your UI) ───────────────────────────
    def send(self, text: str, author: str | None = None, reply_to: str | None = None) -> Event:
        """A person's chat message, optionally a reply to an earlier message."""
        return self.receive(self.thread, "thread", "chat", text, actor_id=author, parent_id=reply_to)

    def react(self, event_id: str, emoji: str, author: str | None = None) -> Event:
        """A person reacts to a message with an emoji."""
        return self.receive(self.thread, "thread", "reaction", emoji, actor_id=author, op=emoji,
                            parent_id=event_id)

    # ── the agent's side ─────────────────────────────────────────────────
    def say(self, text: str, reply_to: str | None = None) -> Event | None:
        ev = self.act(self.thread, "thread", "chat", text, parent_id=reply_to)
        if ev is not None and self.on_message:
            with uncaptured():
                self.on_message(text)
        return ev

    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Talk & lifecycle")
        def reply(message: str, reply_to: str | None = None) -> str | None:
            """Send a message to the user (non-blocking); reply_to=<id> answers a specific message."""
            return None if self.say(str(message), reply_to) is not None else WITHHELD

        @reg.tool(group="Talk & lifecycle")
        def ask_user(question: str) -> str:
            """Ask the user a question; then mark_todo_blocked() the item that waits on it."""
            if self.say(str(question)) is None:
                return WITHHELD
            return "Question sent. Park the item that depends on it with mark_todo_blocked(N, reason)."

        @reg.tool(group="Talk & lifecycle")
        def react(message_id: str, emoji: str) -> str | None:
            """React to a chat message with an emoji (a light acknowledgement)."""
            ev = self.act(self.thread, "thread", "reaction", str(emoji), op=str(emoji), parent_id=message_id)
            return None if ev is not None else WITHHELD

        return reg

    def label(self, event: Event) -> str:
        if event.surface == "reaction":
            return "User reacted"
        return "User replies" if event.parent_id else "User says"

    def reply_hint(self, item: InboxItem) -> str:
        return f"reply(message, reply_to={item.id!r})" if item.meta.get("parent_id") else "reply(message)"

    def context(self, exclude: set[str]) -> str:
        turns = self.log.recent(self.thread, limit=self.recall, surfaces=("chat",), exclude=exclude)
        if not turns:
            return ""
        reactions: dict[str, list[str]] = {}
        for r in self.log.recent(self.thread, limit=10**6, surfaces=("reaction",), exclude=exclude):
            who = "you" if r.actor_type == "agent" else r.actor_id
            reactions.setdefault(r.parent_id or "", []).append(f"{r.op} {who}")
        lines = []
        for ev in turns:
            me = ev.actor_type == "agent" and ev.actor_id == self.agent.agent_id
            line = f"{'You' if me else 'User'} ({ev.event_id}): {' '.join(ev.text.split())[:500]}"
            if ev.event_id in reactions:
                line += f"  [{', '.join(reactions[ev.event_id])}]"
            lines.append(line)
        return "--- recent conversation ---\n" + "\n".join(lines)

    def notify(self, text: str) -> bool:
        # The harness speaking for itself is never held back.
        self.agent.follow(self.thread, "thread")
        self.log.submit(self.thread, actor_type="agent", actor_id=self.agent.agent_id,
                        surface="chat", payload={"text": text})
        self.agent._emit("channel_reply", channel=self.name, artifact=self.thread, text=text)
        if self.on_message:
            with uncaptured():
                self.on_message(text)
        return True
