"""Chat: the agent's built-in conversation, implemented as an ordinary Channel.

Nothing in the harness special-cases chat. It is installed by default
(`Agent(chat=True)`) and uses exactly the interface any other channel uses, which
also makes it the reference example of writing one:

    artifact     one thread per agent, `thread:<agent_id>`
    gate rule    a `chat` event wakes the agent and is answered in chat,
                 whatever artifact it was typed on (e.g. a file's sidebar)
    tools        reply(message), ask_user(question)
    context      the recent conversation, both sides, on every wake: the
                 agent's own replies are events on the log, so nothing depends
                 on in-memory history surviving a restart
    notify       the harness can reach the person here (e.g. "I'm stuck")
    answer_only_when_asked
                 on a wake where nothing arrived by chat (say, only a comment),
                 reply() is held back: the answer belongs where it was asked
"""

from __future__ import annotations

from collections.abc import Callable

from plato.channels import WITHHELD, Channel
from plato.events import Event
from plato.should_respond import Rule
from plato.tools import ToolRegistry
from plato.workspace import InboxItem


class Chat(Channel):
    name = "chat"
    artifact_types = ("thread",)
    gate_rules = (Rule("chat", True, answer="chat", reason="chat-message"),)
    answer_only_when_asked = True

    def __init__(self, recall: int = 12, on_message: Callable[[str], None] | None = None) -> None:
        self.recall = recall
        self.on_message = on_message

    def bind(self, agent) -> None:
        super().bind(agent)
        self.thread = f"thread:{agent.agent_id}"
        agent.follow(self.thread, "thread")

    def send(self, text: str, author: str | None = None) -> Event:
        """A person's chat message (call this from your UI)."""
        return self.receive(self.thread, "thread", "chat", text, actor_id=author)

    def say(self, text: str) -> Event | None:
        ev = self.act(self.thread, "thread", "chat", text)
        if ev is not None and self.on_message:
            self.on_message(text)
        return ev

    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Talk & lifecycle")
        def reply(message: str) -> str | None:
            """Send a message to the user (non-blocking)."""
            return None if self.say(str(message)) is not None else WITHHELD

        @reg.tool(group="Talk & lifecycle")
        def ask_user(question: str) -> str:
            """Ask the user a question; then mark_todo_blocked() the item that waits on it."""
            if self.say(str(question)) is None:
                return WITHHELD
            return "Question sent. Park the item that depends on it with mark_todo_blocked(N, reason)."

        return reg

    def label(self, event: Event) -> str:
        return "User says"

    def reply_hint(self, item: InboxItem) -> str:
        return "reply(message)"

    def context(self, exclude: set[str]) -> str:
        turns = self.log.recent(self.thread, limit=self.recall, surfaces=("chat",), exclude=exclude)
        if not turns:
            return ""
        lines = []
        for ev in turns:
            me = ev.actor_type == "agent" and ev.actor_id == self.agent.agent_id
            lines.append(f"{'You' if me else 'User'}: {' '.join(ev.text.split())[:500]}")
        return "--- recent conversation ---\n" + "\n".join(lines)

    def notify(self, text: str) -> bool:
        # The harness speaking for itself is never held back.
        self.agent.follow(self.thread, "thread")
        self.log.submit(self.thread, actor_type="agent", actor_id=self.agent.agent_id,
                        surface="chat", payload={"text": text})
        self.agent._emit("channel_reply", channel=self.name, artifact=self.thread, text=text)
        if self.on_message:
            self.on_message(text)
        return True
