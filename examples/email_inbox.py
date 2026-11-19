"""Recipe: an email inbox, built only from plato's building blocks.

Each correspondent is an artifact `email:<address>`; an inbound mail is a
`message` event on it, and the agent's reply is a `message` event by the agent.
So the "thread" tail is simply the recent messages with that person.

`send` is whatever actually delivers mail (SMTP, an API…); here it's injectable
so the recipe stays runnable without a mail server.
"""

from __future__ import annotations

from collections.abc import Callable

from plato import Channel, Event, PayloadField, RecentEvents, Rule, ToolRegistry

GUIDANCE = """
--- how to answer email ---
Reply with send_email() to the address on the `→ Reply in:` line. Write it like an email: complete, polite, no chat shorthand. Only reply when the mail needs an answer from you.
"""


class EmailInbox(Channel):
    name = "email"
    artifact_types = ("email",)
    gate_rules = (Rule("message", True, actor_type="human", reason="inbound-email"),)
    tail_parts = (
        PayloadField("subject", "Subject: {value}"),
        RecentEvents(surfaces=("message",), limit=4, max_chars=400),
    )
    guidance = GUIDANCE

    def __init__(self, send: Callable[[str, str, str], None] | None = None) -> None:
        self.sent: list[tuple[str, str, str]] = []
        self._send = send or (lambda to, subject, body: self.sent.append((to, subject, body)))

    def deliver(self, sender: str, subject: str, body: str) -> Event:
        """An email arrived (call this from your mail webhook)."""
        return self.receive(f"email:{sender}", "email", "message", body, actor_id=sender,
                            sender=sender, subject=subject)

    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Email")
        def send_email(to: str, subject: str, body: str) -> str:
            """Send an email."""
            self._send(to, subject, body)
            self.act(f"email:{to}", "email", "message", body, subject=subject)
            return f"sent to {to}"

        return reg

    def label(self, event: Event) -> str:
        return f"Email from {event.payload.get('sender')}"

    def reply_hint(self, item) -> str:
        m = item.meta
        return f"send_email(to={m['sender']!r}, subject={'Re: ' + m['subject']!r}, body='...')"
