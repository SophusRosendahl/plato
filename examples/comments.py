"""Recipe: comment threads on files, built only from plato's building blocks.

    events       a comment is a `comment` event on the file's artifact `file:<path>`
    gate rule    a person's comment wakes the agent; it answers on the artifact
    tail         the quoted passage + the recent thread + a pointer to the file
    tool         comment(path, text, parent_id) records the agent's reply event
    guidance     how to behave in a comment thread

Usage:

    comments = Comments()
    agent = Agent("./ws", model, channels=[comments])
    comments.post("plan.md", "Move the date?", quote="We ship on March 3.")  # from your app
    agent.wake()
"""

from __future__ import annotations

from plato import Channel, Event, FileReference, PayloadField, RecentEvents, Rule, ToolRegistry

GUIDANCE = """
--- how to answer comments ---
Answer a comment with comment(), in its thread (pass the parent_id shown on its `→ Reply in:` line). Keep comments to a sentence or two.
- It asks for a change → leave a short comment saying what you're doing, then make the change with edit_file/write_file. Actually do the work; ask instead only if you would have to guess at something material.
- It asks a question → answer in the comment. If the answer is long or worth keeping, put it in the file and leave a short comment pointing to it.
- It's an acknowledgement ("thanks") → reply briefly.
Never paste file content into a comment. Don't resolve the thread for the user.
"""


class Comments(Channel):
    name = "comments"
    artifact_types = ("file",)
    gate_rules = (Rule("comment", True, reason="comment"),)
    guidance = GUIDANCE

    def __init__(self, thread_limit: int = 6, max_chars: int = 300) -> None:
        self.tail_parts = (
            PayloadField("quote", "The comment refers to this passage:\n> {value}"),
            RecentEvents(surfaces=("comment",), limit=thread_limit, max_chars=max_chars),
            FileReference("path"),
        )

    def post(self, path: str, text: str, *, quote: str | None = None,
             parent_id: str | None = None, author: str | None = None) -> Event:
        """A person comments on a file (call this from your app)."""
        return self.receive(f"file:{path}", "file", "comment", text, actor_id=author,
                            parent_id=parent_id, path=path, quote=quote)

    def thread(self, path: str) -> list[Event]:
        return self.log.recent(f"file:{path}", limit=10**6, surfaces=("comment",))

    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Comments")
        def comment(path: str, text: str, parent_id: str | None = None) -> str:
            """Post a comment on a file; pass parent_id to reply inside a thread."""
            ev = self.act(f"file:{path}", "file", "comment", str(text), parent_id=parent_id, path=path)
            return f"commented on {path} ({ev.event_id})"

        return reg

    def label(self, event: Event) -> str:
        return f"Comment on {event.payload.get('path')}"

    def reply_hint(self, item) -> str:
        m = item.meta
        return f"comment(path={m['path']!r}, text='...', parent_id={m['thread_root']!r})"
