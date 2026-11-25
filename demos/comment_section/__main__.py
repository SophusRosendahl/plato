"""A website with a comment section. Tag @plato and the agent joins the thread.

    python -m demos.comment_section     # then open http://127.0.0.1:8766

The comment section is a Channel built in this file from plato's public blocks.
Plato's core has no idea what a "page comment" is:

    events     `comment` events on the artifact `page:<slug>`, `reaction` events
    gate rule  Rule("comment", True, addressed=True): only a comment that tags
               @plato wakes the agent. Untagged comments are logged and become
               context when it *is* tagged.
    tail       the comment it replies to, the recent thread, a pointer to the article
    tools      comment(text, reply_to), react(comment_id, emoji)

The agent has no chat at all (`chat=False`): it lives only in the comment section.
"""

from __future__ import annotations

import argparse
import re
import shutil
import threading
from collections import deque
from pathlib import Path

from demos._shared.models import describe, make_model
from demos._shared.web import App, event_json
from plato import WITHHELD, Agent, Channel, Event, InReplyTo, PayloadField, RecentEvents, Rule, ToolRegistry

HERE = Path(__file__).parent
MENTION = re.compile(r"@([A-Za-z0-9_]+)")

GUIDANCE = """
--- you were tagged in the comment section ---
You're the assistant on a cooking blog, called in by an @mention in its comments.
1. First acknowledge the comment that tagged you: react(<its id>, '👀').
2. Then answer in its thread with comment(text, reply_to=...) as shown on the
   `→ Reply in:` line. A few friendly sentences; this is a comment, not an essay.
Questions about the recipe: read_file('article.md') and answer from it, and say
so when the article doesn't cover something. Other comments in the thread are
context, not requests; answer only what you were asked.
"""


class PageComments(Channel):
    name = "comments"
    artifact_types = ("page",)
    gate_rules = (Rule("comment", True, addressed=True, reason="mentioned"),)
    guidance = GUIDANCE
    tail_parts = (
        InReplyTo(),
        RecentEvents(surfaces=("comment",), limit=8),
        PayloadField("article", "The article on this page is in the file {value}."),
    )

    def __init__(self, slug: str, article: str) -> None:
        self.page, self.article = f"page:{slug}", article

    # ── the website's side ───────────────────────────────────────────────
    def post(self, author: str, text: str, reply_to: str | None = None) -> Event:
        handles = {h.lower() for h in MENTION.findall(text)}
        to = [self.agent.agent_id] if self.agent.agent_id in handles else sorted(handles)
        return self.receive(self.page, "page", "comment", text, actor_id=author, parent_id=reply_to,
                            article=self.article, **({"to": to} if to else {}))

    def react(self, event_id: str, emoji: str, author: str) -> Event:
        return self.receive(self.page, "page", "reaction", emoji, actor_id=author, op=emoji,
                            parent_id=event_id)

    # ── the agent's side ─────────────────────────────────────────────────
    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Comment section")
        def comment(text: str, reply_to: str | None = None) -> str:
            """Post a comment; reply_to=<thread id> answers inside a thread."""
            ev = self.act(self.page, "page", "comment", str(text), parent_id=reply_to)
            return f"posted ({ev.event_id})" if ev else WITHHELD

        @reg.tool(group="Comment section")
        def react(comment_id: str, emoji: str) -> str:
            """React to a comment with an emoji."""
            ev = self.act(self.page, "page", "reaction", str(emoji), op=str(emoji), parent_id=comment_id)
            return "reacted" if ev else WITHHELD

        return reg

    def label(self, event: Event) -> str:
        return f"{event.actor_id} tagged you (comment {event.event_id})"

    def reply_hint(self, item) -> str:
        return f"comment(text='...', reply_to={item.meta['thread_root']!r})"


def build(workspace: str, model=None) -> tuple[Agent, PageComments, App]:
    comments = PageComments("sourdough-for-impatient-people", "article.md")
    woke: deque[str] = deque(maxlen=500)

    def on_event(kind: str, data: dict) -> None:
        if kind == "gate" and data["respond"]:
            woke.append(data["event"])

    agent = Agent(workspace, model or make_model(), agent_id="plato", chat=False,
                  channels=[comments], on_event=on_event)
    shutil.copy(HERE / "article.md", agent.ws.files / "article.md")
    app = App(HERE)

    @app.get("/api/state")
    def state(q):
        events = agent.log.events_after(int(q.get("after", 0)), [comments.page])
        return {"events": [event_json(e) for e in events], "busy": agent.busy,
                "last_seq": agent.log.last_seq(), "woke": list(woke), "model": describe(agent.model)}

    @app.get("/api/article")
    def article(_):
        return {"markdown": (HERE / "article.md").read_text(encoding="utf-8")}

    @app.post("/api/comment")
    def post_comment(body):
        author = str(body.get("author") or "guest").strip()[:30] or "guest"
        return {"id": comments.post(author, str(body["text"]), body.get("reply_to") or None).event_id}

    @app.post("/api/react")
    def react(body):
        return {"id": comments.react(str(body["id"]), str(body["emoji"]), str(body.get("author") or "guest")).event_id}

    return agent, comments, app


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m demos.comment_section")
    p.add_argument("--workspace", default="demos/comment_section/workspace")
    p.add_argument("--port", type=int, default=8766)
    args = p.parse_args()
    agent, _, app = build(args.workspace)
    threading.Thread(target=agent.serve, daemon=True).start()
    server = app.serve(port=args.port)
    print(f"plato comment section · {describe(agent.model)}\n→ open http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
