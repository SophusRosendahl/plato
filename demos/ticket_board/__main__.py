"""A ticket board where the agent is a teammate you assign work to. No chat.

    python -m demos.ticket_board        # then open http://127.0.0.1:8767

Create a ticket, describe it, assign it to plato. It picks the ticket up, works
it (writing deliverables as files it attaches), comments on progress, and moves
it to Review. Comment on its ticket to steer it; move it to Done when happy.

How it's wired, with nothing but plato's building blocks:

    events       every board action is an event on the ticket's artifact
                 `ticket:T-n`: created, edit (description), assigned, moved,
                 comment, attach
    subscription assigning a ticket to the agent subscribes it to that
                 ticket (`follow`); unassigning unsubscribes it. So comments
                 on its own tickets reach it, and no others do.
    gate rules   Rule("status", op="assigned", addressed=True)  -> wake
                 Rule("comment", actor_type="human")            -> wake
                 everything else (creating, editing, moving) is logged only
    tail         the ticket's current state, folded from its events (title,
                 status, description, attachments), plus its recent comments
    tools        update_ticket(ticket, comment, status, attach)
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

from demos._shared.models import describe, make_limits, make_model
from demos._shared.web import App, event_json
from plato import WITHHELD, Agent, Channel, Event, RecentEvents, Rule, ToolRegistry

HERE = Path(__file__).parent
STATUSES = ("todo", "in_progress", "review", "done")

GUIDANCE = """
--- how to work tickets ---
You're a teammate on a ticket board; people assign you tickets instead of chatting.
When you're assigned one:
1. Pick it up: update_ticket(id, comment="<one line on your approach>", status="in_progress").
2. Plan it in your todo list if it has several steps, then do the work. Put real
   deliverables in files (write_file('tickets/<id>/<name>.md', ...)) and attach
   them: update_ticket(id, attach='tickets/<id>/<name>.md').
3. Hand it back: update_ticket(id, comment="<what you did, what to check>", status="review").
   Never move a ticket to done yourself: the person who assigned it decides that.
If the ticket is unclear, ask in a comment and keep it in_progress. A new comment on
your ticket is feedback: act on it and reply in a comment.
"""


class TicketCard:
    """Tail part: the ticket's current state, folded from its events."""

    def render(self, channel: Tickets, item) -> str:
        t = channel.ticket(item.meta["ticket"])
        files = ", ".join(t["attachments"]) or "none"
        return (f"Ticket {t['id']} · status: {t['status']} · assigned to: {t['assignee'] or 'nobody'}\n"
                f"Title: {t['title']}\nDescription:\n{t['description'] or '(none)'}\nAttachments: {files}")


class Tickets(Channel):
    name = "tickets"
    artifact_types = ("ticket",)
    gate_rules = (
        Rule("status", True, op="assigned", addressed=True, reason="assigned"),
        Rule("comment", True, actor_type="human", reason="comment-on-my-ticket"),
    )
    tail_parts = (TicketCard(), RecentEvents(surfaces=("comment",), limit=6))
    guidance = GUIDANCE

    # ── the fold: a ticket's current state is derived from its events ────
    def ticket(self, tid: str) -> dict:
        t = {"id": tid, "title": "", "description": "", "status": "todo", "assignee": None, "attachments": []}
        for ev in self.log.recent(f"ticket:{tid}", limit=10**6):
            if ev.surface == "status" and ev.op == "created":
                t["title"], t["description"] = ev.payload.get("title", ""), ev.text
            elif ev.surface == "edit":
                t["description"] = ev.text
            elif ev.surface == "status" and ev.op == "assigned":
                t["assignee"] = ev.payload.get("assignee")
            elif ev.surface == "status" and ev.op == "moved":
                t["status"] = ev.payload.get("status", t["status"])
            elif ev.surface == "attach":
                t["attachments"].append(ev.payload.get("path"))
        return t

    def _next_id(self) -> str:
        made = [int(e.artifact_id.rsplit("-", 1)[1]) for e in self.log.events_after(0)
                if e.surface == "status" and e.op == "created" and e.artifact_id.startswith("ticket:T-")]
        return f"T-{max(made, default=0) + 1}"

    # ── what people do on the board ──────────────────────────────────────
    def create(self, title: str, description: str, author: str, assignee: str | None = None) -> str:
        tid = self._next_id()
        self.receive(f"ticket:{tid}", "ticket", "status", description, op="created", actor_id=author,
                     follow=False, ticket=tid, title=title)
        if assignee:
            self.assign(tid, assignee, author)
        return tid

    def assign(self, tid: str, assignee: str | None, author: str) -> Event:
        mine = assignee == self.agent.agent_id
        if mine:  # assignment = subscription: from now on the agent hears this ticket
            self.agent.follow(f"ticket:{tid}", "ticket")
        else:
            self.log.unsubscribe(self.agent.agent_id, f"ticket:{tid}")
        return self.receive(f"ticket:{tid}", "ticket", "status", self.ticket(tid)["title"],
                            op="assigned", actor_id=author, follow=False, ticket=tid, assignee=assignee,
                            **({"to": [assignee]} if assignee else {}))

    def edit(self, tid: str, description: str, author: str) -> Event:
        return self.receive(f"ticket:{tid}", "ticket", "edit", description, actor_id=author, follow=False, ticket=tid)

    def move(self, tid: str, status: str, author: str) -> Event:
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}")
        return self.receive(f"ticket:{tid}", "ticket", "status", f"moved to {status}", op="moved",
                            actor_id=author, follow=False, ticket=tid, status=status)

    def comment(self, tid: str, text: str, author: str) -> Event:
        return self.receive(f"ticket:{tid}", "ticket", "comment", text, actor_id=author, follow=False, ticket=tid)

    # ── what the agent can do ────────────────────────────────────────────
    def tools(self) -> ToolRegistry:
        reg = ToolRegistry()

        @reg.tool(group="Tickets")
        def update_ticket(ticket: str, comment: str | None = None, status: str | None = None,
                          attach: str | None = None) -> str:
            """Comment on a ticket, move it (todo/in_progress/review), and/or attach a file you wrote."""
            art, done = f"ticket:{ticket}", []
            if attach:
                if not self.ws.resolve(attach).is_file():  # a file in your files/ folder
                    return f"[not attached] no file {attach!r}; write it first"
                if self.act(art, "ticket", "attach", attach, ticket=ticket, path=attach) is None:
                    return WITHHELD
                done.append(f"attached {attach}")
            if comment:
                if self.act(art, "ticket", "comment", str(comment), ticket=ticket) is None:
                    return WITHHELD
                done.append("commented")
            if status:
                if status not in STATUSES[:3]:
                    return "[not moved] you can move tickets to todo, in_progress or review; people move them to done"
                self.act(art, "ticket", "status", f"moved to {status}", op="moved", ticket=ticket, status=status)
                done.append(f"moved to {status}")
            return ", ".join(done) or "nothing to do"

        @reg.tool(group="Tickets")
        def get_ticket(ticket: str) -> dict:
            """The ticket's current state (title, description, status, assignee, attachments)."""
            return self.ticket(ticket)

        return reg

    def label(self, event: Event) -> str:
        tid = event.payload.get("ticket")
        if event.op == "assigned":
            return f"Ticket {tid} was assigned to you (by {event.actor_id})"
        return f"New comment on {tid} (by {event.actor_id})"

    def meta(self, event: Event) -> dict:
        return {**super().meta(event), "ticket": event.payload.get("ticket")}

    def reply_hint(self, item) -> str:
        return f"update_ticket({item.meta['ticket']!r}, comment='...', status='review')"


def build(workspace: str, model=None) -> tuple[Agent, Tickets, App]:
    tickets = Tickets()
    agent = Agent(workspace, model or make_model(), limits=make_limits(), agent_id="plato", chat=False, channels=[tickets])
    app = App(HERE)
    me = "you"

    @app.get("/api/state")
    def state(q):
        events = [e for e in agent.log.events_after(int(q.get("after", 0))) if e.artifact_id.startswith("ticket:")]
        return {"events": [event_json(e) for e in events], "busy": agent.busy,
                "last_seq": agent.log.last_seq(), "model": describe(agent.model)}

    @app.post("/api/create")
    def create(body):
        title = str(body["title"]).strip() or "Untitled"
        return {"id": tickets.create(title, str(body.get("description", "")), me, body.get("assignee") or None)}

    @app.post("/api/assign")
    def assign(body):
        return {"id": tickets.assign(str(body["id"]), body.get("assignee") or None, me).event_id}

    @app.post("/api/edit")
    def edit(body):
        return {"id": tickets.edit(str(body["id"]), str(body["description"]), me).event_id}

    @app.post("/api/move")
    def move(body):
        return {"id": tickets.move(str(body["id"]), str(body["status"]), me).event_id}

    @app.post("/api/comment")
    def comment(body):
        return {"id": tickets.comment(str(body["id"]), str(body["text"]), me).event_id}

    @app.get("/api/file")
    def file(q):
        f = agent.ws.resolve(str(q["path"]))
        return {"path": q["path"], "content": f.read_text(encoding="utf-8") if f.is_file() else "(missing)"}

    return agent, tickets, app


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m demos.ticket_board")
    p.add_argument("--workspace", default="demos/ticket_board/workspace")
    p.add_argument("--port", type=int, default=8767)
    args = p.parse_args()
    agent, _, app = build(args.workspace)
    threading.Thread(target=agent.serve, daemon=True).start()
    server = app.serve(port=args.port)
    print(f"plato ticket board · {describe(agent.model)}\n→ open http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
