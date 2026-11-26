"""An agent living in your browser: reply to its messages, react with emojis.

    python -m demos.browser_chat        # then open http://127.0.0.1:8765

Everything the page shows is read straight off the event log: messages,
replies (`parent_id`), reactions (`reaction` events). Reactions are recorded and
the agent sees them on its next wake, but only ❓ ("explain more") and 👎 ("try
again") wake it, because those are the rules this demo gives the chat channel.
The "under the hood" panel shows the gate's verdict on every event.
"""

from __future__ import annotations

import argparse
import threading
from collections import deque
from pathlib import Path

from demos._shared.models import describe, make_model, spend_limits
from demos._shared.web import App, event_json
from plato import Agent, Chat

IDENTITY = """You live in a chat window in someone's browser. Keep answers short and
friendly. When the user replies to one of your messages, the inbox shows which one;
answer with reply(message, reply_to=...). A ❓ reaction means "explain that in more
detail"; a 👎 means "that missed, try again differently". You may react to the
user's messages with react(message_id, emoji) when an emoji says it best."""


def build(workspace: str, model=None) -> tuple[Agent, App]:
    model = model or make_model()
    gates: deque[dict] = deque(maxlen=40)
    agent = Agent(workspace, model, **spend_limits(model), identity=IDENTITY,
                  chat=Chat(wake_on=("❓", "👎")),
                  on_event=lambda kind, data: gates.append(data) if kind == "gate" else None)
    app = App(Path(__file__).parent)

    @app.get("/api/state")
    def state(q):
        after = int(q.get("after", 0))
        return {"events": [event_json(e) for e in agent.log.events_after(after, [agent.thread])],
                "busy": agent.busy, "last_seq": agent.log.last_seq(), "gates": list(gates),
                "model": describe(agent.model)}

    @app.post("/api/send")
    def send(body):
        ev = agent.chat.send(str(body["text"]), reply_to=body.get("reply_to") or None)
        return {"id": ev.event_id}

    @app.post("/api/react")
    def react(body):
        ev = agent.chat.react(str(body["id"]), str(body["emoji"]))
        return {"id": ev.event_id}

    return agent, app


def main() -> None:
    p = argparse.ArgumentParser(prog="python -m demos.browser_chat")
    p.add_argument("--workspace", default="demos/browser_chat/workspace")
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()
    agent, app = build(args.workspace)
    threading.Thread(target=agent.serve, daemon=True).start()
    server = app.serve(port=args.port)
    print(f"plato browser chat · {describe(agent.model)}\n→ open http://127.0.0.1:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print()


if __name__ == "__main__":
    main()
