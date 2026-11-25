# Demos

Three small, runnable demos of the harness. Run them from the repo root.

| Demo | Run | What it shows |
|---|---|---|
| **Hello, always-on** | `python -m demos.hello_cli` | An agent running in the background of your terminal. It sleeps until you write; keep typing while it thinks and your follow-ups join the same wake. |
| **Agent in the browser** | `python -m demos.browser_chat` → http://127.0.0.1:8765 | Chat with reply-to and emoji reactions, all events on the log. ❓ and 👎 wake the agent; other reactions are only recorded. *Under the hood* shows the gate's verdict on every event. |
| **Comment section** | `python -m demos.comment_section` → http://127.0.0.1:8766 | A blog post with comments. Tag `@plato` and it reacts 👀 and answers in the thread; untagged comments don't wake it, but it reads them when tagged. The agent has no chat at all. |

## A real model, or the offline stand-in

Without configuration the demos run on an **offline stand-in model**: it drives
the harness like a model would (acknowledge, "think", answer where asked, stop),
but understands nothing. That's enough to see the mechanics and is what the tests
use. For real answers, point the demos at any OpenAI-compatible endpoint:

```bash
pip install -e '.[live]'
export PLATO_MODEL=accounts/fireworks/models/kimi-k2p6
export PLATO_BASE_URL=https://api.fireworks.ai/inference/v1
export PLATO_API_KEY=...
```

## How they're built

Each demo is a few dozen lines of wiring on top of the core blocks:

- `hello_cli`: `Agent` + `agent.serve()` in a thread + `Chat(on_message=print)`.
- `browser_chat`: the built-in `Chat` with `wake_on=("❓", "👎")`, and a tiny
  stdlib web server that turns clicks into events and reads the log back.
- `comment_section`: a `PageComments` channel defined in the demo itself, with one
  gate rule `Rule("comment", True, addressed=True)`, the `InReplyTo` and
  `RecentEvents` tail parts, and `comment()` / `react()` tools. It runs with
  `chat=False`.

The agent's workspace (memory, todo list, event log, traces) lands in
`demos/<name>/workspace/`; delete it to start fresh.
