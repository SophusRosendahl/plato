# Demos

Four small, runnable demos of the harness. Run them from the repo root.

| Demo | Run | What it shows |
|---|---|---|
| **Hello, always-on** | `python -m demos.hello_cli` | An agent running in the background of your terminal. It sleeps until you write; keep typing while it thinks and your follow-ups join the same wake. |
| **Agent in the browser** | `python -m demos.browser_chat` → http://127.0.0.1:8765 | Chat with reply-to and emoji reactions, all events on the log. ❓ and 👎 wake the agent; other reactions are only recorded. *Under the hood* shows the gate's verdict on every event. |
| **Ticket board** (no chat) | `python -m demos.ticket_board` → http://127.0.0.1:8767 | Create a ticket, describe it, assign it to plato. It picks it up, works it (deliverables as attached files), comments, and moves it to Review. Comment on its ticket to steer it. |
| **Comment section** | `python -m demos.comment_section` → http://127.0.0.1:8766 | A blog post with comments. Tag `@plato` and it reacts 👀 and answers in the thread; untagged comments don't wake it, but it reads them when tagged. The agent has no chat at all. |

## A real model, or the offline stand-in

Without configuration the demos run on an **offline stand-in model**: it drives
the harness like a model would (acknowledge, "think", answer where asked, stop),
but understands nothing. That's enough to see the mechanics and is what the tests
use. For real answers, set `PLATO_MODEL` to any model `plato.connect` supports:

```bash
pip install -e '.[live]'
export PLATO_MODEL=anthropic:claude-opus-5-5   # + ANTHROPIC_API_KEY
# or openai:gpt-5, gemini:..., openrouter:..., ollama:llama3.3 (local) — see plato/connect.py
```

With a real model, every demo spends from **one shared budget**: 100 a month and
5 a day, tracked in `~/.plato/usage.json` across all demos and runs. Change them
with `PLATO_MONTHLY_BUDGET` / `PLATO_DAILY_BUDGET` (0 = no cap) and
`PLATO_USAGE_FILE`. Cost is computed from `PLATO_PRICE_PER_MTOK="input,output"`
(per million tokens). The default, 2.5 and 15, is deliberately pessimistic, so the
meter over-counts on cheaper models and stops before real spend reaches the cap.
Over a cap the agent stops cleanly, says so once, and new messages wait.

## How they're built

Each demo is a few dozen lines of wiring on top of the core blocks:

- `hello_cli`: `Agent` + `agent.serve()` in a thread + `Chat(on_message=print)`.
- `browser_chat`: the built-in `Chat` with `wake_on=("❓", "👎")`, and a tiny
  stdlib web server that turns clicks into events and reads the log back.
- `comment_section`: a `PageComments` channel defined in the demo itself, with one
  gate rule `Rule("comment", True, addressed=True)`, the `InReplyTo` and
  `RecentEvents` tail parts, and `comment()` / `react()` tools. It runs with
  `chat=False`.
- `ticket_board`: a `Tickets` channel with no chat at all; see *Wiring the ticket
  board* below.

The agent's workspace (memory, todo list, event log, traces) lands in
`demos/<name>/workspace/`; delete it to start fresh.

## Wiring the ticket board: an agent without chat

The ticket board is the clearest picture of how plato is meant to be used: the
agent is a participant in *your* system, reacting to the same events your app
already has, and answering where the work lives.

```
   board UI ──▶ Tickets.create / assign / edit / move / comment
                      │   each is one event on artifact `ticket:T-n`
                      ▼
               event log ──▶ dispatcher ──▶ should_respond (Tickets.gate_rules)
                                               │ created / edit / moved → logged only
                                               │ assigned (to plato)    → wake
                                               │ comment on its ticket  → wake
                                               ▼
                              wake: inbox item + TicketCard tail + guidance
                                               │
                              agent code: write_file(...), update_ticket(...)
                                               │   attach / comment / moved events
                                               ▼
                                     event log ──▶ board UI (same fold)
```

The whole integration is one `Channel` subclass in
[`ticket_board/__main__.py`](ticket_board/__main__.py):

1. **Name the events.** Each board action becomes an event on the ticket's
   artifact: `status/created`, `edit`, `status/assigned`, `status/moved`,
   `comment`, `attach`. The board UI and the agent both read state by folding
   those events (`Tickets.ticket()` in Python, `fold()` in the page).
2. **Make assignment a subscription.** `assign()` calls `agent.follow(...)` when
   the assignee is the agent and unsubscribes otherwise. Subscriptions decide
   what the agent can ever hear about, so it never sees other people's tickets.
3. **Write the gate rules.** Two rules decide what's worth waking for:
   ```python
   gate_rules = (
       Rule("status", True, op="assigned", addressed=True, reason="assigned"),
       Rule("comment", True, actor_type="human", reason="comment-on-my-ticket"),
   )
   ```
   Anything without a rule (creating, editing, moving) is logged and never wakes it.
4. **Give it context, not a dump.** `tail_parts = (TicketCard(), RecentEvents(...))`:
   the ticket's current state, folded from its events, plus the recent comments.
5. **Give it the verbs.** `update_ticket(ticket, comment, status, attach)` records
   the agent's own events; the board shows them because they're on the same log.
   Policy lives in the tool: it can't move a ticket to Done, and it can only attach
   files it actually wrote.
6. **Run it without chat:** `Agent(ws, model, chat=False, channels=[Tickets()])`.

Swap the tickets for GitHub issues, Linear, Jira or your own database and the
recipe is the same: map actions to events, decide what wakes it, give it context
and verbs.
