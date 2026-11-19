# plato (spike)

An event-driven, code-first agent harness. The agent is dormant until something
lands in its inbox. It then wakes, acts by writing Python (one `run` tool; tools
are plain functions), plans into a todo list, and may only go dormant once every
todo is done, blocked on the user, or cancelled.

Smart model, dumb harness: the loop does lifecycle plumbing plus a few hard
guarantees, and nothing else.

| Guard | What it prevents |
|---|---|
| Open-todo stop guard | `stop()` with active todos is refused, so work isn't silently abandoned (bounded retries) |
| Silent-stop guard | Stopping on step 1 with new inbox items, no reply and no plan |
| Budget notice | Being cut off mid-thought by a step limit the model can't see |
| Circuit breaker | Repeated no-progress wakes burning budget forever: parks the items, tells the user once, caps later wakes |

State is a folder of plain files (`memory.md`, `todo.md`, `daily_log.md`,
`files/`, `traces/`) plus one SQLite event log, so you can watch the agent think.

## The event log and the `should_respond` gate

Everything that happens is an immutable event on one append-only log
(`events.db`): a person's chat message, a comment, an edit, a file landing, and
the agent's own replies, comments and edits. Events attach to **artifacts** (a
chat thread, a document, an email conversation), and participants **subscribe**
to artifacts.

```
person/app ──submit──▶ event log ──dispatch──▶ should_respond(participant, event)
                                                   │ no: logged, costs nothing
                                                   ▼ yes: inbox item (+ reply surface)
                                               agent wakes
```

`should_respond` is a cheap, deterministic, no-LLM pre-filter, run for every
subscriber before a wake is spent:

- never your own event
- only if subscribed and visible
- a chat message is answered in chat
- a comment, mention or inbound message is answered on that artifact's channel
- a person's edit doesn't wake (the agent isn't a reviewer of work in progress)
- bookkeeping (`create`, `rename`, …) doesn't wake
- an unknown surface never wakes

Every decision carries a reason tag and lands in the trace. The branches that
vary by product are parameters on `Policy`. Whether to actually respond, and
how, stays the model's judgment.

Because the agent's own words are on the log, the conversation is two-sided:
every wake shows the recent conversation, so nothing depends on in-memory
history surviving a restart. Several agents can share one log; each follows
the artifacts it cares about.

## Try it

```bash
pip install -e '.[live,dev]'
pytest                                  # guards exercised with a scripted model, no key needed

export PLATO_MODEL=accounts/fireworks/models/kimi-k2p6
export PLATO_BASE_URL=https://api.fireworks.ai/inference/v1
export PLATO_API_KEY=...
plato -v chat                        # talk to it
plato -v watch ./inbox-folder        # wake it when files land in a folder
```

Any OpenAI-compatible endpoint works.

## Add a tool

```python
from plato import Agent, OpenAICompatModel, tool

@tool(group="Weather")
def forecast(city: str, days: int = 1) -> str:
    """Get the forecast for a city."""
    ...

agent = Agent("./workspace", OpenAICompatModel("gpt-4.1"))
agent.send("Do I need an umbrella in Oslo tomorrow?")
agent.wake()
```

The signature and first docstring line become the tool's entry in the catalogue
the model sees. There is no schema to keep in sync.

## Channels: inputs beyond chat

Chat is one way in. A comment on a document, an email, a webhook: each is a
**channel**. An event on a channel brings three things into the wake beyond
its own text:

- **A reply route**: a `→ Reply in: comment(...)` line naming the call that
  answers it. When a whole wake is routed to channels, chat `reply()` is
  withheld by the harness, not just discouraged by the prompt.
- **A tail of context**, bounded, built at wake time: for a comment, the quoted
  passage, the recent thread ("most recent 6 of 14 comments"), and a reference
  to the document (path + size; the body is read on demand).
- **Channel rules**: a short prompt for how to behave there, rendered once.

```python
from plato import Agent, CommentsChannel, OpenAICompatModel

comments = CommentsChannel()
agent = Agent("./workspace", OpenAICompatModel("gpt-4.1"), channels=[comments])

# your app calls this when a person comments
comments.post("launch-plan.md", "Design slipped a week. Move the date.",
              quote="We ship on March 3.")
agent.wake()   # agent answers with comment(...) in that thread and edits the file
```

A new channel is a small subclass of `Channel` overriding `tools()`,
`reply_hint()`, `tail()` and `rules()`; see `plato/channels.py`.

## Spike caveats

- The executor runs `exec()` in-process. It is **not a sandbox**. A
  subprocess/Docker/E2B executor with the same `run(code)` shape is next.
- No approval gate for side effects yet.
- No LLM history compaction, only a message-count trim.
