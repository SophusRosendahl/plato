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
`inbox.jsonl`, `files/`, `traces/`), so you can watch the agent think.

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

## Spike caveats

- The executor runs `exec()` in-process. It is **not a sandbox**. A
  subprocess/Docker/E2B executor with the same `run(code)` shape is next.
- No approval gate for side effects yet.
- No LLM history compaction, only a message-count trim.
