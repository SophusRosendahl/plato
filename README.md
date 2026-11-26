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

## Demos

```bash
python -m demos.hello_cli         # an always-on agent in your terminal; type while it thinks
python -m demos.browser_chat      # chat in the browser with reply-to and emoji reactions
python -m demos.comment_section   # a blog's comment section; tag @plato and it answers
```

They run without an API key on an offline stand-in model; set `PLATO_MODEL` for
real answers. See [`demos/README.md`](demos/README.md).

## Building blocks

Plato's core knows nothing about comments, email, tickets or any other product
concept. It gives you five blocks, and everything else is built from them.

| Block | What it is |
|---|---|
| **Event log** (`EventLog`) | One append-only SQLite log. Every action by every actor (people, systems, the agent itself) is an immutable event on an **artifact** (anything events attach to: a file, an email thread, a ticket). A **surface** is a free-form string naming the kind of event. Participants **subscribe** to artifacts. |
| **Gate** (`should_respond`, `Policy`, `Rule`) | A cheap, deterministic filter (no LLM) run for every subscriber before a wake is spent. Universal checks are fixed: not your own event, subscribed, visible, addressed to you. Then your ordered `Rule`s decide by surface, op and actor type whether to wake and where the answer goes. No matching rule means no wake. The gate has no built-in rules at all. Every decision carries a reason, written to the trace. |
| **Inbox** | Derived from the log: the dispatcher turns gate yeses into inbox items. A wake marks done exactly what it showed the agent. |
| **Channel** (`Channel`) | Every way in and out, chat included. Owns one or more artifact types and contributes gate rules, tools, a `→ Reply in:` hint, a context tail, guidance, an every-wake `context()` section, and `notify()`. A channel marked `answer_only_when_asked` has its actions held back on a wake where nothing arrived through it, so answers land where the question was asked. |
| **Tail parts** (`RecentEvents`, `PayloadField`, `FileReference`) | Composable, bounded context an event brings into the wake, such as the recent thread, a quoted passage, or a pointer to a file whose body is read on demand. |

```
person / app / webhook ──submit──▶ event log ──dispatch──▶ gate (your rules)
                                                              │ no: logged, costs nothing
                                                              ▼ yes
                                     inbox ──▶ wake: route to channel + tail + guidance
```

The agent's own replies and edits are on the log too, so each wake shows the
recent conversation from both sides, and several agents can share one log (each
with its own rules).

### Chat is just another channel

Chat is built in, since every agent needs somewhere to talk, but the harness
doesn't special-case it. `plato/chat.py` is an ordinary `Channel` and the
reference example for writing one:

- one `thread` artifact
- `Rule("chat", True, answer="chat")`
- `reply()` and `ask_user()` as its tools
- the recent conversation as its every-wake `context()`
- `notify()` for messages from the harness itself
- `answer_only_when_asked`, so a comment-only wake can't leak a chat reply

```python
Agent(ws, model)                                   # chat installed by default
Agent(ws, model, chat=Chat(recall=4, on_message=print))
Agent(ws, model, channels=[Tickets()], chat=False) # an agent that only works tickets
```

## Try it

```bash
pip install -e '.[live,dev]'
pytest                                  # guards exercised with a scripted model, no key needed

export PLATO_MODEL=anthropic:claude-opus-5-5   # and ANTHROPIC_API_KEY
plato -v chat                           # talk to it
plato -v watch ./inbox-folder           # wake it when files land in a folder
```

## Connect a model

One string, `provider:model`, through two standard interfaces:

| Interface | Providers |
|---|---|
| **Anthropic Messages** (official SDK) | `anthropic:` (Claude). On Bedrock / Vertex AI / Foundry, pass that platform's client: `AnthropicModel(client=...)` |
| **OpenAI Chat Completions** | `openai:` `gemini:` `mistral:` `deepseek:` `xai:` `groq:` `together:` `fireworks:` `openrouter:` · a `litellm:` proxy (100+ providers) · local `ollama:` `lmstudio:` `vllm:` · or any base URL: `http://host:port/v1:model` |

```python
from plato import Agent, connect

Agent("./ws", connect("anthropic:claude-opus-5-5"))
Agent("./ws", connect("openrouter:meta-llama/llama-4-maverick"))
Agent("./ws", connect("ollama:llama3.3"))      # local, no key
```

Each preset reads its usual key (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`GEMINI_API_KEY`, …); `PLATO_API_KEY` / `PLATO_BASE_URL` override. A model is just
an object with `complete(messages, tool) -> ModelTurn`, so anything else is a
small adapter away.

The Claude adapter replays Claude's thinking blocks unchanged and uses
`tool_choice: auto` with a strict schema, because current Claude models reject
forced tool choice. Server-side refusal fallbacks are on by default
(`fallbacks="default"`; pass `fallbacks=None` to turn them off).

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

## Recipes: building comments, email, …

`examples/` builds real input types out of those blocks. `examples/comments.py`
is the whole comments feature in about 60 lines:

```python
class Comments(Channel):
    name = "comments"
    artifact_types = ("file",)                         # comments attach to files
    gate_rules = (Rule("comment", True),)               # a person's comment wakes the agent
    guidance = GUIDANCE                                 # how to behave in a thread

    def __init__(self):
        self.tail_parts = (
            PayloadField("quote", "The comment refers to this passage:\n> {value}"),
            RecentEvents(surfaces=("comment",), limit=6),  # the thread so far
            FileReference("path"),                         # the file, read on demand
        )

    def post(self, path, text, *, quote=None, parent_id=None):    # called by your app
        return self.receive(f"file:{path}", "file", "comment", text,
                            parent_id=parent_id, path=path, quote=quote)

    def tools(self):                                    # comment(path, text, parent_id)
        ...                                             # → self.act(...) records the reply

    def reply_hint(self, item):
        return f"comment(path={item.meta['path']!r}, text='...', parent_id={item.meta['thread_root']!r})"
```

```python
comments = Comments()
agent = Agent("./workspace", model, channels=[comments, EmailInbox()])
comments.post("launch-plan.md", "Design slipped a week. Move the date.", quote="We ship on March 3.")
agent.wake()   # answers in that comment thread (not chat) and edits the file
```

`examples/email_inbox.py` does the same for email (each correspondent is an
artifact, so the thread tail is your history with that person). A new input
type is one small `Channel` subclass.

## Spike caveats

- The executor runs `exec()` in-process. It is **not a sandbox**. A
  subprocess/Docker/E2B executor with the same `run(code)` shape is next.
- No approval gate for side effects yet.
- No LLM history compaction, only a message-count trim.
