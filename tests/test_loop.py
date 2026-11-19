"""Loop + guard behaviour, driven by a scripted model (no network, no key)."""

import pytest

from plato import Agent, Config, ScriptedModel, ToolRegistry, Workspace
from plato.agent import trim_history
from plato.model import ModelTurn


def make(tmp_path, steps, default=None, **cfg):
    model = ScriptedModel(steps, default=default)
    agent = Agent(tmp_path / "ws", model, config=Config(**cfg))
    return agent, model


def last_user_msgs(model, call_index=-1):
    return [m["content"] for m in model.calls[call_index] if m["role"] == "user"]


def test_dormant_without_inbox(tmp_path):
    agent, model = make(tmp_path, [])
    assert agent.wake() is None
    assert model.calls == []


def test_inline_answer_is_one_step(tmp_path):
    agent, _ = make(tmp_path, ['reply("Hi! What can I do?")\nstop("Greeted.")'])
    agent.send("hello")
    r = agent.wake()
    assert r.outcome == "stopped" and r.steps == 1
    assert r.replies == ["Hi! What can I do?"]
    assert "Greeted." in agent.ws.log_tail()
    assert agent.ws.pending_inbox() == []  # consumed on render


def test_wake_context_puts_inbox_last_with_footer(tmp_path):
    agent, model = make(tmp_path, ['reply("ok"); stop("x")'])
    agent.ws.remember("User is called Ada")
    agent.send("summarise my notes")
    agent.wake()
    ctx = model.calls[0][1]["content"]
    assert ctx.index("--- memory ---") < ctx.index("--- inbox ---")
    assert "User is called Ada" in ctx and "summarise my notes" in ctx
    assert ctx.rstrip().endswith("(triage: answer inline, plan, or say you can't).")


def test_plan_then_work_then_stop(tmp_path):
    agent, _ = make(tmp_path, [
        'reply("On it.")\nadd_todo("Draft poem -> poem.md")\nadd_todo("Self-review poem.md")',
        'write_file("poem.md", "roses")\nmark_todo_done(1)',
        'mark_todo_done(2)\nreply("Done: poem.md")\nstop("Wrote a poem.")',
    ])
    agent.send("write me a poem")
    r = agent.wake()
    assert r.outcome == "stopped" and r.steps == 3
    assert (agent.ws.files / "poem.md").read_text() == "roses"
    assert agent.ws.todos() == []  # resolved items cleared on stop


def test_variables_persist_within_a_wake(tmp_path):
    agent, model = make(tmp_path, [
        'x = 41\nadd_todo("t")',
        'x + 1',
        'mark_todo_done(1); stop("ok")',
    ])
    agent.send("go")
    agent.wake()
    tool_outputs = [m["content"] for m in model.calls[2] if m["role"] == "tool"]
    assert tool_outputs[-1].startswith("42")  # bare expression echoed, REPL-style


def test_stop_guard_rejects_stop_with_active_todos(tmp_path):
    agent, model = make(tmp_path, [
        'reply("ok")\nadd_todo("Research X")\nstop("planned")',
        'mark_todo_done(1)\nstop("did it")',
    ])
    agent.send("research X")
    r = agent.wake()
    assert r.outcome == "stopped" and r.steps == 2
    correction = last_user_msgs(model)[-1]
    assert "still has active items" in correction and "1. [ ] Research X" in correction


def test_blocked_todo_lets_agent_stop(tmp_path):
    agent, _ = make(tmp_path, [
        'reply("ok")\nadd_todo("Book venue")\nask_user("Which city?")\nmark_todo_blocked(1, "city?")\nstop("asked")',
    ])
    agent.send("book a venue")
    r = agent.wake()
    assert r.outcome == "stopped" and r.replies == ["ok", "Which city?"]
    assert agent.ws.todos()[0].status == "blocked"


def test_stop_guard_is_bounded(tmp_path):
    agent, _ = make(tmp_path, ['reply("ok")\nadd_todo("never done")\nstop()'],
                    default='stop("again")', max_stop_rejections=2)
    agent.send("go")
    r = agent.wake()
    assert r.outcome == "stop_guard_exhausted" and r.steps == 3
    assert "stop guard exhausted" in agent.ws.log_tail()


def test_silent_first_step_stop_is_refused(tmp_path):
    agent, model = make(tmp_path, ['stop("nothing")', 'reply("Hi!")\nstop("greeted")'])
    agent.send("hey")
    r = agent.wake()
    assert r.outcome == "stopped" and r.steps == 2
    assert "without replying" in last_user_msgs(model)[-1]


def test_errors_are_fed_back_and_loop_continues(tmp_path):
    agent, model = make(tmp_path, ['1/0', 'reply("fixed"); stop("ok")'])
    agent.send("go")
    r = agent.wake()
    assert r.outcome == "stopped"
    tool_out = [m["content"] for m in model.calls[1] if m["role"] == "tool"][-1]
    assert "[error] ZeroDivisionError (line 1)" in tool_out


def test_no_tool_call_gets_nudged(tmp_path):
    agent, model = make(tmp_path, [ModelTurn(code=None, text="Sure!"), 'reply("Sure!"); stop("ok")'])
    agent.send("hi")
    assert agent.wake().outcome == "stopped"
    assert "must call the run tool" in last_user_msgs(model)[-1]


def test_budget_wrap_up_notice(tmp_path):
    agent, model = make(tmp_path, ['reply("ok"); add_todo("long")'], default='print("working")',
                        max_steps=9, wrap_up_steps=3)
    agent.send("go")
    agent.wake()
    notices = [c for c in last_user_msgs(model) if "Budget notice" in c]
    assert len(notices) == 1 and "3 step(s) left" in notices[0]


def test_mid_run_message_is_delivered(tmp_path):
    agent = None

    def send_mid_run(messages):
        agent.send("also: make it rhyme")
        return 'print("working")'

    model = ScriptedModel(['reply("ok"); add_todo("poem")', send_mid_run,
                           'mark_todo_done(1); stop("done")'])
    agent = Agent(tmp_path / "ws", model)
    agent.send("write a poem")
    agent.wake()
    assert any("make it rhyme" in c for c in last_user_msgs(model))
    assert agent.ws.pending_inbox() == []


def test_circuit_breaker_trips_and_caps(tmp_path):
    spin = 'print("trying the same thing")'
    agent, model = make(tmp_path, ['reply("ok"); add_todo("impossible task")'], default=spin,
                        max_steps=4, wrap_up_steps=0, breaker_max_wakes=3, breaker_paused_steps=2)
    agent.send("do the impossible")
    outcomes = [agent.wake().outcome]
    for _ in range(2):
        agent.send("are you on it?")
        outcomes.append(agent.wake().outcome)
    assert outcomes == ["budget_exhausted", "budget_exhausted", "stuck"]
    assert agent.ws.todos()[0].status == "blocked"
    # the 2nd/3rd wakes were told they were spinning
    assert any("Circuit breaker" in c for c in last_user_msgs(model))
    # a later nudge on the unchanged set only gets the small paused budget
    agent.send("try again?")
    r = agent.wake()
    assert r.steps == 2


def test_progress_resets_breaker(tmp_path):
    agent, _ = make(tmp_path, ['reply("ok"); add_todo("a"); add_todo("b")', 'mark_todo_done(1)'],
                    default='print("hmm")', max_steps=4, wrap_up_steps=0, breaker_max_wakes=2)
    agent.send("go")
    assert agent.wake().outcome == "budget_exhausted"
    assert agent._no_progress_wakes == 0  # resolving a todo counted as progress


def test_file_tools_are_confined_to_files_folder(tmp_path):
    agent, model = make(tmp_path, ['read_file("../memory.md")', 'reply("no"); stop("x")'])
    agent.send("go")
    agent.wake()
    out = [m["content"] for m in model.calls[1] if m["role"] == "tool"][-1]
    assert "PathEscape" in out and "outside the workspace" in out


def test_custom_tools_appear_in_catalogue_and_run(tmp_path):
    reg = ToolRegistry()

    @reg.tool(group="Weather")
    def forecast(city: str, days: int = 1) -> str:
        """Get the forecast for a city."""
        return f"{city}: sunny for {days} day(s)"

    model = ScriptedModel(['reply(forecast("Oslo", days=2)); stop("ok")'])
    agent = Agent(Workspace(tmp_path / "ws"), model, tools=reg)
    agent.send("weather?")
    r = agent.wake()
    assert r.replies == ["Oslo: sunny for 2 day(s)"]
    desc = agent.tool_schema["function"]["description"]
    assert "forecast(city, days=1)" in desc and "Get the forecast for a city." in desc


def test_trace_is_written(tmp_path):
    agent, _ = make(tmp_path, ['reply("hi"); stop("ok")'])
    agent.send("hi")
    r = agent.wake()
    kinds = [line.split('"kind": "')[1].split('"')[0] for line in r.trace_path.read_text().splitlines()]
    assert kinds == ["wake", "reply", "step", "end"]


@pytest.mark.parametrize("keep", [3, 4, 5])
def test_trim_never_starts_on_orphan_tool(keep):
    msgs = [{"role": "system"}, {"role": "user"}]
    for _ in range(5):
        msgs += [{"role": "assistant", "tool_calls": [{}]}, {"role": "tool"}]
    out = trim_history(msgs, keep)
    assert out[0]["role"] == "system" and out[1]["role"] != "tool"
