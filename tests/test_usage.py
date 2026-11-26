"""Spend limits: counted from every model turn, enforced before every model call."""

import pytest

from plato import Agent, Limits, ModelTurn, ScriptedModel
from plato.usage import token_counts


def turn(code, inp=60, out=20):
    return ModelTurn(code=code, usage={"prompt_tokens": inp, "completion_tokens": out, "total_tokens": inp + out})


def test_token_counts_understand_both_usage_shapes():
    assert token_counts({"prompt_tokens": 10, "completion_tokens": 3}) == (10, 3)
    assert token_counts({"input_tokens": 10, "cache_read_input_tokens": 90, "output_tokens": 3}) == (100, 3)
    assert token_counts({}) == (0, 0)


def test_a_cost_limit_needs_prices():
    with pytest.raises(ValueError, match="price_per_mtok"):
        Limits(daily_cost=1.0)


def test_limit_stops_a_run_partway_and_holds_new_messages(tmp_path):
    model = ScriptedModel([turn('reply("on it")'), turn('print("working")'), turn('print("more work")')])
    agent = Agent(tmp_path / "ws", model, limits=Limits(daily_tokens=150))
    agent.send("write the report")
    r = agent.wake()
    assert r.outcome == "over_limit" and len(model.calls) == 2           # 80 + 80 ≥ 150: third call never made
    assert agent.usage.today() == {"input": 120, "output": 40, "calls": 2}
    notices = [e.text for e in agent.log.recent(agent.thread) if e.actor_type == "agent" and "paused" in e.text]
    assert len(notices) == 1 and "token limit (150)" in notices[0] and "stopped partway" in notices[0]

    agent.send("are you there?")
    held = agent.wake()
    assert held.outcome == "over_limit" and len(model.calls) == 2      # no model call at all
    assert [i.text for i in agent.pending_inbox()] == ["are you there?"]  # waits for budget, not lost
    assert len([e for e in agent.log.recent(agent.thread) if "paused" in e.text]) == 1  # told once a day


def test_usage_survives_a_restart(tmp_path):
    first = Agent(tmp_path / "ws", ScriptedModel([turn('reply("hi"); stop("x")', 90, 20)]),
                  limits=Limits(daily_tokens=100))
    first.send("hello")
    assert first.wake().outcome == "stopped"
    again = Agent(tmp_path / "ws", ScriptedModel([turn('reply("hi"); stop("x")')]), limits=Limits(daily_tokens=100))
    again.send("hello again")
    assert again.wake().outcome == "over_limit"


def test_cost_limit(tmp_path):
    limits = Limits(daily_cost=0.01, price_per_mtok=(100.0, 400.0))  # 60 in + 20 out = $0.014
    agent = Agent(tmp_path / "ws", ScriptedModel([turn('reply("a")'), turn('stop("x")')]), limits=limits)
    agent.send("go")
    r = agent.wake()
    assert r.outcome == "over_limit" and "spend limit (0.01)" in r.summary


def test_no_limits_means_no_checks(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel([turn('reply("a"); stop("x")', 10**7, 10**7)]))
    agent.send("go")
    assert agent.wake().outcome == "stopped"
    assert agent.usage.today()["calls"] == 1                            # still recorded, for visibility
