"""Spend limits: counted from every model turn, enforced before every model call."""

import json
from datetime import date

import pytest

from demos._shared.models import OfflineModel, spend_limits

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


def test_monthly_cost_limit_counts_the_whole_month(tmp_path):
    usage_file = tmp_path / "usage.json"
    usage_file.write_text(json.dumps({                          # earlier this month: 1M in + 1M out
        date.today().replace(day=1).isoformat(): {"input": 1_000_000, "output": 1_000_000, "calls": 50}}))
    limits = Limits(monthly_cost=100.0, price_per_mtok=(2.5, 15.0))   # 2.5 + 15 = 17.5 so far
    agent = Agent(tmp_path / "ws", ScriptedModel([turn('reply("a"); stop("x")')]), limits=limits,
                  usage_file=usage_file)
    agent.send("go")
    assert agent.wake().outcome == "stopped"                    # 17.5 of 100: room left
    usage_file.write_text(json.dumps({date.today().replace(day=1).isoformat():
                                      {"input": 6_000_000, "output": 6_000_000, "calls": 300}}))  # 105
    agent.send("again")
    r = agent.wake()
    assert r.outcome == "over_limit" and "this month's spend limit (100.00)" in r.summary


def test_agents_sharing_a_usage_file_share_one_budget(tmp_path):
    shared = tmp_path / "usage.json"
    limits = Limits(daily_tokens=100)
    a = Agent(tmp_path / "a", ScriptedModel([turn('reply("hi"); stop("x")', 90, 20)]), limits=limits, usage_file=shared)
    b = Agent(tmp_path / "b", ScriptedModel([turn('reply("hi"); stop("x")')]), limits=limits, usage_file=shared)
    a.send("hello")
    a.wake()
    b.send("hello")
    assert b.wake().outcome == "over_limit"                     # a used the shared budget up


def test_demos_cap_real_models_on_a_shared_monthly_budget(tmp_path, monkeypatch):
    monkeypatch.setenv("PLATO_USAGE_FILE", str(tmp_path / "u.json"))
    monkeypatch.delenv("PLATO_MONTHLY_BUDGET", raising=False)
    monkeypatch.delenv("PLATO_PRICE_PER_MTOK", raising=False)
    kw = spend_limits(ScriptedModel([]))
    assert kw["limits"].monthly_cost == 100 and kw["limits"].daily_cost == 5
    assert kw["limits"].price_per_mtok == (2.5, 15.0) and str(kw["usage_file"]).endswith("u.json")
    assert spend_limits(OfflineModel()) == {}                   # the stand-in costs nothing
