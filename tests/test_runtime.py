"""Always-on serving, replies, reactions, addressed rules, output capture, failure recovery."""

import threading
import time

from plato import Agent, Chat, EventLog, Policy, Rule, ScriptedModel
from plato.executor import Executor
from plato.should_respond import ArtifactView, Participant, should_respond


def wait_until(pred, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


def test_wait_for_events_wakes_on_submit(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("thread:a", "thread")
    seen = log.last_seq()
    threading.Timer(0.1, lambda: log.submit("thread:a", actor_type="human", actor_id="u",
                                            surface="chat", payload={"text": "x"})).start()
    t0 = time.time()
    assert log.wait_for_events(seen, timeout=5)
    assert time.time() - t0 < 2                                      # woke on the submit, not the timeout
    assert [e.text for e in log.events_after(seen)] == ["x"]


def test_serve_answers_messages_from_another_thread(tmp_path):
    heard = []
    agent = Agent(tmp_path / "ws", ScriptedModel([], default='reply("pong"); stop("x")'),
                  chat=Chat(on_message=heard.append))
    stop = threading.Event()
    t = threading.Thread(target=agent.serve, kwargs={"stop": stop, "poll": 0.2}, daemon=True)
    t.start()
    agent.send("ping")
    assert wait_until(lambda: heard == ["pong"])
    agent.send("ping again")
    assert wait_until(lambda: heard == ["pong", "pong"])
    stop.set()
    t.join(timeout=3)
    assert not t.is_alive()


def test_follow_up_typed_while_the_model_thinks_joins_the_same_wake(tmp_path):
    thinking, typed = threading.Event(), threading.Event()

    def slow_first_step(_):
        thinking.set()
        typed.wait(5)                                                # the person types now
        return 'reply("On it, drafting the email.")'

    model = ScriptedModel([slow_first_step, 'reply("Sure, I will keep it short."); stop("x")'])
    agent = Agent(tmp_path / "ws", model)
    stop = threading.Event()
    threading.Thread(target=agent.serve, kwargs={"stop": stop, "poll": 0.2}, daemon=True).start()
    agent.send("draft an email to the landlord")
    assert thinking.wait(5)
    assert agent.busy
    agent.send("oh and keep it short")
    typed.set()
    assert wait_until(lambda: not agent.busy and len(model.calls) == 2)
    stop.set()
    second_step = [m["content"] for m in model.calls[1] if m["role"] == "user"][-1]
    assert "arrived while you were working" in second_step and "keep it short" in second_step


def test_replies_carry_what_they_reply_to(tmp_path):
    model = ScriptedModel(['reply("Option B is cheaper."); stop("x")',
                           lambda _: 'reply("Booked B.", reply_to=REPLY); stop("x")'])
    agent = Agent(tmp_path / "ws", model)
    agent.send("which option?")
    agent.wake()
    answer = agent.log.recent(agent.thread, surfaces=("chat",))[-1]
    follow = agent.chat.send("book that one", reply_to=answer.event_id)
    model.steps[0] = f'reply("Booked B.", reply_to={follow.event_id!r}); stop("x")'
    agent.wake()
    ctx = model.calls[1][1]["content"]
    assert f"→ Reply in: reply(message, reply_to={follow.event_id!r})" in ctx
    assert f"In reply to you ({answer.event_id}): Option B is cheaper." in ctx
    assert agent.log.recent(agent.thread, surfaces=("chat",))[-1].parent_id == follow.event_id


def test_reactions_are_logged_and_shown_but_do_not_wake_by_default(tmp_path):
    model = ScriptedModel(['reply("Here is the plan."); stop("x")', 'reply("ok"); stop("x")'])
    agent = Agent(tmp_path / "ws", model)
    agent.send("plan my week")
    agent.wake()
    msg = agent.log.recent(agent.thread, surfaces=("chat",))[-1]
    agent.chat.react(msg.event_id, "👍")
    assert agent.wake() is None                                      # no rule for reactions
    agent.send("thanks")
    agent.wake()
    assert f"You ({msg.event_id}): Here is the plan.  [👍 user]" in model.calls[1][1]["content"]


def test_chosen_reactions_can_wake(tmp_path):
    model = ScriptedModel(['reply("Do X, then Y."); stop("x")',
                           'reply("More detail: X means ..."); stop("x")'])
    agent = Agent(tmp_path / "ws", model, chat=Chat(wake_on=("❓",)))
    agent.send("what next?")
    agent.wake()
    msg = agent.log.recent(agent.thread, surfaces=("chat",))[-1]
    agent.chat.react(msg.event_id, "👍")
    assert agent.wake() is None                                      # 👍 still doesn't wake
    agent.chat.react(msg.event_id, "❓")
    r = agent.wake()
    assert r is not None
    ctx = model.calls[1][1]["content"]
    assert "User reacted: ❓" in ctx and "In reply to you" in ctx and "Do X, then Y." in ctx


def test_agent_can_react(tmp_path):
    model = ScriptedModel([lambda m: 'react(ID, "👀"); stop("acked")'])
    agent = Agent(tmp_path / "ws", model)
    ev = agent.send("fyi the meeting moved")
    model.steps[0] = f'react({ev.event_id!r}, "👀"); stop("acked")'
    r = agent.wake()
    assert r.outcome == "stopped"                                    # a reaction counts as answering
    reaction = agent.log.recent(agent.thread, surfaces=("reaction",))[-1]
    assert (reaction.actor_type, reaction.op, reaction.parent_id) == ("agent", "👀", ev.event_id)


def test_addressed_rules_fire_only_for_the_named_participant():
    rule = Policy(rules=(Rule("comment", True, addressed=True, reason="mentioned"),))
    view = ArtifactView("page")

    def decide(payload):
        ev = {"surface": "comment", "actor_type": "human", "actor_id": "u", "payload": payload}
        return should_respond(Participant("plato"), ev, view, rule).reason

    assert decide({"to": ["plato"]}) == "mentioned"
    assert decide({}) == "no-rule"                                   # untagged: logged, no wake
    assert decide({"to": ["alice"]}) == "addressed-to-others"


def test_callbacks_reach_the_screen_while_the_agents_own_prints_do_not(tmp_path, capsys):
    agent = Agent(tmp_path / "ws", ScriptedModel(['print("agent output"); reply("hi"); stop("x")']),
                  chat=Chat(on_message=lambda t: print(f"UI: {t}")))
    agent.send("hello")
    r = agent.wake()
    screen = capsys.readouterr().out
    assert "UI: hi" in screen                                        # the callback reached the terminal
    assert "agent output" not in screen                              # the agent's print did not…
    assert '"output": "agent output' in r.trace_path.read_text()      # …it became the agent's tool output


def test_other_threads_printing_during_agent_code_are_not_captured(capsys):
    started, done = threading.Event(), threading.Event()

    def other_thread():
        started.wait(5)
        print("from the web server")
        done.set()

    threading.Thread(target=other_thread).start()
    result = Executor({"started": started, "done": done}).run("started.set()\n_ = done.wait(5)\nprint('mine')")
    assert result.output.strip() == "mine"                           # only the agent's own print
    assert "from the web server" in capsys.readouterr().out          # the other thread hit the screen


class FlakyModel:
    """Fails `failures` times, then delegates to a scripted model."""

    def __init__(self, failures, then):
        self.failures, self.then, self.calls = failures, then, 0

    def complete(self, messages, tool):
        self.calls += 1
        if self.failures:
            self.failures -= 1
            raise ConnectionError("provider unavailable")
        return self.then.complete(messages, tool)


def test_a_failed_wake_keeps_the_message_and_says_so_once(tmp_path):
    model = FlakyModel(2, ScriptedModel(['reply("Here you go."); stop("x")']))
    agent = Agent(tmp_path / "ws", model)
    agent.send("summarise my week")
    first = agent.wake()
    assert first.outcome == "error" and "provider unavailable" in first.summary
    assert [i.text for i in agent.pending_inbox()] == ["summarise my week"]   # not lost
    assert agent.wake().outcome == "error"                                   # still down
    notices = [e.text for e in agent.log.recent(agent.thread) if e.actor_type == "agent"]
    assert len(notices) == 1 and "retry" in notices[0]                       # told once, not per retry
    assert agent.wake().replies == ["Here you go."]                          # recovered, same message
    assert agent.error_streak == 0


def test_a_wake_that_failed_after_acting_is_not_replayed(tmp_path):
    def boom(_):
        raise RuntimeError("model crashed mid-task")

    agent = Agent(tmp_path / "ws", ScriptedModel(['reply("Starting on it.")', boom]))
    agent.send("draft the newsletter")
    assert agent.wake().outcome == "error"
    assert agent.pending_inbox() == []                    # it had already replied: retrying would repeat that
    assert "Please check and ask me again" in agent.log.recent(agent.thread)[-1].text


def test_serve_survives_a_provider_outage(tmp_path, monkeypatch):
    heard = []
    model = FlakyModel(1, ScriptedModel([], default='reply("back online"); stop("x")'))
    agent = Agent(tmp_path / "ws", model, chat=Chat(on_message=heard.append))
    waits = []
    real_wait = threading.Event.wait

    def short_wait(self, timeout=None):                   # don't really back off for 5s in a test
        waits.append(timeout)
        return real_wait(self, min(timeout or 0, 0.05))

    monkeypatch.setattr(threading.Event, "wait", short_wait)
    stop = threading.Event()
    t = threading.Thread(target=agent.serve, kwargs={"stop": stop, "poll": 0.1}, daemon=True)
    t.start()
    agent.send("are you there?")
    assert wait_until(lambda: "back online" in heard)
    stop.set()
    t.join(timeout=3)
    assert 5 in waits and heard[0].startswith("I ran into a problem")
