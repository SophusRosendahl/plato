"""The event log, the should_respond gate, and the agent running on top of them."""

import pytest

from plato import Agent, ScriptedModel
from plato.events import EventLog
from plato.should_respond import ArtifactView, Participant, Policy, Rule, should_respond


# ── the gate (pure) ──────────────────────────────────────────────────────
def gate(surface, actor=("human", "u"), artifact="thread", op=None, payload=None, vis="all",
         me="agent", policy=None, subscribed=True):
    ev = {"surface": surface, "actor_type": actor[0], "actor_id": actor[1], "op": op,
          "payload": payload or {}, "visibility": vis}
    return should_respond(Participant(me), ev, ArtifactView(artifact, subscribed), policy)


TICKETS = Policy().with_rules(
    Rule("status", False, op="rename", reason="bookkeeping"),
    Rule("status", True, actor_type="human", reason="status-change"),
    Rule("note", True),
)


@pytest.mark.parametrize("kwargs,respond,surface,reason", [
    # universal checks
    (dict(surface="chat", actor=("agent", "agent")), False, None, "self-trigger"),
    (dict(surface="chat", subscribed=False), False, None, "not-subscribed"),
    (dict(surface="chat", vis="someone-else"), False, None, "not-visible"),
    (dict(surface="chat", vis="x, agent"), True, "chat", "chat-message"),
    (dict(surface="chat", payload={"to": ["bob"]}), False, None, "addressed-to-others"),
    (dict(surface="chat", payload={"to": ["agent"]}), True, "chat", "chat-message"),
    # the one built-in rule
    (dict(surface="chat"), True, "chat", "chat-message"),
    (dict(surface="chat", artifact="file"), True, "chat", "chat-message"),     # chat anywhere → chat
    (dict(surface="chat", actor=("agent", "other-agent")), True, "chat", "chat-message"),
    # everything else needs a rule
    (dict(surface="note", artifact="ticket"), False, None, "no-rule"),
    (dict(surface="edit", artifact="file"), False, None, "no-rule"),
    # with rules: first match wins, answers on the artifact
    (dict(surface="note", artifact="ticket", policy=TICKETS), True, "ticket", "note"),
    (dict(surface="status", artifact="ticket", op="rename", policy=TICKETS), False, None, "bookkeeping"),
    (dict(surface="status", artifact="ticket", op="moved", policy=TICKETS), True, "ticket", "status-change"),
    (dict(surface="status", artifact="ticket", actor=("system", "cron"), policy=TICKETS),
     False, None, "no-rule"),
    (dict(surface="note", artifact="thread", policy=TICKETS), True, "chat", "note"),  # chat artifact
])
def test_gate_branches(kwargs, respond, surface, reason):
    d = gate(**kwargs)
    assert (d.respond, d.surface, d.reason) == (respond, surface, reason)
    assert bool(d) is respond


def test_rule_can_send_answers_to_chat():
    policy = Policy().with_rules(Rule("upload", True, answer="chat"))
    assert gate("upload", artifact="file", policy=policy).surface == "chat"


# ── the log ──────────────────────────────────────────────────────────────
def test_log_rejects_malformed_events(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("thread:a", "thread")
    with pytest.raises(ValueError):
        log.submit("thread:a", actor_type="robot", actor_id="x", surface="chat")
    with pytest.raises(ValueError):
        log.submit("thread:a", actor_type="human", actor_id="x", surface="")
    with pytest.raises(KeyError):
        log.submit("nowhere", actor_type="human", actor_id="x", surface="chat")


def test_dispatch_fans_out_and_is_idempotent(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("ticket:42", "ticket")
    for p in ("alice-agent", "bob-agent"):
        log.subscribe(p, "ticket:42")
        log.set_policy(p, TICKETS)
    log.submit("ticket:42", actor_type="human", actor_id="ana", surface="note", payload={"text": "hi"})
    log.submit("ticket:42", actor_type="human", actor_id="ana", surface="edit", payload={})
    decisions = log.dispatch_once()
    assert sorted((d.participant, d.decision.reason) for d in decisions) == [
        ("alice-agent", "no-rule"), ("alice-agent", "note"),
        ("bob-agent", "no-rule"), ("bob-agent", "note")]
    assert log.dispatch_once() == []                                 # cursor advanced
    assert [e.text for e in log.pending_inbox("alice-agent")] == ["hi"]
    assert log.pending_inbox("alice-agent")[0].reply_surface == "ticket"


def test_participants_on_one_log_can_have_different_rules(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("ticket:1", "ticket")
    log.subscribe("triager", "ticket:1")
    log.subscribe("archivist", "ticket:1")
    log.set_policy("triager", TICKETS)                               # archivist keeps the default
    log.submit("ticket:1", actor_type="human", actor_id="ana", surface="note", payload={"text": "x"})
    log.dispatch_once()
    assert len(log.pending_inbox("triager")) == 1
    assert log.pending_inbox("archivist") == []


def test_mark_done_only_flips_what_was_shown(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("thread:a", "thread")
    log.subscribe("a", "thread:a")
    shown = log.submit("thread:a", actor_type="human", actor_id="u", surface="chat", payload={"text": "1"})
    log.dispatch_once()
    log.submit("thread:a", actor_type="human", actor_id="u", surface="chat", payload={"text": "2"})
    log.dispatch_once()
    log.mark_done("a", {shown.event_id})
    assert [e.text for e in log.pending_inbox("a")] == ["2"]


def test_log_survives_reopen(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("thread:a", "thread")
    log.subscribe("a", "thread:a")
    log.submit("thread:a", actor_type="human", actor_id="u", surface="chat", payload={"text": "persisted"})
    log.dispatch_once()
    again = EventLog(tmp_path / "e.db")
    assert [e.text for e in again.pending_inbox("a")] == ["persisted"]


# ── the agent on the log ─────────────────────────────────────────────────
def test_agent_replies_are_logged_and_recalled_next_wake(tmp_path):
    model = ScriptedModel(['reply("Nice to meet you, Ada."); stop("greeted")',
                           'reply("You said your name is Ada."); stop("answered")'])
    agent = Agent(tmp_path / "ws", model)
    agent.send("hi, I'm Ada")
    agent.wake()
    agent.send("what's my name?")
    agent.wake()
    ctx = model.calls[1][1]["content"]
    assert "--- recent conversation ---\nUser: hi, I'm Ada\nYou: Nice to meet you, Ada." in ctx
    recall = ctx.split("--- recent conversation ---")[1].split("--- inbox ---")[0]
    assert "what's my name?" not in recall                             # trigger not duplicated
    thread = agent.log.recent(agent.thread)
    assert [(e.actor_type, e.text) for e in thread][-1] == ("agent", "You said your name is Ada.")


def test_events_without_a_rule_are_logged_but_do_not_wake(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel([]))
    agent.follow("file:plan.md", "file")
    agent.log.submit("file:plan.md", actor_type="human", actor_id="user", surface="edit",
                     payload={"path": "plan.md"})
    assert agent.wake() is None                                      # the gate said no
    assert len(agent.log.recent("file:plan.md")) == 1                # but it is on the log


def test_agent_file_writes_are_logged_without_waking(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel(['write_file("a.md", "x"); reply("ok"); stop("x")']))
    agent.send("write a.md")
    agent.wake()
    assert [e.surface for e in agent.log.recent("file:a.md")] == ["edit"]
    assert agent.pending_inbox() == []


def test_trace_explains_gate_decisions(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel(['reply("ok"); stop("x")']))
    agent.follow("file:d.md", "file")
    agent.log.submit("file:d.md", actor_type="human", actor_id="user", surface="edit", payload={})
    agent.send("hello")
    r = agent.wake()
    gates = [ln for ln in r.trace_path.read_text().splitlines() if '"kind": "gate"' in ln]
    assert any('"reason": "no-rule"' in g and '"surface": "edit"' in g for g in gates)
    assert any('"reason": "chat-message"' in g for g in gates)
