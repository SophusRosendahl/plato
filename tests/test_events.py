"""The event log, the should_respond gate, and the agent running on top of them."""

import pytest

from plato import Agent, ScriptedModel
from plato.channels import CommentsChannel
from plato.events import EventLog
from plato.should_respond import ArtifactView, Participant, Policy, should_respond


# ── the gate (pure) ──────────────────────────────────────────────────────
def gate(surface, actor=("human", "u"), artifact="thread", op=None, payload=None, vis="all",
         me="agent", policy=None, subscribed=True):
    ev = {"surface": surface, "actor_type": actor[0], "actor_id": actor[1], "op": op,
          "payload": payload or {}, "visibility": vis}
    return should_respond(Participant(me), ev, ArtifactView(artifact, subscribed), policy)


@pytest.mark.parametrize("kwargs,respond,surface,reason", [
    (dict(surface="chat"), True, "chat", "chat-message"),
    (dict(surface="chat", artifact="doc"), True, "chat", "chat-message"),        # doc sidebar chat
    (dict(surface="chat", actor=("agent", "agent")), False, None, "self-trigger"),
    (dict(surface="chat", actor=("agent", "other-agent")), True, "chat", "chat-message"),
    (dict(surface="chat", subscribed=False), False, None, "not-subscribed"),
    (dict(surface="chat", vis="someone-else"), False, None, "not-visible"),
    (dict(surface="chat", vis="x, agent"), True, "chat", "chat-message"),
    (dict(surface="comment", artifact="doc"), True, "doc", "comment"),
    (dict(surface="message", artifact="email"), True, "email", "inbound-message"),
    (dict(surface="mention", artifact="doc", payload={"mentions": ["bob"]}), False, None, "mention-not-me"),
    (dict(surface="mention", artifact="doc", payload={"mentions": ["agent"]}), True, "doc", "mentioned"),
    (dict(surface="edit", artifact="doc"), False, None, "edit-no-wake"),
    (dict(surface="status", artifact="file", op="upload"), True, "file", "status-change"),
    (dict(surface="status", artifact="file", op="rename"), False, None, "status-rename"),
    (dict(surface="status", actor=("system", "cron"), op="tick"), False, None, "system-status"),
    (dict(surface="done"), False, None, "done-event"),
    (dict(surface="telepathy"), False, None, "unknown-surface"),
])
def test_gate_branches(kwargs, respond, surface, reason):
    d = gate(**kwargs)
    assert (d.respond, d.surface, d.reason) == (respond, surface, reason)
    assert bool(d) is respond


def test_policy_parameters():
    assert gate("edit", artifact="doc", policy=Policy(wake_on_edit=True)).reason == "edit"
    assert gate("comment", artifact="board", policy=Policy(chat_artifact_types=frozenset({"board"}))).surface == "chat"


# ── the log ──────────────────────────────────────────────────────────────
def test_log_rejects_malformed_events(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("thread:a", "thread")
    with pytest.raises(ValueError):
        log.submit("thread:a", actor_type="robot", actor_id="x", surface="chat")
    with pytest.raises(ValueError):
        log.submit("thread:a", actor_type="human", actor_id="x", surface="telepathy")
    with pytest.raises(KeyError):
        log.submit("nowhere", actor_type="human", actor_id="x", surface="chat")


def test_dispatch_fans_out_and_is_idempotent(tmp_path):
    log = EventLog(tmp_path / "e.db")
    log.ensure_artifact("doc:plan.md", "doc")
    for p in ("alice-agent", "bob-agent"):
        log.subscribe(p, "doc:plan.md")
    log.submit("doc:plan.md", actor_type="human", actor_id="ana", surface="comment", payload={"text": "hi"})
    log.submit("doc:plan.md", actor_type="human", actor_id="ana", surface="edit", payload={"path": "plan.md"})
    decisions = log.dispatch_once()
    assert sorted((d.participant, d.decision.reason) for d in decisions) == [
        ("alice-agent", "comment"), ("alice-agent", "edit-no-wake"),
        ("bob-agent", "comment"), ("bob-agent", "edit-no-wake")]
    assert log.dispatch_once() == []                                 # cursor advanced
    assert [e.text for e in log.pending_inbox("alice-agent")] == ["hi"]
    assert log.pending_inbox("alice-agent")[0].reply_surface == "doc"


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


def test_human_edit_is_logged_but_does_not_wake(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel([]))
    agent.follow("doc:plan.md", "doc")
    agent.log.submit("doc:plan.md", actor_type="human", actor_id="user", surface="edit",
                     payload={"path": "plan.md"})
    assert agent.wake() is None                                      # the gate said no
    assert len(agent.log.recent("doc:plan.md")) == 1                 # but it is on the log


def test_agent_file_writes_are_logged_without_waking(tmp_path):
    agent = Agent(tmp_path / "ws", ScriptedModel(['write_file("a.md", "x"); reply("ok"); stop("x")']))
    agent.send("write a.md")
    agent.wake()
    assert [e.surface for e in agent.log.recent("doc:a.md")] == ["edit"]
    assert agent.pending_inbox() == []


def test_trace_explains_gate_decisions(tmp_path):
    comments = CommentsChannel()
    agent = Agent(tmp_path / "ws", ScriptedModel(['comment("d.md", "ok"); stop("x")']), channels=[comments])
    agent.follow("doc:d.md", "doc")
    agent.log.submit("doc:d.md", actor_type="human", actor_id="user", surface="edit", payload={})
    comments.post("d.md", "please look")
    r = agent.wake()
    gates = [ln for ln in r.trace_path.read_text().splitlines() if '"kind": "gate"' in ln]
    assert any('"reason": "edit-no-wake"' in g for g in gates)
    assert any('"reason": "comment"' in g for g in gates)


def test_two_agents_share_one_log(tmp_path):
    log = EventLog(tmp_path / "shared.db")
    writer = Agent(tmp_path / "w", ScriptedModel(['comment("spec.md", "Drafted."); stop("x")']),
                   log=log, agent_id="writer", channels=[CommentsChannel()])
    reviewer = Agent(tmp_path / "r", ScriptedModel(['comment("spec.md", "Looks good."); stop("x")']),
                     log=log, agent_id="reviewer", channels=[CommentsChannel()])
    reviewer.follow("doc:spec.md", "doc")
    writer.channels["comments"].post("spec.md", "writer, please draft this")
    assert writer.wake().outcome == "stopped"
    # the reviewer follows the doc, so the human's comment and the writer's reply both reach it
    texts = [i.text for i in reviewer.pending_inbox()]
    assert texts == ["writer, please draft this", "Drafted."]
    reviewer.wake()
    assert writer.pending_inbox()[0].text == "Looks good."           # and the reviewer's reaches the writer
