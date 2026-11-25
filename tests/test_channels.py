"""The generic channel machinery, exercised with a throwaway channel defined here.

Nothing in this file depends on the recipes in examples/: if these pass, the
building blocks work for any kind of input.
"""

from plato import (
    Agent,
    Channel,
    Chat,
    FileReference,
    PayloadField,
    Policy,
    RecentEvents,
    Rule,
    ScriptedModel,
    ToolRegistry,
)


class Tickets(Channel):
    """A minimal support-ticket channel: people add notes, the agent replies."""

    name = "tickets"
    artifact_types = ("ticket",)
    gate_rules = (Rule("note", True, actor_type="human"),)
    guidance = "--- tickets ---\nAnswer with note()."

    def __init__(self, limit=6, max_chars=300):
        self.tail_parts = (
            PayloadField("priority", "Priority: {value}"),
            RecentEvents(surfaces=("note",), limit=limit, max_chars=max_chars),
            FileReference("attachment"),
        )

    def add(self, ticket, text, **payload):
        return self.receive(f"ticket:{ticket}", "ticket", "note", text, ticket=ticket, **payload)

    def tools(self):
        reg = ToolRegistry()

        @reg.tool(group="Tickets")
        def note(ticket: str, text: str) -> str:
            """Add a note to a ticket."""
            self.act(f"ticket:{ticket}", "ticket", "note", text, ticket=ticket)
            return "noted"
        return reg

    def reply_hint(self, item):
        return f"note(ticket={item.meta['ticket']!r}, text='...')"


def make(tmp_path, steps, **kw):
    model = ScriptedModel(steps)
    tickets = Tickets(**kw)
    return Agent(tmp_path / "ws", model, channels=[tickets]), tickets, model


def ctx(model, call=0):
    return model.calls[call][1]["content"]


def handled(agent):
    agent.log.dispatch_once()
    agent.log.mark_done(agent.agent_id, {e.event_id for e in agent.log.pending_inbox(agent.agent_id)})


def test_channel_event_is_routed_with_hint_tail_and_guidance(tmp_path):
    agent, tickets, model = make(tmp_path, ['note("7", "Looking into it."); stop("x")'])
    (agent.ws.files / "log.txt").write_text("a\nb\n")
    tickets.add("7", "Login is broken", priority="high", attachment="log.txt")
    r = agent.wake()
    c = ctx(model)
    assert "→ Reply in: note(ticket='7', text='...')" in c
    assert "note on ticket:7: Login is broken" in c                  # default label
    assert "Priority: high" in c                                     # PayloadField
    assert "File: log.txt (4 chars, 3 lines)" in c                    # FileReference, body not inlined
    assert c.count("--- tickets ---") == 1                           # guidance once
    assert "Answer each item where it arrived" in c                  # generic footer
    assert r.outcome == "stopped" and r.steps == 1                    # act() counts as answering


def test_no_rule_no_wake_even_with_a_channel(tmp_path):
    agent, tickets, _ = make(tmp_path, [])
    tickets.receive("ticket:7", "ticket", "edit", "renamed title")
    assert agent.wake() is None


def test_recent_events_is_bounded_and_excludes_trigger(tmp_path):
    agent, tickets, model = make(tmp_path, ['note("7", "ok"); stop("x")'], limit=2, max_chars=20)
    tickets.add("7", "first")
    agent.log.submit("ticket:7", actor_type="agent", actor_id="agent", surface="note",
                     payload={"text": "agent answer", "ticket": "7"})
    tickets.add("7", "a very long follow-up note indeed")
    handled(agent)
    tickets.add("7", "and now?")
    agent.wake()
    c = ctx(model)
    assert "most recent 2 of 3" in c
    assert "- You (" in c and "agent answer" in c
    assert "a very long follow-…" in c                               # per-event cap
    assert "): first" not in c                                       # oldest dropped
    assert c.count("and now?") == 1                                  # trigger not repeated


def test_agent_action_does_not_rewake(tmp_path):
    agent, tickets, _ = make(tmp_path, ['note("7", "done"); stop("x")'])
    tickets.add("7", "please fix")
    agent.wake()
    assert [e.actor_type for e in agent.log.recent("ticket:7")] == ["human", "agent"]
    assert agent.pending_inbox() == []


def test_chat_reply_is_withheld_on_a_channel_wake(tmp_path):
    agent, tickets, model = make(tmp_path, ['reply("in chat by mistake")', 'note("7", "here"); stop("x")'])
    tickets.add("7", "question?")
    r = agent.wake()
    assert r.replies == []
    assert "not delivered" in [m["content"] for m in model.calls[1] if m["role"] == "tool"][-1]
    assert all(e.actor_type == "human" for e in agent.log.recent(agent.thread))


def test_mixed_batch_keeps_chat_live(tmp_path):
    agent, tickets, model = make(tmp_path, ['reply("hi"); note("7", "ok"); stop("x")'])
    tickets.add("7", "question?")
    agent.send("hello in chat")
    assert agent.wake().replies == ["hi"]
    assert "Your inbox has new items" in ctx(model)


def test_chat_message_mid_run_lifts_suppression(tmp_path):
    agent = None

    def chat_arrives(_):
        agent.send("quick question in chat")
        return 'print("working")'

    tickets = Tickets()
    agent = Agent(tmp_path / "ws", ScriptedModel(['note("7", "on it")', chat_arrives, 'reply("sure"); stop("x")']),
                  channels=[tickets])
    tickets.add("7", "please fix")
    assert agent.wake().replies == ["sure"]


def test_event_on_an_unowned_artifact_type_falls_back_to_chat(tmp_path):
    model = ScriptedModel(['reply("saw it"); stop("x")'])
    agent = Agent(tmp_path / "ws", model, policy=Policy().with_rules(Rule("alert", True, reason="alert")))
    agent.follow("monitor:cpu", "monitor")
    agent.log.submit("monitor:cpu", actor_type="system", actor_id="prom", surface="alert",
                     payload={"text": "CPU at 98%"})
    r = agent.wake()
    assert r.replies == ["saw it"]                                   # no channel owns it → chat
    assert "alert on monitor:cpu: CPU at 98%" in ctx(model)


# ── chat is just another channel ─────────────────────────────────────────
def test_agent_without_chat_works_only_through_its_channels(tmp_path):
    tickets = Tickets()
    model = ScriptedModel(['note("7", "on it"); stop("x")'])
    agent = Agent(tmp_path / "ws", model, channels=[tickets], chat=False)
    desc = agent.tool_schema["function"]["description"]
    assert "reply(" not in desc and "note(ticket, text)" in desc
    tickets.add("7", "help")
    assert agent.wake().outcome == "stopped"
    assert agent.policy.rules == Tickets.gate_rules                  # no chat rule anywhere


def test_chat_can_be_configured_like_any_channel(tmp_path):
    heard = []
    model = ScriptedModel(['reply("one"); stop("x")', 'reply("two"); stop("x")', 'reply("three"); stop("x")'])
    agent = Agent(tmp_path / "ws", model, chat=Chat(recall=2, on_message=heard.append))
    for text in ("a", "b", "c"):
        agent.send(text)
        agent.wake()
    assert heard == ["one", "two", "three"]
    recall = ctx(model, call=2).split("--- recent conversation ---\n")[1].split("\n\n")[0]
    assert [ln.split(" (")[0] + ":" + ln.split("):")[1] for ln in recall.splitlines()] == [
        "User: b", "You: two"]                                       # bounded to 2 turns


def test_answer_only_when_asked_is_generic(tmp_path):
    class QuietTickets(Tickets):
        answer_only_when_asked = True

    tickets = QuietTickets()
    model = ScriptedModel(['note("7", "unprompted")', 'reply("hi"); stop("x")'])
    agent = Agent(tmp_path / "ws", model, channels=[tickets])
    agent.send("hello")                                              # input arrives by chat only
    r = agent.wake()
    assert [a["channel"] for a in r.actions] == ["chat"]              # the ticket note was held back
    assert agent.log.recent("ticket:7") == []
