"""The recipes in examples/: comments and email, built only from public blocks."""

from examples.comments import Comments
from examples.email_inbox import EmailInbox
from plato import Agent, EventLog, ScriptedModel


def ctx(model, call=0):
    return model.calls[call][1]["content"]


# ── comments ─────────────────────────────────────────────────────────────
def test_comment_wake_carries_quote_thread_and_file_reference(tmp_path):
    comments = Comments()
    model = ScriptedModel(['comment("plan.md", "Moving it."); stop("x")'])
    agent = Agent(tmp_path / "ws", model, channels=[comments])
    (agent.ws.files / "plan.md").write_text("We ship on March 3.\n")
    root = comments.post("plan.md", "Is March 3 realistic?", quote="We ship on March 3.")
    comments.act("file:plan.md", "file", "comment", "Tight but doable.", parent_id=root.event_id, path="plan.md")
    agent.log.dispatch_once()
    agent.log.mark_done(agent.agent_id, {root.event_id})
    c = comments.post("plan.md", "Design slipped. Move the date.", parent_id=root.event_id,
                      quote="We ship on March 3.")
    agent.wake()
    w = ctx(model)
    assert f"→ Reply in: comment(path='plan.md', text='...', parent_id='{root.event_id}')" in w
    assert "Comment on plan.md: Design slipped. Move the date." in w
    assert "> We ship on March 3." in w
    assert "Is March 3 realistic?" in w and "- You (" in w            # the thread
    assert "File: plan.md (" in w and "We ship on March 3.\n" not in w.split("--- inbox ---")[0]
    assert "--- how to answer comments ---" in w
    assert c.event_id not in w.split("Earlier on this artifact")[1]   # trigger not in its own tail


def test_comment_reply_lands_in_thread(tmp_path):
    comments = Comments()
    posted = {}
    agent = Agent(tmp_path / "ws", ScriptedModel([
        lambda _: f'comment("d.md", "Will do.", parent_id={posted["id"]!r}); stop("x")']), channels=[comments])
    posted["id"] = comments.post("d.md", "please shorten").event_id
    agent.wake()
    thread = comments.thread("d.md")
    assert [e.actor_type for e in thread] == ["human", "agent"]
    assert thread[1].parent_id == posted["id"]


def test_two_agents_review_through_comments_on_one_log(tmp_path):
    log = EventLog(tmp_path / "shared.db")
    writer = Agent(tmp_path / "w", ScriptedModel(['comment("spec.md", "Drafted."); stop("x")']),
                   log=log, agent_id="writer", channels=[Comments()])
    reviewer = Agent(tmp_path / "r", ScriptedModel(['comment("spec.md", "Looks good."); stop("x")']),
                     log=log, agent_id="reviewer", channels=[Comments()])
    reviewer.follow("file:spec.md", "file")
    writer.channels["comments"].post("spec.md", "writer, please draft this")
    assert writer.wake().outcome == "stopped"
    assert [i.text for i in reviewer.pending_inbox()] == ["writer, please draft this", "Drafted."]
    reviewer.wake()
    assert writer.pending_inbox()[0].text == "Looks good."


# ── email ────────────────────────────────────────────────────────────────
def test_email_is_answered_by_email_with_history(tmp_path):
    inbox = EmailInbox()
    model = ScriptedModel([
        'send_email("ada@x.com", "Re: Meeting", "Tuesday works."); stop("x")',
        'send_email("ada@x.com", "Re: Meeting", "See you at 10."); stop("x")',
    ])
    agent = Agent(tmp_path / "ws", model, channels=[inbox])
    inbox.deliver("ada@x.com", "Meeting", "Can we meet Tuesday?")
    agent.wake()
    inbox.deliver("ada@x.com", "Meeting", "Great, what time?")
    agent.wake()
    second = ctx(model, call=1)
    assert "→ Reply in: send_email(to='ada@x.com', subject='Re: Meeting', body='...')" in second
    assert "Email from ada@x.com: Great, what time?" in second
    assert "Can we meet Tuesday?" in second and "Tuesday works." in second   # the thread so far
    assert inbox.sent == [("ada@x.com", "Re: Meeting", "Tuesday works."),
                          ("ada@x.com", "Re: Meeting", "See you at 10.")]


def test_recipes_compose_on_one_agent(tmp_path):
    comments, inbox = Comments(), EmailInbox()
    model = ScriptedModel(['comment("a.md", "ok"); send_email("b@x.com", "Re: hi", "hello"); stop("x")'])
    agent = Agent(tmp_path / "ws", model, channels=[comments, inbox])
    comments.post("a.md", "thoughts?")
    inbox.deliver("b@x.com", "hi", "hello there")
    r = agent.wake()
    w = ctx(model)
    assert "--- how to answer comments ---" in w and "--- how to answer email ---" in w
    assert r.replies == [] and r.outcome == "stopped"
