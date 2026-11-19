"""Channels: comment events, their context tail, reply routing, chat suppression."""

from plato import Agent, ScriptedModel, ToolRegistry
from plato.channels import Channel, CommentsChannel


def make(tmp_path, steps, **kw):
    model = ScriptedModel(steps)
    comments = CommentsChannel(**kw)
    agent = Agent(tmp_path / "ws", model, channels=[comments])
    return agent, comments, model


def wake_context(model, call=0):
    return model.calls[call][1]["content"]


def handled(agent):
    """Dispatch and mark everything pending as seen, without running a wake."""
    agent.log.dispatch_once()
    agent.log.mark_done(agent.agent_id, {e.event_id for e in agent.log.pending_inbox(agent.agent_id)})


def test_comment_lands_in_inbox_with_route_and_rules(tmp_path):
    agent, comments, model = make(tmp_path, [
        'comment("notes.md", "Fixed.", parent_id="x"); stop("answered comment")'])
    (agent.ws.files / "notes.md").write_text("line one\nline two\n")
    c = comments.post("notes.md", "Typo in line two?", quote="line two")
    agent.wake()
    ctx = wake_context(model)
    assert f"→ Reply in: comment(entity='notes.md', text='...', parent_id='{c.event_id}')" in ctx
    assert "Comment on notes.md: Typo in line two?" in ctx
    assert "> line two" in ctx                                       # quoted passage
    assert "Document: notes.md (18 chars, 3 lines)" in ctx             # reference, not body
    assert "line one" not in ctx                                     # body is read on demand
    assert ctx.count("--- how to answer comments ---") == 1
    assert "Respond in the channel named" in ctx                     # channel footer


def test_thread_tail_is_bounded_and_excludes_trigger(tmp_path):
    agent, comments, model = make(tmp_path, ['comment("d.md", "ok"); stop("x")'],
                                  max_thread=2, max_comment_chars=20)
    root = comments.post("d.md", "first")
    agent.log.submit("doc:d.md", actor_type="agent", actor_id="agent", surface="comment",
                     parent_id=root.event_id, payload={"text": "agent answer", "entity": "d.md"})
    comments.post("d.md", "a very long follow-up comment indeed", parent_id=root.event_id)
    handled(agent)
    comments.post("d.md", "and now?", parent_id=root.event_id)
    agent.wake()
    ctx = wake_context(model)
    assert "most recent 2 of 3 comments" in ctx
    assert "- You (" in ctx and "agent answer" in ctx
    assert "a very long follow-…" in ctx                             # per-comment cap
    assert "): first" not in ctx                                     # oldest dropped
    assert ctx.count("and now?") == 1                                # trigger not repeated in tail
    assert f"parent_id='{root.event_id}'" in ctx                     # replies go to thread root


def test_agent_comment_lands_in_thread_and_does_not_rewake(tmp_path):
    posted = {}
    agent, comments, _ = make(tmp_path, [
        lambda _: f'comment("d.md", "Will do.", parent_id={posted["id"]!r}); stop("x")'])
    posted["id"] = comments.post("d.md", "please shorten").event_id
    r = agent.wake()
    assert r.outcome == "stopped"
    thread = comments.thread("d.md")
    assert [x.actor_type for x in thread] == ["human", "agent"]
    assert thread[1].parent_id == posted["id"]
    assert agent.pending_inbox() == []                               # its own comment didn't wake it


def test_chat_reply_is_withheld_on_a_channel_wake(tmp_path):
    agent, comments, model = make(tmp_path, [
        'reply("Answered in chat by mistake")',
        'comment("d.md", "Answered here."); stop("x")',
    ])
    comments.post("d.md", "question?")
    r = agent.wake()
    assert r.replies == []                                           # nothing reached chat
    tool_out = [m["content"] for m in model.calls[1] if m["role"] == "tool"][-1]
    assert "not delivered" in tool_out
    kinds = [ln.split('"kind": "')[1].split('"')[0] for ln in r.trace_path.read_text().splitlines()]
    assert "reply_suppressed" in kinds and "channel_reply" in kinds
    assert all(e.actor_type == "human" for e in agent.log.recent(agent.thread))  # nothing logged to chat


def test_mixed_batch_keeps_chat_live(tmp_path):
    agent, comments, model = make(tmp_path, ['reply("hi"); comment("d.md", "ok"); stop("x")'])
    comments.post("d.md", "question?")
    agent.send("hello in chat")
    r = agent.wake()
    assert r.replies == ["hi"]
    assert "Your inbox has new items" in wake_context(model)


def test_chat_message_mid_run_lifts_suppression(tmp_path):
    agent = None

    def chat_arrives(_):
        agent.send("also, quick question in chat")
        return 'print("working")'

    model = ScriptedModel(['comment("d.md", "on it")', chat_arrives, 'reply("sure"); stop("x")'])
    comments = CommentsChannel()
    agent = Agent(tmp_path / "ws", model, channels=[comments])
    comments.post("d.md", "please fix")
    r = agent.wake()
    assert r.replies == ["sure"]


def test_comment_counts_as_an_answer_for_the_silent_stop_guard(tmp_path):
    agent, comments, _ = make(tmp_path, ['comment("d.md", "Thanks!"); stop("acked")'])
    comments.post("d.md", "thanks, looks great")
    r = agent.wake()
    assert r.outcome == "stopped" and r.steps == 1


class Email(Channel):
    name = "email"
    artifact_type = "email"

    def __init__(self):
        self.sent = []

    def receive(self, sender: str, subject: str, body: str):
        artifact = f"email:{sender}"
        self.agent.follow(artifact, self.artifact_type)
        return self.log.submit(artifact, actor_type="human", actor_id=sender, surface="message",
                               payload={"text": body, "from": sender, "subject": subject})

    def tools(self):
        reg = ToolRegistry()

        @reg.tool(group="Email")
        def send_email(to: str, body: str) -> str:
            """Send an email."""
            self.sent.append((to, body))
            self.agent._emit("channel_reply", channel="email", to=to)
            return "sent"
        return reg

    def label(self, event):
        return f"Email from {event.payload['from']}"

    def reply_hint(self, item):
        return f"send_email(to={item.meta['from']!r}, body='...')"

    def tail(self, item):
        return f"Subject: {item.meta['subject']}"


def test_custom_channel_plugs_in(tmp_path):
    email = Email()
    model = ScriptedModel(['send_email("ada@example.com", "Yes, Tuesday works."); stop("x")'])
    agent = Agent(tmp_path / "ws", model, channels=[email])
    email.receive("ada@example.com", "Meeting", "Can we meet Tuesday?")
    agent.wake()
    ctx = wake_context(model)
    assert "→ Reply in: send_email(to='ada@example.com', body='...')" in ctx
    assert "Email from ada@example.com: Can we meet Tuesday?" in ctx
    assert "Subject: Meeting" in ctx
    assert "send_email(to, body)" in agent.tool_schema["function"]["description"]
    assert email.sent == [("ada@example.com", "Yes, Tuesday works.")]
