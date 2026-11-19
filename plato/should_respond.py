"""`should_respond`: the one cheap, deterministic gate in front of every wake.

"Should this participant act on this event, and where should it answer?" lives
in one auditable function. No LLM, no I/O: given the same (participant, event,
view, policy) it always returns the same `Decision`. The dispatcher runs it per
subscriber on every appended event *before* spending a wake, so it must stay
cheap.

It is deliberately a pre-filter, not judgment. It only answers "is this plausibly
for you, and on which surface?" (not yourself? subscribed? visible? a surface
that asks for a response?). Whether to actually say something, and what, is the
model's call once woken.

Rules that recur across artifact types are parameters on `Policy`; the rest is
fixed, ordered cheapest-and-most-suppressive first, and every branch carries a
`reason` tag so a decision can explain itself in a log line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_SURFACES = frozenset({
    "chat",      # a conversational message on a chat thread (or a doc's chat sidebar)
    "comment",   # a comment on an artifact
    "message",   # an inbound message on an external channel (email, Slack, a webhook)
    "edit",      # a change to an artifact's content
    "status",    # lifecycle / bookkeeping (op says which: created, upload, task moved…)
    "mention",   # @-mention of a participant
    "done",      # an actor's own mark-done bookkeeping
})


@dataclass(frozen=True)
class Policy:
    surfaces: frozenset[str] = DEFAULT_SURFACES
    # Artifacts you talk to in a chat window: a response goes to chat. On every
    # other artifact a response belongs on that artifact's own channel.
    chat_artifact_types: frozenset[str] = frozenset({"thread"})
    # Does a person editing a document wake the agent? Default no: the agent is
    # not a proactive reviewer of work in progress, and autosave would cost a
    # wake per pause. The edit still lands on the log.
    wake_on_edit: bool = False
    # Human `status` ops that are bookkeeping, not a request (no new content).
    non_waking_ops: frozenset[str] = field(
        default_factory=lambda: frozenset({"artifact.created", "create", "rename", "delete", "attach"}))


@dataclass(frozen=True)
class Participant:
    id: str
    type: str = "agent"  # 'agent' | 'human' | 'system'


@dataclass(frozen=True)
class ArtifactView:
    """The cheap facts about the artifact the event landed on."""
    type: str
    subscribed: bool = True


@dataclass(frozen=True)
class Decision:
    respond: bool
    surface: str | None  # 'chat' or the artifact type whose channel should answer
    reason: str

    def __bool__(self) -> bool:
        return self.respond


def _no(reason: str) -> Decision:
    return Decision(False, None, reason)


def _visible_to(event: dict, participant: Participant) -> bool:
    vis = event.get("visibility") or "all"
    if vis == "all":
        return True
    return participant.id in {v.strip() for v in vis.split(",") if v.strip()}


def should_respond(
    participant: Participant, event: dict, view: ArtifactView, policy: Policy | None = None
) -> Decision:
    policy = policy or Policy()
    surface, actor_type, actor_id = event.get("surface"), event.get("actor_type"), event.get("actor_id")
    op = event.get("op") or (event.get("payload") or {}).get("op")

    # Never wake on your own event. (Lets the agent's replies live on the same
    # log without re-waking it.)
    if actor_type == participant.type and actor_id == participant.id:
        return _no("self-trigger")
    if not view.subscribed:
        return _no("not-subscribed")
    if not _visible_to(event, participant):
        return _no("not-visible")
    if surface not in policy.surfaces:
        return _no("unknown-surface")

    on_artifact = "chat" if view.type in policy.chat_artifact_types else view.type

    # A chat message is answered in chat, whatever artifact it was typed on.
    if surface == "chat":
        return Decision(True, "chat", "chat-message")
    # A mention wakes only the participants it names (no names = the mention is the signal).
    if surface == "mention":
        named = frozenset((event.get("payload") or {}).get("mentions") or ())
        if named and participant.id not in named:
            return _no("mention-not-me")
        return Decision(True, on_artifact, "mentioned")
    if surface == "comment":
        return Decision(True, on_artifact, "comment")
    if surface == "message":
        return Decision(True, on_artifact, "inbound-message")
    if surface == "edit":
        return Decision(True, on_artifact, "edit") if policy.wake_on_edit else _no("edit-no-wake")
    if surface == "status":
        if op in policy.non_waking_ops:
            return _no(f"status-{op}")
        if actor_type == "human":  # e.g. a file uploaded, a task moved to blocked
            return Decision(True, on_artifact, "status-change")
        return _no("system-status")
    if surface == "done":
        return _no("done-event")
    return _no("unhandled-surface")  # safe default: an unrecognised surface never costs a wake
