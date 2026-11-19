"""`should_respond`: the one cheap, deterministic gate in front of every wake.

"Should this participant act on this event, and where should it answer?" lives
in one auditable function. No LLM, no I/O: the same (participant, event, view,
policy) always gives the same `Decision`. The dispatcher runs it for every
subscriber of every appended event *before* a wake is spent.

It is a pre-filter, not judgment: "is this plausibly for you, and where would an
answer go?" Whether to say anything, and what, is the model's call once woken.

Two layers:

1. **Universal checks**, fixed, cheapest first. Never your own event; only
   artifacts you subscribe to; only events visible to you; and if an event is
   addressed (`payload["to"]`), only its addressees.
2. **Rules**, which you supply. An ordered list of `Rule`s; the first one that
   matches the event's surface (and optionally `op` / actor type) decides. No
   matching rule means no wake, so an unrecognised kind of event never costs
   money.

The core ships exactly one rule: a `chat` message is answered in chat (the
agent's own conversation). Everything else — comments, emails, uploads, task
status changes — is defined by whoever builds on top, usually by a `Channel`
contributing its rules.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

ANY = "*"
CHAT = "chat"
ON_ARTIFACT = "artifact"


@dataclass(frozen=True)
class Rule:
    """One routing rule. Matches on surface (and optionally op / actor type).

    `respond=True` wakes the participant; `answer` says where its response
    belongs: `"artifact"` (on the channel that handles the event's artifact
    type) or `"chat"`. `reason` is the tag the decision reports.
    """

    surface: str
    respond: bool
    answer: str = ON_ARTIFACT
    op: str = ANY
    actor_type: str = ANY
    reason: str = ""

    def matches(self, surface: str | None, op: str | None, actor_type: str | None) -> bool:
        return (
            self.surface in (ANY, surface)
            and self.op in (ANY, op)
            and self.actor_type in (ANY, actor_type)
        )


CHAT_RULE = Rule(CHAT, True, answer=CHAT, reason="chat-message")


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = (CHAT_RULE,)
    # Artifacts you talk to in a chat window: answering "on the artifact" means chat.
    chat_artifact_types: frozenset[str] = frozenset({"thread"})

    def with_rules(self, *rules: Rule) -> Policy:
        """A copy with `rules` added after the existing ones (earlier rules win)."""
        return replace(self, rules=self.rules + tuple(rules))


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
    surface: str | None  # 'chat', or the artifact type whose channel should answer
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
    payload = event.get("payload") or {}
    op = event.get("op") or payload.get("op")

    # ── universal checks ──
    if actor_type == participant.type and actor_id == participant.id:
        return _no("self-trigger")  # lets an agent's own actions live on the log without re-waking it
    if not view.subscribed:
        return _no("not-subscribed")
    if not _visible_to(event, participant):
        return _no("not-visible")
    addressed = payload.get("to")
    if addressed and participant.id not in set(addressed):
        return _no("addressed-to-others")

    # ── rules: first match wins ──
    for rule in policy.rules:
        if rule.matches(surface, op, actor_type):
            reason = rule.reason or f"{surface}{'/' + op if op else ''}"
            if not rule.respond:
                return _no(reason)
            if rule.answer == CHAT or view.type in policy.chat_artifact_types:
                return Decision(True, CHAT, reason)
            return Decision(True, view.type, reason)
    return _no("no-rule")  # safe default: an unrecognised event never costs a wake
