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

The gate ships no rules at all. Every kind of event, chat included, gets its
meaning from rules a `Channel` contributes (the built-in chat channel
contributes `Rule("chat", ...)` like any other).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

ANY = "*"
ON_ARTIFACT = "artifact"


@dataclass(frozen=True)
class Rule:
    """One routing rule. Matches on surface (and optionally op / actor type).

    `respond=True` wakes the participant; `answer` says where its response
    belongs: `"artifact"` (the channel that owns the event's artifact type), or
    the name of a channel to route to instead (e.g. a chat message typed in a
    file's sidebar can still be answered in `"chat"`). `reason` is the tag the
    decision reports.
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


@dataclass(frozen=True)
class Policy:
    rules: tuple[Rule, ...] = ()

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
    surface: str | None  # where the answer goes: an artifact type, or a channel name
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
            return Decision(True, view.type if rule.answer == ON_ARTIFACT else rule.answer, reason)
    return _no("no-rule")  # safe default: an unrecognised event never costs a wake
