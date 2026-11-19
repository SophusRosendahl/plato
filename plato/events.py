"""The event log: one append-only record of everything every actor does.

Every action — a person's chat message, a comment, an edit, a file landing, the
agent's own replies — is an immutable event on one log. Everything else is
derived from it:

    submit()            append an event to an artifact's stream
    subscribe()         who follows an artifact (the dispatcher's fan-out list)
    dispatch_once()     tail new events → for each subscriber run the cheap
                        `should_respond` gate → materialize the yeses into that
                        participant's inbox
    pending_inbox()     what a participant hasn't handled yet
    recent()            the tail of one artifact (a thread, a document's comments)

An **artifact** is anything events attach to: a chat thread, a document, an
email conversation. It has a type, which the gate uses to decide where a
response belongs.

SQLite (stdlib) keeps this a single file you can open and query. Several agents
can share one log: each subscribes to the artifacts it follows.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from plato.should_respond import ArtifactView, Decision, Participant, Policy, should_respond

ACTOR_TYPES = frozenset({"human", "agent", "system"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id    TEXT UNIQUE NOT NULL,
    artifact_id TEXT NOT NULL,
    actor_type  TEXT NOT NULL,
    actor_id    TEXT NOT NULL,
    surface     TEXT NOT NULL,
    op          TEXT,
    parent_id   TEXT,
    visibility  TEXT NOT NULL DEFAULT 'all',
    payload     TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_artifact ON events (artifact_id, seq);
CREATE TABLE IF NOT EXISTS artifacts (
    artifact_id TEXT PRIMARY KEY,
    type        TEXT NOT NULL,
    owner_id    TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS subscriptions (
    participant_id TEXT NOT NULL,
    artifact_id    TEXT NOT NULL,
    PRIMARY KEY (participant_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS inbox_items (
    participant_id TEXT NOT NULL,
    event_id       TEXT NOT NULL,
    seq            INTEGER NOT NULL,
    reply_surface  TEXT,
    reason         TEXT,
    done           INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (participant_id, event_id)
);
CREATE TABLE IF NOT EXISTS cursors (
    name TEXT PRIMARY KEY,
    seq  INTEGER NOT NULL
);
"""


@dataclass
class Event:
    seq: int
    event_id: str
    artifact_id: str
    actor_type: str
    actor_id: str
    surface: str
    op: str | None
    parent_id: str | None
    visibility: str
    payload: dict
    created_at: str
    # Filled in when read as an inbox item: where the answer belongs, and why it woke.
    reply_surface: str | None = None
    reason: str | None = None
    artifact_type: str | None = field(default=None)

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    def as_gate_input(self) -> dict:
        return {
            "surface": self.surface, "actor_type": self.actor_type, "actor_id": self.actor_id,
            "op": self.op, "visibility": self.visibility, "payload": self.payload,
        }


@dataclass
class Dispatched:
    """One gate decision, kept for observability (why did / didn't this wake?)."""
    event: Event
    participant: str
    decision: Decision


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: str | Path, policy: Policy | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.policy = policy or Policy()
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)

    # ── write path ───────────────────────────────────────────────────────
    def ensure_artifact(self, artifact_id: str, type: str, owner_id: str | None = None) -> None:  # noqa: A002
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO artifacts (artifact_id, type, owner_id, created_at) VALUES (?,?,?,?)",
                (artifact_id, type, owner_id, _now()),
            )

    def artifact_type(self, artifact_id: str) -> str:
        row = self._db.execute("SELECT type FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
        return row["type"] if row else "unknown"

    def submit(
        self,
        artifact_id: str,
        *,
        actor_type: str,
        actor_id: str,
        surface: str,
        payload: dict | None = None,
        op: str | None = None,
        parent_id: str | None = None,
        visibility: str = "all",
    ) -> Event:
        """Append one immutable event. Raises on a malformed event: silently
        dropping a source-of-truth write is worse than a loud failure."""
        if actor_type not in ACTOR_TYPES:
            raise ValueError(f"actor_type must be one of {sorted(ACTOR_TYPES)}")
        if surface not in self.policy.surfaces:
            raise ValueError(f"unknown surface {surface!r}; known: {sorted(self.policy.surfaces)}")
        event_id = "e" + uuid.uuid4().hex[:10]
        with self._lock:
            if self._db.execute("SELECT 1 FROM artifacts WHERE artifact_id=?", (artifact_id,)).fetchone() is None:
                raise KeyError(f"unknown artifact {artifact_id!r}; call ensure_artifact() first")
            self._db.execute(
                "INSERT INTO events (event_id, artifact_id, actor_type, actor_id, surface, op, parent_id,"
                " visibility, payload, created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                (event_id, artifact_id, actor_type, actor_id, surface, op, parent_id, visibility,
                 json.dumps(payload or {}), _now()),
            )
        return self.get(event_id)

    # ── subscriptions ────────────────────────────────────────────────────
    def subscribe(self, participant_id: str, artifact_id: str) -> None:
        with self._lock:
            self._db.execute("INSERT OR IGNORE INTO subscriptions VALUES (?,?)", (participant_id, artifact_id))

    def unsubscribe(self, participant_id: str, artifact_id: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM subscriptions WHERE participant_id=? AND artifact_id=?",
                             (participant_id, artifact_id))

    def subscribers(self, artifact_id: str) -> list[str]:
        rows = self._db.execute("SELECT participant_id FROM subscriptions WHERE artifact_id=?", (artifact_id,))
        return [r["participant_id"] for r in rows]

    # ── read path ────────────────────────────────────────────────────────
    def _event(self, row: sqlite3.Row) -> Event:
        d = dict(row)
        extra = {k: d.pop(k) for k in ("reply_surface", "reason", "done", "participant_id") if k in d}
        ev = Event(**{**d, "payload": json.loads(d["payload"])})
        ev.reply_surface, ev.reason = extra.get("reply_surface"), extra.get("reason")
        ev.artifact_type = self.artifact_type(ev.artifact_id)
        return ev

    def get(self, event_id: str) -> Event:
        row = self._db.execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(event_id)
        return self._event(row)

    def recent(
        self, artifact_id: str, limit: int = 20, surfaces: tuple[str, ...] | None = None,
        exclude: set[str] | None = None,
    ) -> list[Event]:
        """The tail of one artifact's stream, oldest first."""
        sql, args = "SELECT * FROM events WHERE artifact_id=?", [artifact_id]
        if surfaces:
            sql += f" AND surface IN ({','.join('?' * len(surfaces))})"
            args += list(surfaces)
        rows = [self._event(r) for r in self._db.execute(sql + " ORDER BY seq DESC", args)]
        rows = [e for e in rows if not exclude or e.event_id not in exclude]
        return list(reversed(rows[:limit]))

    def count(self, artifact_id: str, surfaces: tuple[str, ...] | None = None,
              exclude: set[str] | None = None) -> int:
        return len(self.recent(artifact_id, limit=10**9, surfaces=surfaces, exclude=exclude))

    # ── dispatcher ───────────────────────────────────────────────────────
    def dispatch_once(self, cursor: str = "dispatcher") -> list[Dispatched]:
        """Fan new events out to subscribers through the gate; materialize the
        yeses into inboxes. Idempotent (the inbox PK dedups a retried event),
        and the cursor only advances past fully handled events."""
        with self._lock:
            row = self._db.execute("SELECT seq FROM cursors WHERE name=?", (cursor,)).fetchone()
            after = row["seq"] if row else 0
            events = [self._event(r) for r in
                      self._db.execute("SELECT * FROM events WHERE seq>? ORDER BY seq", (after,))]
            out: list[Dispatched] = []
            for ev in events:
                view_type = ev.artifact_type or "unknown"
                for pid in self.subscribers(ev.artifact_id):
                    decision = should_respond(
                        Participant(pid), ev.as_gate_input(), ArtifactView(view_type, True), self.policy)
                    out.append(Dispatched(ev, pid, decision))
                    if decision:
                        self._db.execute(
                            "INSERT OR IGNORE INTO inbox_items (participant_id, event_id, seq, reply_surface,"
                            " reason) VALUES (?,?,?,?,?)",
                            (pid, ev.event_id, ev.seq, decision.surface, decision.reason))
                self._db.execute("INSERT INTO cursors VALUES (?,?) ON CONFLICT(name) DO UPDATE SET seq=?",
                                 (cursor, ev.seq, ev.seq))
            return out

    # ── inbox (derived from the log) ─────────────────────────────────────
    def pending_inbox(self, participant_id: str) -> list[Event]:
        rows = self._db.execute(
            "SELECT e.*, i.reply_surface, i.reason FROM inbox_items i JOIN events e USING (event_id)"
            " WHERE i.participant_id=? AND i.done=0 ORDER BY e.seq", (participant_id,))
        return [self._event(r) for r in rows]

    def mark_done(self, participant_id: str, event_ids: set[str]) -> None:
        """Flip exactly the items the participant was shown; anything that
        arrived after rendering stays pending."""
        with self._lock:
            self._db.executemany(
                "UPDATE inbox_items SET done=1 WHERE participant_id=? AND event_id=?",
                [(participant_id, eid) for eid in event_ids])
