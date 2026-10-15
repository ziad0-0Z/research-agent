"""SQLite store: runs, sources, evidence, human reviews. (Long-term memory: memory/store.py, same file.)

Integrity rules live here as well as in the tools, so even a buggy caller
cannot save evidence for an unknown source or store the same quote twice.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from ..models import Evidence, HumanReview, Source, utcnow

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    status TEXT NOT NULL,              -- running | paused_for_review | completed | failed | blocked
    state_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sources (
    run_id TEXT NOT NULL,
    id TEXT NOT NULL,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    domain_type TEXT NOT NULL,
    published TEXT,
    origin TEXT NOT NULL,
    quality_score REAL NOT NULL,
    injection_detected INTEGER NOT NULL,
    retrieved_at TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (run_id, id)
);
CREATE INDEX IF NOT EXISTS idx_sources_url ON sources(run_id, url);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    subquestion_id TEXT NOT NULL,
    claim TEXT NOT NULL,
    quote TEXT NOT NULL,
    source_id TEXT NOT NULL,
    verified INTEGER NOT NULL,
    support TEXT NOT NULL DEFAULT 'unchecked',
    content_hash TEXT NOT NULL,
    retrieved_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    PRIMARY KEY (run_id, id),
    UNIQUE(run_id, content_hash)
);
CREATE INDEX IF NOT EXISTS idx_evidence_run ON evidence(run_id);

CREATE TABLE IF NOT EXISTS reviews (
    run_id TEXT PRIMARY KEY REFERENCES runs(id),
    reasons_json TEXT NOT NULL,
    status TEXT NOT NULL,
    reviewer_note TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    decided_at TEXT
);
"""


class UnknownSourceError(ValueError):
    pass


class DuplicateEvidenceError(ValueError):
    pass


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


BUSY_TIMEOUT_SECONDS = 30


class Store:
    def __init__(self, path: str | Path = ":memory:"):
        self.path = str(path)
        self._lock = threading.Lock()
        # The UI and a running research job each hold their own connection to the same file. A writer
        # waits up to BUSY_TIMEOUT for the other one instead of failing at once with "database is locked".
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=BUSY_TIMEOUT_SECONDS)
        self._conn.execute(f"PRAGMA busy_timeout={int(BUSY_TIMEOUT_SECONDS * 1000)}")
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def close(self) -> None:
        self._conn.close()

    def _exec(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, params)
            self._conn.commit()
            return cur

    def _exec_many(self, statements: list[tuple[str, tuple[Any, ...]]]) -> None:
        """Several statements in one transaction: all of them, or none (rolled back on error)."""
        with self._lock:
            try:
                for sql, params in statements:
                    self._conn.execute(sql, params)
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def _query(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # ---- runs ----
    def create_run(self, run_id: str, question: str) -> None:
        now = utcnow().isoformat()
        self._exec(
            "INSERT INTO runs (id, question, status, created_at, updated_at) VALUES (?,?,?,?,?)",
            (run_id, question, "running", now, now),
        )

    def update_run(self, run_id: str, *, status: str | None = None, state: dict | None = None) -> None:
        row = self.get_run(run_id)
        if row is None:
            raise KeyError(run_id)
        new_status = status or row["status"]
        new_state = json.dumps(state) if state is not None else row["state_json"]
        self._exec(
            "UPDATE runs SET status=?, state_json=?, updated_at=? WHERE id=?",
            (new_status, new_state, utcnow().isoformat(), run_id),
        )

    def latest_runs(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._query("SELECT * FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))

    def get_run(self, run_id: str) -> sqlite3.Row | None:
        rows = self._query("SELECT * FROM runs WHERE id=?", (run_id,))
        return rows[0] if rows else None

    # ---- sources ----
    def add_source(self, run_id: str, source: Source, text: str = "") -> None:
        self._exec(
            "INSERT OR REPLACE INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                run_id, source.id, str(source.url), source.title, source.domain_type,
                source.published.isoformat() if source.published else None,
                source.origin, source.quality_score, int(source.injection_detected),
                source.retrieved_at.isoformat(), text,
            ),
        )

    def get_source(self, run_id: str, source_id: str) -> Source | None:
        rows = self._query("SELECT * FROM sources WHERE run_id=? AND id=?", (run_id, source_id))
        return self._row_to_source(rows[0]) if rows else None

    def find_source_by_url(self, run_id: str, url: str) -> Source | None:
        rows = self._query("SELECT * FROM sources WHERE run_id=? AND url=?", (run_id, url))
        return self._row_to_source(rows[0]) if rows else None

    def list_sources(self, run_id: str) -> list[Source]:
        rows = self._query("SELECT * FROM sources WHERE run_id=? ORDER BY rowid", (run_id,))
        return [self._row_to_source(r) for r in rows]

    def update_source_quality(self, run_id: str, source_id: str, score: float) -> None:
        self._exec("UPDATE sources SET quality_score=? WHERE run_id=? AND id=?", (score, run_id, source_id))

    @staticmethod
    def _row_to_source(r: sqlite3.Row) -> Source:
        return Source(
            id=r["id"], url=r["url"], title=r["title"], domain_type=r["domain_type"],
            published=r["published"], origin=r["origin"], quality_score=r["quality_score"],
            injection_detected=bool(r["injection_detected"]), retrieved_at=r["retrieved_at"],
        )

    def get_source_text(self, run_id: str, source_id: str) -> str | None:
        rows = self._query("SELECT text FROM sources WHERE run_id=? AND id=?", (run_id, source_id))
        return rows[0]["text"] if rows else None

    # ---- evidence ----
    def add_evidence(self, ev: Evidence) -> None:
        if self.get_source(ev.run_id, ev.source_id) is None:
            raise UnknownSourceError(f"source {ev.source_id!r} does not exist")
        try:
            self._exec(
                "INSERT INTO evidence VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    ev.id, ev.run_id, ev.subquestion_id, ev.claim, ev.quote, ev.source_id,
                    int(ev.verified), ev.support, ev.content_hash, ev.retrieved_at.isoformat(),
                    ev.status,
                ),
            )
        except sqlite3.IntegrityError as exc:
            raise DuplicateEvidenceError(f"duplicate evidence {ev.content_hash}") from exc

    def set_support(self, run_id: str, evidence_id: str, support: str) -> None:
        if support not in ("unchecked", "supported", "partial", "not_supported"):
            raise ValueError(f"invalid support level {support!r}")
        self._exec("UPDATE evidence SET support=? WHERE run_id=? AND id=?", (support, run_id, evidence_id))

    def get_evidence(self, run_id: str, evidence_id: str) -> Evidence | None:
        rows = self._query("SELECT * FROM evidence WHERE run_id=? AND id=?", (run_id, evidence_id))
        return self._row_to_evidence(rows[0]) if rows else None

    def list_evidence(self, run_id: str) -> list[Evidence]:
        rows = self._query("SELECT * FROM evidence WHERE run_id=? ORDER BY rowid", (run_id,))
        return [self._row_to_evidence(r) for r in rows]

    @staticmethod
    def _row_to_evidence(r: sqlite3.Row) -> Evidence:
        return Evidence(
            id=r["id"], run_id=r["run_id"], subquestion_id=r["subquestion_id"], claim=r["claim"],
            quote=r["quote"], source_id=r["source_id"], verified=bool(r["verified"]),
            support=r["support"], content_hash=r["content_hash"], retrieved_at=r["retrieved_at"], status=r["status"],
        )

    # ---- human reviews ----
    def upsert_review(self, review: HumanReview) -> None:
        self._exec(
            "INSERT OR REPLACE INTO reviews VALUES (?,?,?,?,?,?)",
            (
                review.run_id, json.dumps(review.reasons), review.status, review.reviewer_note,
                review.created_at.isoformat(), _iso(review.decided_at),
            ),
        )

    def get_review(self, run_id: str) -> HumanReview | None:
        rows = self._query("SELECT * FROM reviews WHERE run_id=?", (run_id,))
        if not rows:
            return None
        r = rows[0]
        return HumanReview(
            run_id=r["run_id"], reasons=json.loads(r["reasons_json"]), status=r["status"],
            reviewer_note=r["reviewer_note"], created_at=r["created_at"], decided_at=r["decided_at"],
        )

    def pending_reviews(self) -> list[HumanReview]:
        rows = self._query("SELECT run_id FROM reviews WHERE status='pending' ORDER BY created_at")
        return [self.get_review(r["run_id"]) for r in rows]  # type: ignore[misc]
