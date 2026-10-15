"""Long-term memory tables (same SQLite file as the run store, separate tables).

Separate from the `evidence` table on purpose: deleting or retiring memory never touches a run's
audit trail, and every memory write goes through this one class.

    memory_evidence   one verified evidence item, keyed by content_hash(url, quote)
    memory_pages      page snapshot the quote was verified against (the output guardrail re-checks it)
    memory_reports    released reports of completed runs (shown as "related past research" only)
    memory_context    facts the user stated verbatim in a question (team size, budget) - shown only
    memory_tombstones items a human deleted: an old run committed again cannot bring them back
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from ..store import Store

SCHEMA = """
CREATE TABLE IF NOT EXISTS memory_pages (
    page_hash TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS memory_evidence (
    mem_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    claim TEXT NOT NULL,
    quote TEXT NOT NULL,
    claim_type TEXT NOT NULL,              -- pricing | market | risk | technical
    specialist TEXT NOT NULL,
    subquestion_text TEXT NOT NULL,
    source_url TEXT NOT NULL,
    source_host TEXT NOT NULL,
    source_title TEXT NOT NULL,
    source_domain_type TEXT NOT NULL,
    source_published TEXT,
    source_quality REAL NOT NULL,
    origin TEXT NOT NULL,                  -- where the page originally came from: tavily | ddg | offline
    page_hash TEXT NOT NULL REFERENCES memory_pages(page_hash),
    support TEXT NOT NULL,                 -- supported | partial | unjudged
    first_run_id TEXT NOT NULL,
    last_run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    last_verified_at TEXT NOT NULL,
    expires_at TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'active', -- active | superseded | unverifiable | retired
    status_reason TEXT NOT NULL DEFAULT '',
    superseded_by TEXT,
    contested_with TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_memev_status ON memory_evidence(status);
CREATE INDEX IF NOT EXISTS idx_memev_url ON memory_evidence(source_url);

CREATE TABLE IF NOT EXISTS memory_reports (
    run_id TEXT PRIMARY KEY,
    question TEXT NOT NULL,
    title TEXT NOT NULL,
    report_json TEXT NOT NULL,
    final_markdown TEXT NOT NULL,
    confidence REAL NOT NULL,
    review_status TEXT NOT NULL,           -- released (no review needed) | approved
    released_at TEXT NOT NULL,
    mem_ids TEXT NOT NULL DEFAULT '[]'
);

CREATE TABLE IF NOT EXISTS memory_context (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    stated_span TEXT NOT NULL,
    run_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(key, value, run_id)
);

CREATE TABLE IF NOT EXISTS memory_tombstones (
    content_hash TEXT PRIMARY KEY,
    mem_id TEXT NOT NULL,
    deleted_at TEXT NOT NULL
);
"""

MEMORY_STATUSES = ("active", "superseded", "unverifiable", "retired")


@dataclass
class MemoryItem:
    mem_id: str
    content_hash: str
    claim: str
    quote: str
    claim_type: str
    specialist: str
    subquestion_text: str
    source_url: str
    source_host: str
    source_title: str
    source_domain_type: str
    source_published: str | None
    source_quality: float
    origin: str
    page_hash: str
    support: str
    first_run_id: str
    last_run_id: str
    created_at: datetime
    last_verified_at: datetime
    expires_at: datetime
    status: str = "active"
    status_reason: str = ""
    superseded_by: str | None = None
    contested_with: list[str] = field(default_factory=list)

    def is_fresh(self, now: datetime) -> bool:
        return now < self.expires_at

    def as_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        for k in ("created_at", "last_verified_at", "expires_at"):
            d[k] = d[k].isoformat()
        return d


class MemoryStore:
    """Thin layer over the run Store's connection: one DB file, one lock."""

    def __init__(self, store: Store):
        self.store = store
        with store._lock:
            store._conn.executescript(SCHEMA)
            store._conn.commit()

    # ---- pages ----
    def put_page(self, page_hash: str, url: str, title: str, text: str, fetched_at: datetime) -> None:
        self.store._exec("INSERT OR REPLACE INTO memory_pages VALUES (?,?,?,?,?)",
                         (page_hash, url, title, text, fetched_at.isoformat()))

    def get_page(self, page_hash: str) -> sqlite3.Row | None:
        rows = self.store._query("SELECT * FROM memory_pages WHERE page_hash=?", (page_hash,))
        return rows[0] if rows else None

    # ---- evidence ----
    def get(self, mem_id: str) -> MemoryItem | None:
        rows = self.store._query("SELECT * FROM memory_evidence WHERE mem_id=?", (mem_id,))
        return _row_to_item(rows[0]) if rows else None

    def get_by_hash(self, content_hash: str) -> MemoryItem | None:
        rows = self.store._query("SELECT * FROM memory_evidence WHERE content_hash=?", (content_hash,))
        return _row_to_item(rows[0]) if rows else None

    def list(self, *, status: str | None = None, claim_type: str | None = None) -> list[MemoryItem]:
        sql, params = "SELECT * FROM memory_evidence WHERE 1=1", []
        if status:
            sql += " AND status=?"
            params.append(status)
        if claim_type:
            sql += " AND claim_type=?"
            params.append(claim_type)
        rows = self.store._query(sql + " ORDER BY created_at, mem_id", tuple(params))
        return [_row_to_item(r) for r in rows]

    def by_url(self, url: str, status: str = "active") -> list[MemoryItem]:
        rows = self.store._query("SELECT * FROM memory_evidence WHERE source_url=? AND status=? ORDER BY created_at",
                                 (url.rstrip("/"), status))
        return [_row_to_item(r) for r in rows]

    def insert(self, item: MemoryItem) -> None:
        self.store._exec(
            "INSERT INTO memory_evidence VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (item.mem_id, item.content_hash, item.claim, item.quote, item.claim_type, item.specialist,
             item.subquestion_text, item.source_url, item.source_host, item.source_title,
             item.source_domain_type, item.source_published, item.source_quality, item.origin, item.page_hash,
             item.support, item.first_run_id, item.last_run_id, item.created_at.isoformat(),
             item.last_verified_at.isoformat(), item.expires_at.isoformat(), item.status, item.status_reason,
             item.superseded_by, json.dumps(item.contested_with)))

    def update(self, mem_id: str, **fields: Any) -> None:
        allowed = {"support", "last_run_id", "last_verified_at", "expires_at", "status", "status_reason",
                   "superseded_by", "contested_with", "page_hash", "source_quality"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"cannot update memory fields {sorted(bad)}")
        if "status" in fields and fields["status"] not in MEMORY_STATUSES:
            raise ValueError(f"invalid memory status {fields['status']!r}")
        values = []
        for k, v in fields.items():
            if isinstance(v, datetime):
                v = v.isoformat()
            elif k == "contested_with":
                v = json.dumps(sorted(set(v)))
            values.append(v)
        sets = ", ".join(f"{k}=?" for k in fields)
        self.store._exec(f"UPDATE memory_evidence SET {sets} WHERE mem_id=?", (*values, mem_id))

    def delete(self, mem_id: str, now: datetime) -> bool:
        """Hard delete + tombstone, so committing an old run again cannot resurrect the item."""
        item = self.get(mem_id)
        if item is None:
            return False
        # one transaction: a busy database can never leave a tombstone without the delete (or the reverse)
        self.store._exec_many([
            ("INSERT OR REPLACE INTO memory_tombstones VALUES (?,?,?)", (item.content_hash, mem_id, now.isoformat())),
            ("DELETE FROM memory_evidence WHERE mem_id=?", (mem_id,)),
        ])
        return True

    def purge(self, mem_id: str) -> None:
        """Automatic clean-up (expired / unverifiable): no tombstone, so new research may learn it again."""
        self.store._exec("DELETE FROM memory_evidence WHERE mem_id=?", (mem_id,))

    def is_tombstoned(self, content_hash: str) -> bool:
        return bool(self.store._query("SELECT 1 FROM memory_tombstones WHERE content_hash=?", (content_hash,)))

    # ---- reports ----
    def put_report(self, *, run_id: str, question: str, title: str, report_json: str, final_markdown: str,
                   confidence: float, review_status: str, released_at: datetime, mem_ids: list[str]) -> bool:
        cur = self.store._exec(
            "INSERT OR IGNORE INTO memory_reports VALUES (?,?,?,?,?,?,?,?,?)",
            (run_id, question, title, report_json, final_markdown, confidence, review_status,
             released_at.isoformat(), json.dumps(mem_ids)))
        return cur.rowcount == 1

    def reports(self) -> list[sqlite3.Row]:
        return self.store._query("SELECT * FROM memory_reports ORDER BY released_at")

    # ---- user-stated context ----
    def put_context(self, key: str, value: str, span: str, run_id: str, now: datetime) -> None:
        self.store._exec("INSERT OR IGNORE INTO memory_context (key, value, stated_span, run_id, created_at) "
                         "VALUES (?,?,?,?,?)", (key, value, span, run_id, now.isoformat()))

    def context(self) -> list[sqlite3.Row]:
        return self.store._query("SELECT * FROM memory_context ORDER BY created_at")

    # ---- stats ----
    def stats(self) -> dict[str, Any]:
        by_status = {r["status"]: r["n"] for r in self.store._query(
            "SELECT status, COUNT(*) n FROM memory_evidence GROUP BY status")}
        by_type = {r["claim_type"]: r["n"] for r in self.store._query(
            "SELECT claim_type, COUNT(*) n FROM memory_evidence WHERE status='active' GROUP BY claim_type")}
        n_reports = self.store._query("SELECT COUNT(*) n FROM memory_reports")[0]["n"]
        n_pages = self.store._query("SELECT COUNT(*) n FROM memory_pages")[0]["n"]
        n_ctx = self.store._query("SELECT COUNT(*) n FROM memory_context")[0]["n"]
        n_tomb = self.store._query("SELECT COUNT(*) n FROM memory_tombstones")[0]["n"]
        return {"evidence_by_status": by_status, "active_by_type": by_type, "reports": n_reports,
                "pages": n_pages, "context_facts": n_ctx, "deleted": n_tomb}


def _row_to_item(r: sqlite3.Row) -> MemoryItem:
    return MemoryItem(
        mem_id=r["mem_id"], content_hash=r["content_hash"], claim=r["claim"], quote=r["quote"],
        claim_type=r["claim_type"], specialist=r["specialist"], subquestion_text=r["subquestion_text"],
        source_url=r["source_url"], source_host=r["source_host"], source_title=r["source_title"],
        source_domain_type=r["source_domain_type"], source_published=r["source_published"],
        source_quality=r["source_quality"], origin=r["origin"], page_hash=r["page_hash"], support=r["support"],
        first_run_id=r["first_run_id"], last_run_id=r["last_run_id"],
        created_at=datetime.fromisoformat(r["created_at"]),
        last_verified_at=datetime.fromisoformat(r["last_verified_at"]),
        expires_at=datetime.fromisoformat(r["expires_at"]), status=r["status"], status_reason=r["status_reason"],
        superseded_by=r["superseded_by"], contested_with=json.loads(r["contested_with"] or "[]"))
