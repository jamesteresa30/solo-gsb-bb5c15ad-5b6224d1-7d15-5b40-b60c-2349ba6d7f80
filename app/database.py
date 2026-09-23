"""SQLite connection handling, schema and initial seeding."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from pathlib import Path

from .seed import seed_items

# Default keeps local runs working out of the box; the Docker image sets this
# to /data/quiz.db (a named volume) so data survives container restarts.
DB_PATH = Path(os.getenv("QUIZ_DB_PATH", "data/quiz.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id             TEXT PRIMARY KEY,
    difficulty     REAL NOT NULL,
    discrimination REAL NOT NULL,
    content_tag    TEXT,
    stem           TEXT NOT NULL,
    options        TEXT NOT NULL, -- JSON: [{"key": "A", "text": "..."}]
    answer         TEXT NOT NULL, -- correct option key, e.g. "B"
    exposure_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sessions (
    id               TEXT PRIMARY KEY,
    theta            REAL NOT NULL,
    current_item_id  TEXT REFERENCES items(id), -- item currently served, not yet answered
    max_items        INTEGER NOT NULL,
    exposure_limit   INTEGER NOT NULL,
    duration_seconds INTEGER, -- optional time limit in seconds (60..86400); NULL = no limit
    expires_at       TEXT,    -- UTC ISO-8601 deadline (created_at + duration_seconds)
    status           TEXT NOT NULL DEFAULT 'active'
                     CHECK (status IN ('active', 'finished')),
    finish_reason    TEXT, -- 'max_items' | 'no_available_items'
                     -- 'time_expired' | 'tag_minimums_unfulfillable'
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);

-- Optional per-session minimum answer quotas keyed by content tag.  A new
-- table (created IF NOT EXISTS) also serves as the migration: databases from
-- older versions simply acquire the table on startup and existing sessions
-- have no rows, so their behaviour is unchanged.
CREATE TABLE IF NOT EXISTS session_tag_minimums (
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    content_tag TEXT NOT NULL,
    minimum     INTEGER NOT NULL CHECK (minimum >= 1),
    PRIMARY KEY (session_id, content_tag)
);

CREATE TABLE IF NOT EXISTS responses (
    id                        TEXT PRIMARY KEY, -- client-supplied idempotency key
    session_id                TEXT NOT NULL REFERENCES sessions(id),
    item_id                   TEXT NOT NULL REFERENCES items(id),
    response                  TEXT NOT NULL,
    score                     INTEGER NOT NULL CHECK (score IN (0, 1)),
    correct_answer            TEXT NOT NULL,
    theta_before              REAL NOT NULL,
    theta_after               REAL NOT NULL,
    p_correct                 REAL NOT NULL,
    information_at_theta      REAL NOT NULL,
    exposure_count_at_select  INTEGER NOT NULL,
    created_at                TEXT NOT NULL,
    UNIQUE (session_id, item_id) -- an item can only be answered once per session
);

CREATE INDEX IF NOT EXISTS idx_responses_session ON responses(session_id);
"""


def get_db() -> Iterator[sqlite3.Connection]:
    """FastAPI dependency: a per-request SQLite connection.

    ``check_same_thread=False`` is required because FastAPI may run a sync
    dependency's enter, the endpoint and its exit on different threadpool
    workers under load.  Usage is still strictly serial (anyio awaits each
    stage), so a connection is never touched by two threads at once.
    ``busy_timeout`` makes competing BEGIN IMMEDIATE writers wait in a queue
    instead of raising "database is locked" immediately.
    """
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    try:
        yield conn
    finally:
        conn.close()


def _ensure_column(conn: sqlite3.Connection, table: str, name: str, decl: str) -> None:
    """Add a column to an existing table if it is missing (idempotent)."""
    columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
    if name not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")


def init_db() -> None:
    """Create tables (idempotent) and seed a deterministic demo bank once."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(str(DB_PATH)) as conn:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        # Migrate databases created before the optional session time limit.
        _ensure_column(conn, "sessions", "duration_seconds", "INTEGER")
        _ensure_column(conn, "sessions", "expires_at", "TEXT")
        count = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        if count == 0:
            conn.executemany(
                """
                INSERT INTO items
                    (id, difficulty, discrimination, content_tag, stem,
                     options, answer, exposure_count)
                VALUES
                    (:id, :difficulty, :discrimination, :content_tag, :stem,
                     :options, :answer, 0)
                """,
                seed_items(),
            )
        conn.commit()
