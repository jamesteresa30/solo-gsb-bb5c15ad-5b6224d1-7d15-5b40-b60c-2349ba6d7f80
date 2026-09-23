"""FastAPI application: adaptive quiz item selection engine.

Flow
----
1. ``POST /api/sessions``            create an examinee session
2. ``POST /api/sessions/{id}/next``  get the most informative eligible item
3. ``POST /api/sessions/{id}/responses``  submit an answer; theta is updated
4. repeat 2-3 until the session finishes (max items or no eligible item)

Mutating endpoints use ``BEGIN IMMEDIATE`` so concurrent requests on the same
SQLite database are serialised; exposure counts and ability updates therefore
cannot be lost to read/write races.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from . import irt
from .database import get_db, init_db
from .models import (
    BankItemOut,
    HealthOut,
    ItemBankOut,
    ItemDetailOut,
    ItemOut,
    NextItemOut,
    ProgressOut,
    ResponseCreate,
    ResponseHistoryOut,
    ResponseOut,
    SessionCreate,
    SessionOut,
)

DEFAULT_EXPOSURE_LIMIT = int(os.getenv("DEFAULT_EXPOSURE_LIMIT", "5"))

app = FastAPI(
    title="自适应测验选题 API",
    description=(
        "基于 2PL IRT 模型的计算机自适应测验（CAT）："
        "单次 Newton 更新估计能力，信息量最大选题，全局曝光上限控制。"
    ),
    version="1.0.0",
)


# -------------------------------------------------------------------------
# Lifecycle / small helpers
# -------------------------------------------------------------------------

@app.on_event("startup")
def _startup() -> None:
    init_db()


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[None]:
    """Serialised write transaction (BEGIN IMMEDIATE) with explicit rollback."""
    conn.isolation_level = None  # autocommit mode: we manage the transaction
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_utc(value: str) -> datetime:
    """Parse a stored ISO-8601 UTC timestamp into an aware datetime."""
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:  # defensive: timestamps are always written as UTC
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def expire_if_due(
    conn: sqlite3.Connection,
    session: sqlite3.Row,
    now: datetime,
) -> sqlite3.Row | None:
    """Finish an active session whose deadline has passed.

    Must be called first thing inside a serialised write transaction.  When
    the session has an ``expires_at`` in the past it is marked finished with
    ``finish_reason = 'time_expired'`` and the pending item is cleared; the
    pending item does *not* get another exposure (selection already counted
    one) and no response row is written.

    The finishing UPDATE is committed immediately (still inside the same
    serialised transaction, before any other write) and the connection is
    returned to autocommit mode, so the caller can raise the 409 without the
    surrounding ``transaction()`` rolling the expiry back.  Returns the
    refreshed row when the session was expired, otherwise ``None``.
    """
    if session["status"] != "active" or session["expires_at"] is None:
        return None
    expires_at = parse_utc(session["expires_at"])
    if now < expires_at:
        return None
    conn.execute(
        """UPDATE sessions
           SET status = 'finished', finish_reason = 'time_expired',
               current_item_id = NULL, updated_at = ?
           WHERE id = ? AND status = 'active'""",
        (now.isoformat(timespec="seconds"), session["id"]),
    )
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session["id"],)
    ).fetchone()
    conn.commit()
    return row


def time_expired_409(expired: sqlite3.Row) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "测验时限已到（UTC 截止时间 "
            f"{expired['expires_at']}），会话已结束："
            "finish_reason=time_expired"
        ),
    )


def get_session_or_404(conn: sqlite3.Connection, session_id: str) -> sqlite3.Row:
    row = conn.execute(
        "SELECT * FROM sessions WHERE id = ?", (session_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="session not found")
    return row


def parse_options(raw: str) -> list[dict]:
    return json.loads(raw)


def serialize_item(row: sqlite3.Row, *, reveal: bool = False) -> ItemOut:
    return ItemOut(
        id=row["id"],
        difficulty=row["difficulty"],
        discrimination=row["discrimination"],
        content_tag=row["content_tag"],
        stem=row["stem"],
        options=parse_options(row["options"]),
    )


def serialize_bank_item(row: sqlite3.Row) -> BankItemOut:
    return BankItemOut(
        id=row["id"],
        difficulty=row["difficulty"],
        discrimination=row["discrimination"],
        content_tag=row["content_tag"],
        stem=row["stem"],
        answer=row["answer"],
        exposure_count=row["exposure_count"],
    )


def answered_count(conn: sqlite3.Connection, session_id: str) -> int:
    return conn.execute(
        "SELECT COUNT(*) FROM responses WHERE session_id = ?", (session_id,)
    ).fetchone()[0]


def tag_minimums_for(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    """Persisted content-tag quotas of a session (empty dict when none set)."""
    return {
        row["content_tag"]: row["minimum"]
        for row in conn.execute(
            "SELECT content_tag, minimum FROM session_tag_minimums "
            "WHERE session_id = ?",
            (session_id,),
        ).fetchall()
    }


def answered_by_tag(conn: sqlite3.Connection, session_id: str) -> dict[str, int]:
    """Number of already-answered items per non-empty content tag."""
    counts: Counter[str] = Counter()
    rows = conn.execute(
        """
        SELECT i.content_tag AS content_tag, COUNT(*) AS n
        FROM responses r JOIN items i ON i.id = r.item_id
        WHERE r.session_id = ? AND i.content_tag IS NOT NULL
          AND i.content_tag <> ''
        GROUP BY i.content_tag
        """,
        (session_id,),
    ).fetchall()
    for row in rows:
        counts[row["content_tag"]] = row["n"]
    return dict(counts)


def tag_shortfalls(
    minimums: dict[str, int], answered: dict[str, int]
) -> dict[str, int]:
    """Remaining quota deficit per tag; tags already met are omitted."""
    return {
        tag: min_qty - answered.get(tag, 0)
        for tag, min_qty in minimums.items()
        if answered.get(tag, 0) < min_qty
    }


def serialize_session(
    row: sqlite3.Row, n_answered: int, tag_minimums: dict[str, int] | None = None
) -> SessionOut:
    return SessionOut(
        id=row["id"],
        theta=row["theta"],
        current_item_id=row["current_item_id"],
        max_items=row["max_items"],
        exposure_limit=row["exposure_limit"],
        duration_seconds=row["duration_seconds"],
        expires_at=row["expires_at"],
        tag_minimums=tag_minimums or {},
        status=row["status"],
        finish_reason=row["finish_reason"],
        answered_count=n_answered,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


def serialize_response(row: sqlite3.Row) -> ResponseOut:
    return ResponseOut(
        id=row["id"],
        session_id=row["session_id"],
        item_id=row["item_id"],
        response=row["response"],
        correct=bool(row["score"]),
        score=row["score"],
        correct_answer=row["correct_answer"],
        theta_before=row["theta_before"],
        theta_after=row["theta_after"],
        theta_change=row["theta_after"] - row["theta_before"],
        p_correct=row["p_correct"],
        information_at_theta=row["information_at_theta"],
        exposure_count=row["exposure_count_at_select"],
        created_at=row["created_at"],
    )


# -------------------------------------------------------------------------
# Health / docs
# -------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthOut, tags=["meta"])
def health(conn: sqlite3.Connection = Depends(get_db)) -> HealthOut:
    bank_size = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    return HealthOut(status="ok", bank_size=bank_size)


# -------------------------------------------------------------------------
# Item bank (read-only inspection)
# -------------------------------------------------------------------------

@app.get("/api/items", response_model=ItemBankOut, tags=["items"])
def list_items(
    limit: int = 100,
    offset: int = 0,
    conn: sqlite3.Connection = Depends(get_db),
) -> ItemBankOut:
    rows = conn.execute(
        "SELECT * FROM items ORDER BY id LIMIT ? OFFSET ?", (limit, offset)
    ).fetchall()
    return ItemBankOut(count=len(rows), items=[serialize_bank_item(r) for r in rows])


@app.get("/api/items/{item_id}", response_model=ItemDetailOut, tags=["items"])
def get_item(
    item_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> ItemDetailOut:
    row = conn.execute(
        "SELECT * FROM items WHERE id = ?", (item_id,)
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="item not found")
    base = serialize_bank_item(row).model_dump()
    base["options"] = parse_options(row["options"])
    return ItemDetailOut(**base)


# -------------------------------------------------------------------------
# Sessions
# -------------------------------------------------------------------------

@app.post(
    "/api/sessions",
    response_model=SessionOut,
    status_code=201,
    tags=["sessions"],
)
def create_session(
    body: SessionCreate, conn: sqlite3.Connection = Depends(get_db)
) -> SessionOut:
    session_id = uuid.uuid4().hex
    created = datetime.now(timezone.utc)
    now = created.isoformat(timespec="seconds")
    duration = body.duration_seconds
    expires_at = (
        (created + timedelta(seconds=duration)).isoformat(timespec="seconds")
        if duration is not None
        else None
    )
    with transaction(conn):
        minimums = body.tag_minimums or {}
        if minimums:
            # Every quota key must reference a non-empty content tag that
            # actually exists in the bank; the quota total must leave room
            # inside max_items (positive ints are already enforced by the
            # request model).  422 mirrors Pydantic-level request errors.
            empty_keys = [k for k in minimums if not k.strip()]
            if empty_keys:
                raise HTTPException(
                    status_code=422,
                    detail="tag_minimums 的键不能为空白内容标签",
                )
            quota_total = sum(minimums.values())
            if quota_total > body.max_items:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"tag_minimums 配额之和（{quota_total}）"
                        f"不能超过 max_items（{body.max_items}）"
                    ),
                )
            placeholders = ",".join("?" for _ in minimums)
            known = {
                r["content_tag"]
                for r in conn.execute(
                    f"SELECT DISTINCT content_tag FROM items "
                    f"WHERE content_tag IS NOT NULL AND content_tag <> '' "
                    f"AND content_tag IN ({placeholders})",
                    tuple(minimums),
                ).fetchall()
            }
            unknown = sorted(set(minimums) - known)
            if unknown:
                raise HTTPException(
                    status_code=422,
                    detail=f"tag_minimums 包含题库不存在的内容标签：{unknown}",
                )

        conn.execute(
            """
            INSERT INTO sessions
                (id, theta, current_item_id, max_items, exposure_limit,
                 duration_seconds, expires_at,
                 status, finish_reason, created_at, updated_at)
            VALUES (?, ?, NULL, ?, ?, ?, ?, 'active', NULL, ?, ?)
            """,
            (
                session_id,
                body.initial_theta,
                body.max_items,
                body.exposure_limit or DEFAULT_EXPOSURE_LIMIT,
                duration,
                expires_at,
                now,
                now,
            ),
        )
        if minimums:
            conn.executemany(
                "INSERT INTO session_tag_minimums (session_id, content_tag, minimum) "
                "VALUES (?, ?, ?)",
                [(session_id, tag, qty) for tag, qty in minimums.items()],
            )
        row = get_session_or_404(conn, session_id)
    return serialize_session(row, 0, minimums)


@app.get("/api/sessions/{session_id}", response_model=SessionOut, tags=["sessions"])
def get_session(
    session_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> SessionOut:
    row = get_session_or_404(conn, session_id)
    return serialize_session(
        row, answered_count(conn, session_id), tag_minimums_for(conn, session_id)
    )


@app.get(
    "/api/sessions/{session_id}/progress",
    response_model=ProgressOut,
    tags=["sessions"],
)
def session_progress(
    session_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> ProgressOut:
    session = get_session_or_404(conn, session_id)
    n_answered = answered_count(conn, session_id)
    answered_ids = [
        r["item_id"]
        for r in conn.execute(
            "SELECT item_id FROM responses WHERE session_id = ? ORDER BY created_at",
            (session_id,),
        ).fetchall()
    ]
    current_item = None
    if session["current_item_id"]:
        item_row = conn.execute(
            "SELECT * FROM items WHERE id = ?", (session["current_item_id"],)
        ).fetchone()
        if item_row is not None:
            current_item = serialize_item(item_row)
    bank_size = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
    minimums = tag_minimums_for(conn, session_id)
    answered_tags = answered_by_tag(conn, session_id)
    # Report zero answered for every quota tag even if it has no response yet.
    tag_answered = {tag: answered_tags.get(tag, 0) for tag in minimums}
    tag_answered.update(answered_tags)
    return ProgressOut(
        session_id=session_id,
        status=session["status"],
        theta=session["theta"],
        answered_count=n_answered,
        max_items=session["max_items"],
        bank_size=bank_size,
        percent_complete=round(n_answered / session["max_items"] * 100, 1),
        exposure_limit=session["exposure_limit"],
        duration_seconds=session["duration_seconds"],
        expires_at=session["expires_at"],
        tag_minimums=minimums,
        tag_answered=dict(sorted(tag_answered.items())),
        finish_reason=session["finish_reason"],
        current_item=current_item,
        answered_item_ids=answered_ids,
    )


@app.get(
    "/api/sessions/{session_id}/responses",
    response_model=ResponseHistoryOut,
    tags=["sessions"],
)
def session_responses(
    session_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> ResponseHistoryOut:
    session = get_session_or_404(conn, session_id)
    rows = conn.execute(
        "SELECT * FROM responses WHERE session_id = ? ORDER BY created_at",
        (session_id,),
    ).fetchall()
    return ResponseHistoryOut(
        session_id=session_id,
        status=session["status"],
        theta=session["theta"],
        count=len(rows),
        results=[serialize_response(r) for r in rows],
    )


# -------------------------------------------------------------------------
# Adaptive item selection
# -------------------------------------------------------------------------

@app.post(
    "/api/sessions/{session_id}/next",
    response_model=NextItemOut,
    tags=["adaptive"],
)
def next_item(
    session_id: str, conn: sqlite3.Connection = Depends(get_db)
) -> NextItemOut:
    """Select the next item.

    Eligibility: ``exposure_count < exposure_limit`` and the item has not
    already been answered in this session.  Selection: highest Fisher
    information at the current theta; ties break on the smallest item id, so
    the result is fully deterministic.

    When the session carries ``tag_minimums``, the per-tag deficit is
    computed from stored responses.  If the number of remaining answer slots
    equals the total deficit, candidates are restricted to deficient tags
    (same information/id tie-break); otherwise all eligible items remain in
    the pool.  If a deficient tag cannot reach its quota with the items still
    eligible, the session finishes with
    ``finish_reason = 'tag_minimums_unfulfillable'`` without serving an item
    (hence without adding an exposure).

    The whole read-check-select-write sequence runs inside one serialised
    write transaction.  That way a burst of concurrent calls on the same
    session is linearised: the first selects and counts one exposure, every
    later call finds the pending item and returns it unchanged.

    If the session has a ``duration_seconds`` limit whose UTC ``expires_at``
    has passed, the session is first atomically finished
    (``finish_reason = 'time_expired'``) with its pending item cleared, and
    the request is rejected with 409; no exposure is added.
    """
    outcome: dict

    with transaction(conn):
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")

        # Case 0: the time limit has elapsed -> finish atomically and reject
        # the call with 409.  No item is selected (no new exposure) and no
        # response row is written; any pending item is cleared.
        expired = expire_if_due(conn, session, datetime.now(timezone.utc))
        if expired is not None:
            raise time_expired_409(expired)

        n_answered = answered_count(conn, session_id)
        now = utc_now()

        # Case 1: a served, unanswered item is pending -> return it as-is.
        if session["current_item_id"]:
            item_row = conn.execute(
                "SELECT * FROM items WHERE id = ?", (session["current_item_id"],)
            ).fetchone()
            outcome = {
                "status": session["status"],
                "item_row": item_row,
                "information": irt.information(
                    session["theta"],
                    item_row["discrimination"],
                    item_row["difficulty"],
                ) if item_row is not None else None,
                "exposure_limit": session["exposure_limit"],
                "answered_count": n_answered,
                "max_items": session["max_items"],
                "finish_reason": session["finish_reason"],
                "note": "pending_item: 该题尚未作答，重复请求不重复计曝光",
            }

        # Case 2: session already finished.
        elif session["status"] == "finished":
            outcome = {
                "status": "finished",
                "item_row": None,
                "information": None,
                "exposure_limit": session["exposure_limit"],
                "answered_count": n_answered,
                "max_items": session["max_items"],
                "finish_reason": session["finish_reason"],
                "note": "测验已结束",
            }

        # Case 3: item quota reached -> finish eagerly.
        elif n_answered >= session["max_items"]:
            conn.execute(
                """UPDATE sessions
                   SET status = 'finished', finish_reason = 'max_items',
                       updated_at = ?
                   WHERE id = ?""",
                (now, session_id),
            )
            outcome = {
                "status": "finished",
                "item_row": None,
                "information": None,
                "exposure_limit": session["exposure_limit"],
                "answered_count": n_answered,
                "max_items": session["max_items"],
                "finish_reason": "max_items",
                "note": "已达到最大题量",
            }

        else:
            # Candidate read under the write lock: exposure counts seen here
            # cannot change before we update them.
            candidates = conn.execute(
                """
                SELECT * FROM items
                WHERE exposure_count < ?
                  AND id NOT IN (
                      SELECT item_id FROM responses WHERE session_id = ?
                  )
                """,
                (session["exposure_limit"], session_id),
            ).fetchall()

            minimums = tag_minimums_for(conn, session_id)
            shortfalls = (
                tag_shortfalls(minimums, answered_by_tag(conn, session_id))
                if minimums
                else {}
            )
            total_shortfall = sum(shortfalls.values())
            slots_left = session["max_items"] - n_answered

            # A shortfall can only ever shrink (a quota-tagged answer covers
            # at most one deficit; a non-quota answer shrinks only the slots),
            # so total_shortfall <= slots_left always holds while selection
            # obeys the restriction below.  When the remaining answer slots
            # equal the deficit exactly, every remaining item must come from
            # a deficient tag.
            if minimums and total_shortfall == slots_left:
                candidates = [
                    c for c in candidates if c["content_tag"] in shortfalls
                ]

            # Case 4: no eligible item left in the (possibly restricted)
            # pool -> finish.  With quotas set this also covers the case in
            # which some deficient tag has no eligible item left while
            # non-quota items are still available: the session cannot be
            # completed either way.
            if not candidates:
                if minimums:
                    finish_reason = "tag_minimums_unfulfillable"
                    note = (
                        "欠额标签已没有满足曝光上限的候选题，无法补齐标签配额："
                        f"缺口 {shortfalls}"
                    )
                else:
                    finish_reason = "no_available_items"
                    note = "没有满足曝光上限的候选题"
                conn.execute(
                    """UPDATE sessions
                       SET status = 'finished',
                           finish_reason = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (finish_reason, now, session_id),
                )
                outcome = {
                    "status": "finished",
                    "item_row": None,
                    "information": None,
                    "exposure_limit": session["exposure_limit"],
                    "answered_count": n_answered,
                    "max_items": session["max_items"],
                    "finish_reason": finish_reason,
                    "note": note,
                }

            else:
                # Even with slots to spare, if any deficient tag cannot
                # possibly reach its quota (not enough eligible items left
                # for that tag), finishing early is the only correct outcome:
                # no item is served and therefore no exposure is added.
                infeasible: dict[str, int] = {}
                if minimums:
                    available_by_tag = Counter(
                        c["content_tag"] for c in candidates
                    )
                    infeasible = {
                        tag: deficit
                        for tag, deficit in shortfalls.items()
                        if available_by_tag.get(tag, 0) < deficit
                    }
                if infeasible:
                    conn.execute(
                        """UPDATE sessions
                           SET status = 'finished',
                               finish_reason = 'tag_minimums_unfulfillable',
                               updated_at = ?
                           WHERE id = ?""",
                        (now, session_id),
                    )
                    outcome = {
                        "status": "finished",
                        "item_row": None,
                        "information": None,
                        "exposure_limit": session["exposure_limit"],
                        "answered_count": n_answered,
                        "max_items": session["max_items"],
                        "finish_reason": "tag_minimums_unfulfillable",
                        "note": (
                            "欠额标签的可用题数不足以补齐配额，"
                            f"无法满足的缺口 {infeasible}"
                        ),
                    }

                # Case 5: select the maximum-information item (id tie-break),
                # count exactly one exposure and mark it pending atomically.
                else:
                    theta = session["theta"]
                    best = min(
                        candidates,
                        key=lambda r: (
                            -irt.information(
                                theta, r["discrimination"], r["difficulty"]
                            ),
                            r["id"],
                        ),
                    )
                    info = irt.information(
                        theta, best["discrimination"], best["difficulty"]
                    )
                    conn.execute(
                        "UPDATE items SET exposure_count = exposure_count + 1 "
                        "WHERE id = ?",
                        (best["id"],),
                    )
                    conn.execute(
                        "UPDATE sessions SET current_item_id = ?, updated_at = ? "
                        "WHERE id = ?",
                        (best["id"], now, session_id),
                    )
                    item_row = conn.execute(
                        "SELECT * FROM items WHERE id = ?", (best["id"],)
                    ).fetchone()
                    outcome = {
                        "status": "active",
                        "item_row": item_row,
                        "information": info,
                        "exposure_limit": session["exposure_limit"],
                        "answered_count": n_answered,
                        "max_items": session["max_items"],
                        "finish_reason": None,
                        "note": None,
                    }

    item_row = outcome["item_row"]
    return NextItemOut(
        session_id=session_id,
        status=outcome["status"],
        item=serialize_item(item_row) if item_row is not None else None,
        information=outcome["information"],
        exposure_count=item_row["exposure_count"] if item_row is not None else None,
        exposure_limit=outcome["exposure_limit"],
        answered_count=outcome["answered_count"],
        max_items=outcome["max_items"],
        finish_reason=outcome["finish_reason"],
        note=outcome["note"],
    )


# -------------------------------------------------------------------------
# Answer submission (idempotent, one Newton update)
# -------------------------------------------------------------------------

@app.post(
    "/api/sessions/{session_id}/responses",
    response_model=ResponseOut,
    status_code=201,
    tags=["adaptive"],
)
def submit_response(
    session_id: str,
    body: ResponseCreate,
    conn: sqlite3.Connection = Depends(get_db),
) -> ResponseOut | JSONResponse:
    """Score one answer and apply exactly one Newton update.

    Idempotency contract: the client-generated ``response_id`` is the
    dedupe key.  The deadline check is the *first* statement inside the
    serialised write transaction: once ``expires_at`` has passed the session
    is atomically finished (``time_expired``) with its pending item cleared,
    and the call is rejected with 409 — no exposure is added and no response
    row is written.  Afterwards the idempotency lookup is checked under the
    same write lock, so even a burst of concurrent retries observes the
    first committed record and is replayed (HTTP 200) instead of being
    scored a second time.
    """
    response_key = body.response.strip().upper()
    stored_row: sqlite3.Row | None = None
    replay = False

    with transaction(conn):
        # 1) Deadline check, first inside the serialised write transaction:
        #    expiry finishes the session atomically and rejects the call;
        #    nothing else (exposure, response row) is written.
        session = conn.execute(
            "SELECT * FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if session is None:
            raise HTTPException(status_code=404, detail="session not found")
        expired = expire_if_due(conn, session, datetime.now(timezone.utc))
        if expired is not None:
            raise time_expired_409(expired)

        # 2) Idempotency replay, checked after acquiring the write lock.
        existing = conn.execute(
            "SELECT * FROM responses WHERE id = ?", (body.response_id,)
        ).fetchone()
        if existing is not None:
            if existing["session_id"] != session_id or existing["item_id"] != body.item_id:
                raise HTTPException(
                    status_code=409,
                    detail="response_id 已被其他会话或题目使用",
                )
            stored_row, replay = existing, True
        else:
            # 3) Validate the pending item state (existence, deadline and
            #    finished status have already been checked above).
            if session["status"] == "finished":
                raise HTTPException(status_code=409, detail="测验已结束")
            if session["current_item_id"] != body.item_id:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "item_id 与当前待作答题不一致或该题已作答："
                        "请先调用 "
                        f"POST /api/sessions/{session_id}/next 获取题目；"
                        "如为重试请求，请复用原 response_id"
                    ),
                )

            item = conn.execute(
                "SELECT * FROM items WHERE id = ?", (body.item_id,)
            ).fetchone()
            if item is None:
                raise HTTPException(status_code=404, detail="item not found")

            option_keys = {opt["key"] for opt in parse_options(item["options"])}
            if response_key not in option_keys:
                raise HTTPException(
                    status_code=422,
                    detail=f"无效作答选项：{body.response!r}，可选 {sorted(option_keys)}",
                )

            # 4) Score and apply exactly one Newton-Raphson update.
            score = 1 if response_key == item["answer"] else 0
            theta_before = session["theta"]
            theta_after, p_correct, _, info = irt.newton_step(
                theta_before, item["discrimination"], item["difficulty"], score
            )
            now = utc_now()
            exposure_at_select = item["exposure_count"]

            try:
                conn.execute(
                    """
                    INSERT INTO responses (
                        id, session_id, item_id, response, score, correct_answer,
                        theta_before, theta_after, p_correct,
                        information_at_theta, exposure_count_at_select, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        body.response_id,
                        session_id,
                        body.item_id,
                        response_key,
                        score,
                        item["answer"],
                        theta_before,
                        theta_after,
                        p_correct,
                        info,
                        exposure_at_select,
                        now,
                    ),
                )
            except sqlite3.IntegrityError:
                # Database-level backstop for a concurrent key collision.
                conn.rollback()
                racer = conn.execute(
                    "SELECT * FROM responses WHERE id = ?", (body.response_id,)
                ).fetchone()
                if racer is not None:
                    if racer["session_id"] != session_id or racer["item_id"] != body.item_id:
                        raise HTTPException(
                            status_code=409,
                            detail="response_id 已被其他会话或题目使用",
                        )
                    # Re-read outside the rolled-back transaction below.
                    stored_row, replay = racer, True
                else:
                    raise HTTPException(status_code=409, detail="重复作答冲突")
            else:
                # 5) Advance session state; finish eagerly at max_items.
                conn.execute(
                    """UPDATE sessions
                       SET theta = ?, current_item_id = NULL, updated_at = ?
                       WHERE id = ?""",
                    (theta_after, now, session_id),
                )
                total = conn.execute(
                    "SELECT COUNT(*) FROM responses WHERE session_id = ?",
                    (session_id,),
                ).fetchone()[0]
                if total >= session["max_items"]:
                    conn.execute(
                        """UPDATE sessions
                           SET status = 'finished', finish_reason = 'max_items',
                               updated_at = ?
                           WHERE id = ?""",
                        (now, session_id),
                    )
                stored_row = conn.execute(
                    "SELECT * FROM responses WHERE id = ?", (body.response_id,)
                ).fetchone()

    payload = serialize_response(stored_row).model_dump()
    return JSONResponse(status_code=200 if replay else 201, content=payload)
