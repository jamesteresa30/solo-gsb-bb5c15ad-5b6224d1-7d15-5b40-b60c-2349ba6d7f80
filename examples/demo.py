"""End-to-end demo using only the Python standard library.

Run with the API already listening on http://127.0.0.1:8000:

    python examples/demo.py

It creates a time-limited session, runs a mini adaptive quiz, demonstrates
idempotent retry with a repeated response_id, shows /next de-duplication,
prints session progress and the answer history, and demonstrates the
duration_seconds validation (60..86400).  Finally it demonstrates content-tag
minimum quotas (``tag_minimums``), including the
``tag_minimums_unfulfillable`` finish reason when a quota cannot be met.
"""

from __future__ import annotations

import json
import math
import random
import urllib.error
import urllib.request

BASE_URL = "http://127.0.0.1:8000"


def call(method: str, path: str, payload: dict | None = None) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        BASE_URL + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def expect_http_error(method: str, path: str, payload: dict) -> None:
    """Show that a request is rejected (used for duration validation)."""
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        BASE_URL + path, data=data,
        headers={"Content-Type": "application/json"}, method=method,
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except urllib.error.HTTPError as exc:
        body = json.loads(exc.read().decode())
        print(f"  -> 被拒绝 HTTP {exc.code}: {json.dumps(body['detail'], ensure_ascii=False)}")
    else:
        raise AssertionError("应当被服务端拒绝")


def show(label: str, obj: object) -> None:
    print(f"\n=== {label} ===")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def main() -> None:
    health = call("GET", "/health")
    show("GET /health", health)

    # duration_seconds 必须在 60..86400 之间；越界由 Pydantic 以 422 拒绝。
    expect_http_error(
        "POST", "/api/sessions",
        {"initial_theta": 0.0, "max_items": 8, "duration_seconds": 30},
    )

    # 本场测验限时 600 秒；响应中 duration_seconds 与 UTC expires_at 均会返回。
    session = call(
        "POST", "/api/sessions",
        {"initial_theta": 0.0, "max_items": 8, "exposure_limit": 5,
         "duration_seconds": 600},
    )
    sid = session["id"]
    show("POST /api/sessions", session)
    assert session["duration_seconds"] == 600
    assert session["expires_at"], "设置时限后应返回 UTC 截止时间"
    print(f"  -> 时限 600 秒，UTC 截止时间 expires_at = {session['expires_at']}")
    print("  -> 到期后 /next 与提交答案会返回 409，会话置为 finished(time_expired)")

    rng = random.Random(7)
    theta = session["theta"]

    for turn in range(1, 6):
        chosen = call("POST", f"/api/sessions/{sid}/next")
        item = chosen["item"]
        show(
            f"第 {turn} 题 (POST .../next)",
            {
                "item_id": item["id"],
                "stem": item["stem"],
                "information": round(chosen["information"], 4),
                "exposure_count": chosen["exposure_count"],
            },
        )

        # Re-calling /next before answering must return the SAME item and
        # must not increment exposure again.
        again = call("POST", f"/api/sessions/{sid}/next")
        assert again["item"]["id"] == item["id"], "重复 /next 应返回同一题"
        assert again["exposure_count"] == chosen["exposure_count"], \
            "重复 /next 不应重复计曝光"

        # Simulated examinee using the same 2PL model: answer correctly with
        # probability P(theta), otherwise pick a random distractor key.
        z = item["discrimination"] * (theta - item["difficulty"])
        p = 1 / (1 + math.exp(-z))
        keys = [opt["key"] for opt in item["options"]]
        # The served item hides the answer, but the demo needs to "be right"
        # sometimes: read it from the admin endpoint, else pick a distractor.
        detail = call("GET", f"/api/items/{item['id']}")
        if rng.random() < p:
            answer = detail["answer"]
        else:
            answer = rng.choice([k for k in keys if k != detail["answer"]])
        response_id = f"demo-resp-{turn}-sid-{sid[:8]}"

        result = call(
            "POST", f"/api/sessions/{sid}/responses",
            {"response_id": response_id, "item_id": item["id"], "response": answer},
        )
        theta = result["theta_after"]
        show(
            f"提交作答 (POST .../responses)",
            {
                "response_id": result["id"],
                "item_id": result["item_id"],
                "response": result["response"],
                "correct": result["correct"],
                "theta_before": round(result["theta_before"], 4),
                "theta_after": round(result["theta_after"], 4),
            },
        )

        # Idempotency: replay the exact same request body.
        replay = call(
            "POST", f"/api/sessions/{sid}/responses",
            {"response_id": response_id, "item_id": item["id"], "response": answer},
        )
        assert replay["created_at"] == result["created_at"], "应返回首次记录"
        assert replay["theta_after"] == result["theta_after"], "不应重复更新"
        print(f"  -> 重复 response_id 返回首次结果，能力值未二次更新 ✔")

    progress = call("GET", f"/api/sessions/{sid}/progress")
    show("GET .../progress (会话进度)", {
        "status": progress["status"],
        "theta": round(progress["theta"], 4),
        "answered_count": progress["answered_count"],
        "percent_complete": progress["percent_complete"],
        "duration_seconds": progress["duration_seconds"],
        "expires_at": progress["expires_at"],
        "answered_item_ids": progress["answered_item_ids"],
        "current_item": (progress["current_item"] or {}).get("id"),
    })

    history = call("GET", f"/api/sessions/{sid}/responses")
    show("GET .../responses (答题记录)", {
        "count": history["count"],
        "results": [
            {
                "item_id": r["item_id"],
                "correct": r["correct"],
                "theta_after": round(r["theta_after"], 4),
            }
            for r in history["results"]
        ],
    })

    bank = call("GET", "/api/items?limit=3")
    show("GET /api/items?limit=3 (题库与曝光)", bank)

    demo_tag_minimums()


def demo_tag_minimums() -> None:
    """Show tag_minimums: restricted endgame and the unfulfillable finish."""
    # 校验：配额之和不能超过 max_items -> 422
    expect_http_error(
        "POST", "/api/sessions",
        {"max_items": 3, "tag_minimums": {"算术": 2, "代数": 2}},
    )
    # 校验：键必须是题库现有非空 content_tag -> 422
    expect_http_error(
        "POST", "/api/sessions",
        {"max_items": 3, "tag_minimums": {"不存在的标签": 1}},
    )

    # 6 道题全部要求来自「数论」：剩余答题位始终等于缺口总数，
    # 因此每题都必须在欠额标签内按信息量/ID 规则选出。
    session = call(
        "POST", "/api/sessions",
        {"max_items": 6, "exposure_limit": 50, "tag_minimums": {"数论": 6}},
    )
    sid = session["id"]
    assert session["tag_minimums"] == {"数论": 6}
    tags: list[str] = []
    for turn in range(1, 7):
        chosen = call("POST", f"/api/sessions/{sid}/next")
        item = chosen["item"]
        tags.append(item["content_tag"])
        detail = call("GET", f"/api/items/{item['id']}")
        call(
            "POST", f"/api/sessions/{sid}/responses",
            {"response_id": f"demo-tag-{sid[:8]}-{turn}",
             "item_id": item["id"], "response": detail["answer"]},
        )
    progress = call("GET", f"/api/sessions/{sid}/progress")
    show("tag_minimums 配额会话（数论 x6）", {
        "served_tags": tags,
        "status": progress["status"],
        "finish_reason": progress["finish_reason"],
        "tag_minimums": progress["tag_minimums"],
        "tag_answered": progress["tag_answered"],
    })
    assert all(t == "数论" for t in tags), tags
    assert progress["finish_reason"] == "max_items"

    # 不可补齐：要求 1 道「概率统计」题（exposure_limit=1）。随后用若干
    # 一次性配额会话把该标签题全部曝光到上限；若本库此前已跑过演示，第一
    # 个消耗会话就会直接结束，循环同样终止（重复运行结果稳定）。
    quota_session = call(
        "POST", "/api/sessions",
        {"max_items": 3, "exposure_limit": 1,
         "tag_minimums": {"概率统计": 1}},
    )
    while True:
        burner = call(
            "POST", "/api/sessions",
            {"max_items": 1, "exposure_limit": 1,
             "tag_minimums": {"概率统计": 1}},
        )
        picked = call("POST", f"/api/sessions/{burner['id']}/next")
        if picked["item"] is None:
            assert picked["finish_reason"] == "tag_minimums_unfulfillable"
            break
        assert picked["item"]["content_tag"] == "概率统计"
    done = call("POST", f"/api/sessions/{quota_session['id']}/next")
    show("欠额标签已无可用题 -> 会话结束", {
        "status": done["status"],
        "finish_reason": done["finish_reason"],
        "item": done["item"],
        "note": done["note"],
    })
    assert done["status"] == "finished"
    assert done["finish_reason"] == "tag_minimums_unfulfillable"
    assert done["item"] is None, "结束时不额外选题/曝光"
    print("  -> 欠额标签题数不足时以 tag_minimums_unfulfillable 结束，未增加曝光 ✔")


if __name__ == "__main__":
    try:
        main()
    except urllib.error.URLError as exc:
        raise SystemExit(
            f"无法连接 {BASE_URL}，请先启动服务（uvicorn app.main:app 或 docker compose up）。\n{exc}"
        )
