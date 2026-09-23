"""Deterministic demo item bank.

30 four-option arithmetic items whose difficulty grows linearly from -3 to +3
and whose discrimination varies pseudo-randomly.  A fixed RNG seed means every
fresh database gets exactly the same bank (stable item ids ITEM-001..030).
"""

from __future__ import annotations

import json
import random

N_ITEMS = 30
TAGS = ["算术", "代数", "几何", "概率统计", "数论"]
_OPTION_KEYS = ["A", "B", "C", "D"]


def _make_stem(difficulty: float, rng: random.Random) -> tuple[str, int]:
    if difficulty < -1.5:
        op = rng.choice("+-")
        x, y = rng.randint(2, 20), rng.randint(2, 20)
    elif difficulty < 1.5:
        op = rng.choice("+-*")
        if op == "*":
            x, y = rng.randint(3, 9), rng.randint(3, 9)
        else:
            x, y = rng.randint(10, 60), rng.randint(5, 40)
    else:
        op = rng.choice("+-*")
        if op == "*":
            x, y = rng.randint(8, 15), rng.randint(4, 12)
        else:
            x, y = rng.randint(25, 90), rng.randint(10, 60)

    if op == "+":
        return f"计算：{x} + {y} = ?", x + y
    if op == "-":
        return f"计算：{x} − {y} = ?", x - y
    return f"计算：{x} × {y} = ?", x * y


def _make_options(answer: int, rng: random.Random) -> tuple[list[dict], str]:
    """Build 4 distinct integer options and return the correct option key."""
    values: set[int] = {answer}
    deltas = [1, 2, 3, 4, 5, 6, 7, 8, 10, 20]
    rng.shuffle(deltas)
    for delta in deltas:
        if len(values) == 4:
            break
        for candidate in (answer + delta, answer - delta, answer + 2 * delta):
            if candidate not in values:
                values.add(candidate)
                break

    ordered = list(values)
    rng.shuffle(ordered)
    options = [
        {"key": key, "text": str(value)}
        for key, value in zip(_OPTION_KEYS, ordered)
    ]
    correct_key = _OPTION_KEYS[ordered.index(answer)]
    return options, correct_key


def seed_items() -> list[dict]:
    rng = random.Random(20260923)
    rows: list[dict] = []
    for i in range(1, N_ITEMS + 1):
        difficulty = round(-3.0 + (i - 1) * (6.0 / (N_ITEMS - 1)), 2)
        discrimination = round(0.8 + ((i * 7) % 11) / 10.0, 2)
        stem, answer_value = _make_stem(difficulty, rng)
        options, correct_key = _make_options(answer_value, rng)
        rows.append(
            {
                "id": f"ITEM-{i:03d}",
                "difficulty": difficulty,
                "discrimination": discrimination,
                "content_tag": TAGS[(i - 1) % len(TAGS)],
                "stem": stem,
                "options": json.dumps(options, ensure_ascii=False),
                "answer": correct_key,
            }
        )
    return rows
