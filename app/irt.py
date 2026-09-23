"""IRT math for the adaptive engine.

The item bank uses the two-parameter logistic model (2PL).

For an examinee with ability ``theta`` and an item with discrimination ``a``
and difficulty ``b``, the probability of a correct response is::

    P(theta) = 1 / (1 + exp(-a * (theta - b)))

The Fisher information of that item at ``theta`` is::

    I(theta) = a**2 * P(theta) * (1 - P(theta))

Ability is estimated by *one* Newton-Raphson step on the log-likelihood of a
single binary response ``u`` (1 = correct, 0 = wrong).  For one observation::

    d1/dtheta = a * (u - P)
    d2/dtheta = -a**2 * P * (1 - P) = -I(theta)

so the Newton update is::

    theta' = theta - d1/d2 = theta + a * (u - P) / I(theta)
"""

from __future__ import annotations

import math

THETA_MIN = -6.0
THETA_MAX = 6.0

# Information is tiny when prediction is nearly certain; floor the Newton
# denominator so a single extreme response cannot send theta to infinity.
_DENOM_FLOOR = 1e-4


def clamp_theta(theta: float) -> float:
    """Keep ability estimates inside a numerically sane range."""
    return min(max(float(theta), THETA_MIN), THETA_MAX)


def probability(theta: float, a: float, b: float) -> float:
    """2PL probability of a correct answer (logistic, overflow-safe)."""
    z = -a * (theta - b)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(min(z, 60.0)))
    exp_z = math.exp(max(z, -60.0))
    return exp_z / (1.0 + exp_z)


def information(theta: float, a: float, b: float) -> float:
    """Fisher information of item (a, b) at ability theta."""
    p = probability(theta, a, b)
    return (a * a) * p * (1.0 - p)


def newton_step(
    theta: float, a: float, b: float, score: int
) -> tuple[float, float, float, float]:
    """Apply one Newton-Raphson update for one binary response.

    Returns ``(new_theta, p_correct, gradient, information)``.
    """
    p = probability(theta, a, b)
    gradient = a * (score - p)
    info = (a * a) * p * (1.0 - p)
    step = gradient / max(info, _DENOM_FLOOR)
    return clamp_theta(theta + step), p, gradient, info
