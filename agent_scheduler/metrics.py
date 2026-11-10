"""Latency percentiles, Jain's fairness index and max-min fair shares."""

from __future__ import annotations

import math


def percentile(xs: list[float], p: float) -> float:
    """Linear-interpolated percentile, p in [0, 100]. NaN for empty input."""
    if not xs:
        return math.nan
    s = sorted(xs)
    k = (len(s) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def jain(xs: list[float]) -> float:
    """Jain's index (sum x)^2 / (n * sum x^2). 1.0 = perfectly even, 1/n = one gets all."""
    if not xs:
        return math.nan
    sq = sum(x * x for x in xs)
    if sq == 0:
        return 1.0
    return sum(xs) ** 2 / (len(xs) * sq)


def max_min_shares(demands: dict[str, float], capacity: float) -> dict[str, float]:
    """Water-filling max-min fair allocation of `capacity` over `demands`."""
    share = {k: 0.0 for k in demands}
    left = dict(demands)
    cap = capacity
    while left and cap > 1e-12:
        level = cap / len(left)
        sat = {k: d for k, d in left.items() if d <= level}
        if not sat:
            for k in left:
                share[k] += level
            return share
        for k, d in sat.items():
            share[k] += d
            cap -= d
            del left[k]
    return share


def normalized_jain(goodput: dict[str, float], demand: dict[str, float]) -> float:
    """Jain over goodput_i / maxmin_share_i, capacity = total achieved goodput.
    1.0 means the achieved throughput was split max-min fairly given each tenant's demand.
    """
    cap = sum(goodput.values())
    shares = max_min_shares(demand, cap)
    xs = [goodput[k] / shares[k] if shares[k] > 0 else 1.0 for k in demand]
    return jain(xs)
