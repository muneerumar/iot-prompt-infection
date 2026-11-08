"""Statistics: Wilson intervals, context-clustered bootstrap, exact McNemar for paired conditions.

The unit of analysis is the episode. Episodes are nested in contexts (buildings), so intervals for
rates resample contexts, not episodes. Conditions are compared on the SAME episodes (paired).
"""
from __future__ import annotations

import math
import random
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.959964):
    if n == 0:
        return None, None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def mean_ci(xs, z: float = 1.959964):
    n = len(xs)
    if n == 0:
        return None, None, None
    m = sum(xs) / n
    if n == 1:
        return m, m, m
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = z * sd / math.sqrt(n)
    return m, m - h, m + h


def cluster_bootstrap(values: list[float], clusters: list[str], reps: int = 2000, seed: int = 0):
    """Mean with a 95% percentile interval, resampling clusters with replacement."""
    if len(values) != len(clusters):
        raise ValueError("values and clusters must have equal length")
    if not values:
        return None, None, None
    groups = defaultdict(list)
    for v, c in zip(values, clusters):
        groups[c].append(v)
    keys = sorted(groups)
    rng = random.Random(seed)
    est = sum(values) / len(values)
    if len(keys) < 2:
        return est, None, None
    boots = []
    for _ in range(reps):
        pick = [rng.choice(keys) for _ in keys]
        vals = [v for k in pick for v in groups[k]]
        boots.append(sum(vals) / len(vals))
    boots.sort()
    return est, boots[int(0.025 * reps)], boots[int(0.975 * reps) - 1]


def mcnemar_exact(a: list[bool], b: list[bool]) -> dict:
    """Paired binary outcomes (e.g. harm under condition A vs B on the same episodes)."""
    if len(a) != len(b):
        raise ValueError("paired outcomes must have equal length")
    n01 = sum(1 for x, y in zip(a, b) if not x and y)
    n10 = sum(1 for x, y in zip(a, b) if x and not y)
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"n01": n01, "n10": n10, "p": p}


def holm(pvals: dict) -> dict:
    """Holm-Bonferroni adjusted p-values for a family of comparisons."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
