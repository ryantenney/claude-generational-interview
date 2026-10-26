"""Small dependency-free statistics helpers."""
from __future__ import annotations

import math
from typing import Sequence

# two-sided 95% Student t critical values by degrees of freedom
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
         11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
         20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
         29: 2.045, 30: 2.042}


def clean(xs: Sequence[float | None]) -> list[float]:
    return [float(x) for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]


def mean(xs: Sequence[float | None]) -> float | None:
    v = clean(xs)
    return sum(v) / len(v) if v else None


def sd(xs: Sequence[float | None]) -> float | None:
    v = clean(xs)
    if len(v) < 2:
        return None
    m = sum(v) / len(v)
    return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))


def ci95(xs: Sequence[float | None]) -> tuple[float, float, float] | None:
    """(mean, low, high) using a t interval; None with fewer than 2 values."""
    v = clean(xs)
    if len(v) < 2:
        return None
    m, s = sum(v) / len(v), sd(v)
    half = _T975.get(len(v) - 1, 1.96) * s / math.sqrt(len(v))
    return m, m - half, m + half


def slope(ys: Sequence[float | None], xs: Sequence[float] | None = None) -> float | None:
    """Ordinary least squares slope of y on x (x defaults to 0..n-1), skipping missing y."""
    pts = [(float(x), float(y)) for x, y in zip(xs if xs is not None else range(len(ys)), ys) if y is not None]
    if len(pts) < 3:
        return None
    mx, my = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
    den = sum((x - mx) ** 2 for x, _ in pts)
    return sum((x - mx) * (y - my) for x, y in pts) / den if den else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0 or sy == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (sx * sy)


def paired_diff(a: Sequence[float | None], b: Sequence[float | None]) -> dict | None:
    """Paired comparison of a vs b across runs: mean diff, 95% CI and Cohen's dz."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if len(pairs) < 2:
        return None
    d = [x - y for x, y in pairs]
    ci = ci95(d)
    s = sd(d)
    return {"n": len(d), "mean_diff": ci[0], "low": ci[1], "high": ci[2],
            "dz": (ci[0] / s) if s else None, "excludes_zero": ci[1] > 0 or ci[2] < 0}
