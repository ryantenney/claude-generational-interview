"""Minimal dependency-free SVG line charts. Axes and text use currentColor so they follow the page theme."""
from __future__ import annotations

import math
from html import escape

# Okabe-Ito: distinguishable under common colour-vision deficiencies
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    span = (hi - lo) or 1.0
    raw = span / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if m * mag >= raw)
    ticks, v = [], math.ceil(lo / step - 1e-9) * step
    while v <= hi + 1e-9 and len(ticks) < 20:
        ticks.append(round(v, 10))
        v += step
    return ticks


def line_chart(title: str, series: list[tuple[str, list[tuple[float, float]]]], *, xlabel: str, ylabel: str,
               markers: list[float] | None = None, marker_label: str = "moderator note",
               ymin: float | None = None, ymax: float | None = None, desc: str = "") -> str:
    W, H = 680, 290
    L, R, T, B = 52, 16, 52, 44
    pts = [p for _, s in series for p in s]
    if not pts:
        return ""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0 = ymin if ymin is not None else min(0.0, min(ys))
    y1 = ymax if ymax is not None else max(ys) * 1.08 or 1.0
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    def sx(x): return L + (x - x0) / (x1 - x0) * (W - L - R)
    def sy(y): return H - B - (y - y0) / (y1 - y0) * (H - T - B)

    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" role="img" '
         f'aria-labelledby="t d" style="width:100%;height:auto;max-width:{W}px;font-family:system-ui,sans-serif;font-size:11px">',
         f'<title id="t">{escape(title)}</title><desc id="d">{escape(desc or title)}</desc>',
         f'<text x="{L}" y="17" font-size="13" font-weight="600" fill="currentColor">{escape(title)}</text>']
    for t in _nice_ticks(y0, y1):
        y = sy(t)
        o.append(f'<line x1="{L}" x2="{W - R}" y1="{y:.1f}" y2="{y:.1f}" stroke="currentColor" stroke-opacity="0.12"/>')
        o.append(f'<text x="{L - 6}" y="{y + 3.5:.1f}" text-anchor="end" fill="currentColor" fill-opacity="0.7">{t:g}</text>')
    for t in _nice_ticks(x0, x1, 8):
        if x0 <= t <= x1:
            o.append(f'<text x="{sx(t):.1f}" y="{H - B + 15}" text-anchor="middle" fill="currentColor" fill-opacity="0.7">{t:g}</text>')
    o.append(f'<line x1="{L}" x2="{W - R}" y1="{H - B}" y2="{H - B}" stroke="currentColor" stroke-opacity="0.4"/>')
    o.append(f'<line x1="{L}" x2="{L}" y1="{T}" y2="{H - B}" stroke="currentColor" stroke-opacity="0.4"/>')
    o.append(f'<text x="{(L + W - R) / 2}" y="{H - 6}" text-anchor="middle" fill="currentColor" fill-opacity="0.8">{escape(xlabel)}</text>')
    o.append(f'<text transform="translate(13 {(T + H - B) / 2}) rotate(-90)" text-anchor="middle" fill="currentColor" fill-opacity="0.8">{escape(ylabel)}</text>')
    for m in markers or []:
        if x0 <= m <= x1:
            o.append(f'<line x1="{sx(m):.1f}" x2="{sx(m):.1f}" y1="{T}" y2="{H - B}" stroke="currentColor" stroke-opacity="0.45" stroke-dasharray="4 3"/>')
    for i, (name, s) in enumerate(series):
        if not s:
            continue
        c = PALETTE[i % len(PALETTE)]
        d = " ".join(f"{'M' if j == 0 else 'L'}{sx(x):.1f},{sy(y):.1f}" for j, (x, y) in enumerate(sorted(s)))
        o.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2" stroke-linejoin="round"/>')
        for x, y in s:
            o.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{c}"/>')
    lx = L
    for i, (name, _) in enumerate(series):
        c = PALETTE[i % len(PALETTE)]
        o.append(f'<rect x="{lx}" y="33" width="10" height="3" fill="{c}"/>')
        o.append(f'<text x="{lx + 14}" y="37" fill="currentColor">{escape(name)}</text>')
        lx += 22 + 6.2 * len(name)
    if markers:
        o.append(f'<text x="{lx}" y="37" fill="currentColor" fill-opacity="0.7">┆ {escape(marker_label)}</text>')
    o.append("</svg>")
    return "\n".join(o)
