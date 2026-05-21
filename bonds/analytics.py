"""
Statistical analysis over the parsed PDS Corporate Board Summary.

Pure-stdlib (statistics) number crunching so the Bond Analyst skill can get
real correlation / variance / trend figures instead of guessing. Operates on
the same in-process board cache the analyze-bond pipeline uses.
"""

import statistics
from collections import defaultdict

from .sources._common import normalize_id
from .sources.pds_corporate import _alias_prefixes, _load_board, _num


def _summary(values: list[float]) -> dict | None:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    s = {
        "n": len(vals),
        "mean": round(statistics.fmean(vals), 4),
        "median": round(statistics.median(vals), 4),
        "min": round(min(vals), 4),
        "max": round(max(vals), 4),
    }
    if len(vals) >= 2:
        s["stdev"] = round(statistics.stdev(vals), 4)
        s["variance"] = round(statistics.variance(vals), 4)
    return s


def _pairs(rows: list[dict], a: str, b: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for r in rows:
        if r[a] is not None and r[b] is not None:
            xs.append(r[a])
            ys.append(r[b])
    return xs, ys


def _corr(rows: list[dict], a: str, b: str) -> dict | None:
    xs, ys = _pairs(rows, a, b)
    if len(xs) < 3:
        return None
    try:
        return {"pearson_r": round(statistics.correlation(xs, ys), 4), "n": len(xs)}
    except statistics.StatisticsError:
        return None


def _trend(rows: list[dict], x: str, y: str) -> dict | None:
    xs, ys = _pairs(rows, x, y)
    if len(xs) < 3:
        return None
    try:
        slope, intercept = statistics.linear_regression(xs, ys)
        return {"slope": round(slope, 4), "intercept": round(intercept, 4), "n": len(xs)}
    except statistics.StatisticsError:
        return None


async def compute_stats(issuer: str | None = None) -> dict:
    """Return summary stats, correlations, trend, and by-issuer aggregates over the
    corporate board (optionally filtered to one issuer/keyword)."""
    board = await _load_board()
    q = normalize_id(issuer) if issuer else None
    prefixes = _alias_prefixes(q) if q else set()

    rows: list[dict] = []
    for r in board["rows"]:
        p, y = r["price"], r["yield"]
        local = p.get("Local ID", "")
        if not local:
            continue
        if q:
            hay = normalize_id(" ".join([p.get("Local ID", ""), p.get("Global ID", ""), p.get("ISIN", "")]))
            if q not in hay and not any(pref in hay for pref in prefixes):
                continue
        rows.append({
            "issuer": (p.get("Global ID", "").split() or [""])[0],
            "coupon": _num(p.get("CPN")),
            "tenor_years": _num(p.get("YRS")),
            "last_price": _num(p.get("Last Price")),
            "last_yield": _num(y.get("Last Yield")),
            "bid_yield": _num(y.get("Bid Yield")),
        })

    if not rows:
        return {"as_of": board["date"], "issuer": issuer, "count": 0, "note": "no matching securities"}

    yields = [r["last_yield"] for r in rows]
    by_issuer = defaultdict(list)
    for r in rows:
        by_issuer[r["issuer"]].append(r)

    issuer_table = []
    for iss, group in sorted(by_issuer.items(), key=lambda kv: -len(kv[1])):
        coupons = [g["coupon"] for g in group if g["coupon"] is not None]
        gy = [g["last_yield"] for g in group if g["last_yield"] is not None]
        issuer_table.append({
            "issuer": iss,
            "count": len(group),
            "mean_coupon": round(statistics.fmean(coupons), 4) if coupons else None,
            "mean_last_yield": round(statistics.fmean(gy), 4) if gy else None,
        })

    return {
        "as_of": board["date"],
        "issuer": issuer,
        "count": len(rows),
        "count_with_yield": len([v for v in yields if v is not None]),
        "summary": {
            "coupon": _summary([r["coupon"] for r in rows]),
            "tenor_years": _summary([r["tenor_years"] for r in rows]),
            "last_yield": _summary(yields),
            "last_price": _summary([r["last_price"] for r in rows]),
        },
        "correlations": {
            "coupon_vs_tenor": _corr(rows, "coupon", "tenor_years"),
            "coupon_vs_last_yield": _corr(rows, "coupon", "last_yield"),
            "tenor_vs_last_yield": _corr(rows, "tenor_years", "last_yield"),
        },
        "trend": {
            # term structure: last_yield as a function of tenor (slope = yield change per +1yr)
            "last_yield_vs_tenor": _trend(rows, "tenor_years", "last_yield"),
            "coupon_vs_tenor": _trend(rows, "tenor_years", "coupon"),
        },
        "by_issuer": issuer_table[:25],
    }
