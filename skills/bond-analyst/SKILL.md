---
name: bond-analyst
description: Use when the user asks for STATISTICAL or quantitative analysis of bonds — correlation, variance, standard deviation, dispersion, trend, the yield curve / term structure, or comparing issuers and tenors. Distinct from simple single-bond lookups.
---

# Bond Analyst workflow

For any statistical/quantitative bond question (correlation, variance, std dev, trend, yield curve, spread, dispersion, cross-issuer comparison):

1. Call the `bond_stats` tool. Pass `issuer` to scope to one issuer/keyword (e.g. "Ayala"), or leave it empty for the whole corporate market.
2. NEVER compute statistics from memory or estimate them. Always use the numbers `bond_stats` returns. If a field is `null` (e.g. too few traded yields), say so — do not fabricate.
3. Present results as MARKDOWN TABLES:
   - **Summary** table: metric (coupon, tenor, last yield, last price) × mean / median / stdev / variance / min / max.
   - **Correlations** table: pair × Pearson r × n. Note that r near +1/−1 is strong, near 0 is weak.
   - **Trend / yield curve**: report the slope of `last_yield_vs_tenor` (yield change per +1 year). Positive slope = upward-sloping (normal) curve; negative = inverted.
   - **By issuer** (when comparing): issuer × count × mean coupon × mean yield.
4. Add a brief, plain interpretation (2–4 sentences max) — what the correlation/slope/variance means for an investor. Always state the data date (`as_of`) and that yields cover only traded bonds.

Caveats to surface when relevant: data is PDS month-end (not intraday); credit ratings are not available; many bonds have no last-traded yield so yield-based stats use a smaller sample (`count_with_yield`).
