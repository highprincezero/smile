---
name: bond-visualizer
description: Use ONLY when the user explicitly asks to VISUALIZE, chart, plot, graph, or "show me" bond data — yield curve, coupon distribution, yield-by-issuer, volume, holdings vs market, or a trend. Renders a Unicode bar chart via the visualize tool.
---

# Bond visualizer workflow

Render charts with the `visualize` tool (it returns a Unicode bar chart in a code block). Do NOT hand-draw bars yourself.

1. **Reuse data you already have.** If an earlier step in this conversation already produced the numbers — a `bond_stats` result, a `list_bonds` set, an `analyze_bond` lookup, a strategy table, or the user's uploaded file — pass those figures straight to `visualize`. Do NOT re-run those tools just to chart the same thing.

2. Only call a data tool if you genuinely don't have the needed numbers yet.

3. Call `visualize(title, data)`:
   - `data` = newline-separated `label = value` rows, value numeric (`%` ok). Example: `EDC 05-31 R28 = 6.95`.
   - Pick the metric and ordering from the request (yield by tenor for a curve, value-sorted for a ranking).

4. After the returned chart, add 2–3 short interpretation bullets (blank line between them). State the data date.

Only visualize when the user explicitly asks — don't append a chart to ordinary answers.
