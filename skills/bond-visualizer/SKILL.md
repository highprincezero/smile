---
name: bond-visualizer
description: Use when the user asks to VISUALIZE, chart, plot, graph, or "show me" bond data — yield curve, coupon distribution, yield-by-issuer, volume, holdings vs market, or any trend. Renders text/Unicode bar charts in chat.
---

# Bond visualizer workflow

The chat is text-only — render "charts" as **Unicode bar charts inside a fenced code block** (monospace makes the bars line up). Never claim to draw an image/graph.

1. Get real data first — never invent numbers:
   - `bond_stats` for distributions, trends, yield-curve, by-issuer.
   - `list_bonds` / `analyze_bond` for specific securities.
   - If the user attached a file, chart from it (and/or compare against PDS).

2. Build a horizontal bar chart in a ``` code block:
   - One row per item: `label  bar  value`.
   - Bar = `█` repeated proportional to the value, scaled so the **max ≈ 20 blocks**; pad shorter bars with `░`.
   - Sort sensibly: by tenor for a yield curve, by value for a ranking.
   - Pick the charted metric from the question (yield, coupon, volume…).

3. Add a one-line title above the chart, then 2–3 short bullets interpreting it (with blank lines between them).

4. Ground every value in tool/file output. State the data date (`as_of`).

Example shape (illustrative — use real values):
```
Yield by tenor — green issuers (PDS Apr 30 2026)
1.1y  EDC 05-27  ████████████████░░░░  6.05%
3.1y  EDC 05-29  █████████████████░░░  6.63%
5.1y  EDC 05-31  ████████████████████  6.95%
```
