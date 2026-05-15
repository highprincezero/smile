---
name: bond-analysis
description: Use when the user mentions a specific bond by security ID, ticker, or asks for yield, maturity, rating, or bond data analysis.
---

# Bond analysis workflow

When the user asks about a specific bond:

1. Call `analyze_bond(security_id, type)` to fetch raw data. `type` is one of `"government"` or `"corporate"`.
2. If the description or prospectus excerpt is long (over ~300 words), call `summarize(text)` on it.
3. Present **yield, maturity date, rating, and issuer** in a markdown table.

Use Philippine peso (₱) for currency. Cite the source URL returned in the bond data.

Do NOT call `extract_keywords` on bond data — the structured fields are already available.
