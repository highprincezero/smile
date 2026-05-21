---
name: bond-strategist
description: Use when the user wants a STRATEGY, plan, recommendation, or next action based on bonds — e.g. "what should I buy", "build me a ladder/portfolio", "how should I position", "given the yield curve what's the play", "what's my next move". Builds on bond-data-collector (real securities) and bond-analyst (stats) to produce a concrete plan.
---

# Bond strategist workflow

When the user wants a strategy, recommendation, or next action on bonds, REASON FROM REAL DATA — never strategize from memory or invented numbers.

1. **Gather inputs first** (this is mandatory before recommending anything):
   - Call `bond_stats` (whole market, or an issuer) for the yield-curve slope, variance/dispersion, and correlations.
   - Call `list_bonds` (and `analyze_bond` for specific candidates) to get real securities with their Local ID, coupon, maturity, and yield.
2. **Anchor on the objective.** If the user's goal/horizon/risk appetite is unclear, ask in ONE short line (income vs. capital preservation, time horizon, risk tolerance). If they've given enough, proceed.
3. **Derive the strategy from the numbers:**
   - Use the `last_yield_vs_tenor` slope from `bond_stats`: clearly upward → extending duration captures term premium; flat/inverted → stay short.
   - Use dispersion/variance to decide concentration vs. diversification across issuers.
   - Choose a structure: **ladder** (spread maturities for reinvestment flexibility), **barbell** (short + long, skip the middle), or a targeted **income pick** — justified by the curve and the user's horizon.
4. **Present the plan:**
   - One-line **thesis**.
   - A **strategy table**: Action | Bond (real Local ID) | Tenor | Yield | Rationale.
   - 2–4 lines on **risks / assumptions** (rate moves, liquidity, credit — note ratings aren't available from the data source).
5. Every security ID and number must come from tool output. State the data date (`as_of`). Close with: this is informational, not financial advice.
