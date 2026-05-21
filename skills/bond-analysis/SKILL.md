---
name: bond-analysis
description: Use when the user mentions a specific bond by security ID, ticker, or asks for yield, maturity, rating, or bond data analysis.
---

# Bond analysis workflow

When the user asks about a specific bond:

1. Call `analyze_bond(security_id, type)` to fetch raw data. `type` is one of `"government"` or `"corporate"`.
2. **If the result starts with `LIVE_BOND_DATA_UNAVAILABLE`** (or is any error): tell the user that live bond data lookup is temporarily offline, and do NOT fabricate, estimate, or guess any price/yield/maturity/rating. You may still explain the bond type in general terms and offer to help once the lookup is restored. Stop here — do not produce a data table.
3. Only if real data is returned: if a description/excerpt is long (over ~300 words), call `summarize(text)` on it, then present **yield, maturity date, rating, and issuer** in a markdown table, in Philippine peso (₱), and cite the returned source URL.

## Always present bond info as tables

Any time you list bonds, issuers, or suggested securities — or show bond data — use a **markdown table**, never bullet points or prose lists.

- If the user asks broadly (e.g. "latest bond prices", "what can I look at?") **without** a specific ID, do NOT dump a paragraph of bullets. Reply with a short one-line intro, then a table of options, then one closing line asking them to pick. Use the **Local ID** format PDS uses (e.g. `ALI 26 R24`, `BDO 01-27`).

  | Issuer | Type | Example Local ID |
  |---|---|---|
  | BDO Unibank | Corporate | `BDO 01-27` |
  | Bank of the PH Islands | Corporate | `BPI 12-26 R26` |
  | SM Prime Holdings | Corporate | `SMPH 26 R23` |

- When real data is returned, present it as a single metrics table (Issuer, ISIN, Coupon, Maturity, Last Price, Last Yield, Bid Yield, Volume) in Philippine peso (₱), then at most 2–3 short interpretation lines.

Keep replies tight: no repeated capability menus. Never imply you can fetch live market prices when the tool reports it cannot. Do NOT call `extract_keywords` on bond data.
