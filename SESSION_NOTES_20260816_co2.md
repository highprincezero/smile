# Session 2026-08-16 — CO2 automated refresh (greenfy/smile)

## Shipped (uncommitted, ~/Desktop/smile)
- `bonds/data/co2_disclosures.json` — verified figures: ACEN 373,328 t (9M 2024, bond-level, PHP bonds), ALCO 27,048.9 t (FY2022, programme). EDC + CREIT: none published → absent.
- `bonds/schema.py` — `CO2Disclosure` model (value/period/scope/as_of/source_url/note) replaces bare `co2_avoided_t` float.
- `bonds/enrich.py` — loads JSON (mtime cache), attaches record in `classify()` for green bonds.
- `bonds/co2_refresh.py` — deterministic pipeline: httpx+pdfplumber fetch issuer disclosures → tool-less Claude extraction → guarded atomic write. Weekly TTL. Triggers: green_registry._load + server startup.
- 2 live runs byte-identical. Guard: figure only written if cited to a fetched doc.

## Key findings
- ACEN page lists per-instrument figures; PDS bonds = PHP Green Bonds = 373,328 (agents summing all instruments caused earlier 2.5M/3M wobble).
- EDC publishes NO bond-attributable CO2 (checked 17-A, IR2024/2025 site+PDFs). Old seed 125,493.9 uncitable → dropped.
- CREIT 2025 ASR: aspirational only → correctly none.
- energy.com.ph HTML is WAF-403'd but wp-content PDFs + integratedreport subdomain fetch fine.

## Pending
- Deploy (pull→push→smile-sync) — awaiting approval
- Frontend CO2 row on /bonds/:id (greenfy repo not local)
- gdoc row 5 note: EDC/CREIT have no published figure
