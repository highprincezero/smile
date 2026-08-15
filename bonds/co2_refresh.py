"""
Automated CO2e-avoided refresher (deterministic pipeline).

CO2 figures have no structured feed -- each issuer publishes them in its own
Green Bond Impact Report / use-of-proceeds disclosure. This module:

  1. FETCHES each issuer's known disclosure documents itself (httpx; pdfplumber
     for PDFs, tag-strip for HTML). No agent browsing -> no SDK buffer limits,
     no run-to-run variance in what was read.
  2. Hands the *document text* to a tool-less Claude extraction call that must
     return one strict JSON record (value + period + scope + source).
  3. Writes validated records to `data/co2_disclosures.json` (atomic). A figure
     is only written with its source URL; failures keep the existing record.

The app serves from the JSON via an mtime cache (bonds/enrich.py).
Triggers: bond-data pull + app startup, weekly-TTL-gated (see green_registry,
server.py). Run manually:  python -m bonds.co2_refresh [--force]
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
from pathlib import Path

import httpx

_DATA_PATH = Path(__file__).with_name("data") / "co2_disclosures.json"
_STAMP_PATH = Path(__file__).with_name("data") / ".co2_refresh_stamp"
_TTL = float(os.getenv("CO2_REFRESH_TTL", 7 * 24 * 3600))  # weekly
_HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
_MAX_DOC_CHARS = 30_000  # cap text handed to the extractor

# Issuer disclosure documents, keyed by the PDS ticker stem enrich.py classifies by.
# `docs` = the issuer's own green-bond impact/use-of-proceeds disclosures (canonical
# home of the figure). `discover` = a listing page scanned for NEWER report PDFs so
# the pipeline picks up next year's report without a code change.
_ISSUERS: dict[str, dict] = {
    "ACEN": {
        "name": "ACEN Corporation, Philippines (PDS-listed PHP green bonds)",
        "docs": ["https://www.acenrenewables.com/investors/green-finance-leadership/use-of-green-bond-proceeds/"],
        "hint": "Use the PHP Green Bonds figure only (the PDS-listed bonds are the PHP issuances) -- NOT USD bonds, AUD loans, or preferred equity, and do NOT sum across instruments.",
    },
    "EDC": {
        "name": "Energy Development Corporation (EDC), Philippines",
        "docs": [
            "https://integratedreport.energy.com.ph/financial-capital/",
            "https://integratedreport.energy.com.ph/natural-capital/",
        ],
        "hint": "Only a figure explicitly attributed to the ASEAN green bonds / their financed projects counts. Company-wide avoided emissions (e.g. 'total avoided in lieu of coal') do NOT count.",
    },
    "ALCO": {
        "name": "Arthaland Corporation, Philippines",
        "docs": ["https://arthaland.com/assets/documents/ASEAN-Green-Bonds-2022-Impact-Report-Final.pdf"],
        "discover": {
            "page": "https://arthaland.com/sustainability",
            "pattern": r"[^\"']*Green-Bonds[^\"']*Impact[^\"']*\.pdf",
        },
        "hint": "Use the report's TOTAL avoided GHG emissions for the green-bond portfolio; convert kg to tonnes if needed.",
    },
    "CREIT": {
        "name": "Citicore Renewable Energy REIT (CREIT), Philippines",
        "docs": [],  # discovered: latest Annual Sustainability Report
        "discover": {
            "page": "https://creit.com.ph/investor-relations/annual-and-sustainability-reports/",
            "pattern": r"/assets/[^\"']*(?:ASR|Sustainability)[^\"']*\.pdf",
        },
        "hint": "Only an ACHIEVED figure attributed to green bond/loan financed assets counts -- not aspirational ('expected to reduce') or portfolio-potential statements.",
    },
}

_EXTRACT_PROMPT = """You are a precise data extractor. Below is text from {issuer}'s own \
disclosure document(s). Extract the MOST RECENT figure for CO2e emissions AVOIDED that the \
issuer attributes to its GREEN BONDS (or green-bond-financed projects/portfolio).

{hint}

Rules: use ONLY the text below -- no outside knowledge, no estimates. If the text contains \
no such issuer-attributed figure, return found=false.

Return ONE JSON object, nothing else:
{{"found": true|false, "value_t": <tonnes CO2e as a number, or null>, "period": "<e.g. '9M 2024', 'FY2022', or null>", "scope": "bond-level|programme-level|null", "as_of": "<YYYY-MM-DD or null>", "source_url": "<which SOURCE URL below the figure came from, or null>"}}

{docs}
"""


def _stale() -> bool:
    try:
        return (time.time() - float(_STAMP_PATH.read_text())) > _TTL
    except (OSError, ValueError):
        return True


def _load() -> dict:
    try:
        return json.loads(_DATA_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _valid(rec: dict, allowed_urls: list[str]) -> bool:
    """Trust a figure only if positive, scoped, and cited to a doc we actually fetched."""
    return (
        bool(rec.get("found"))
        and isinstance(rec.get("value_t"), (int, float))
        and rec["value_t"] > 0
        and bool(rec.get("period"))
        and rec.get("scope") in ("bond-level", "programme-level")
        and rec.get("source_url") in allowed_urls
    )


def _pdf_text(content: bytes) -> str:
    import pdfplumber

    out = []
    with pdfplumber.open(io.BytesIO(content)) as pdf:
        for page in pdf.pages:
            t = page.extract_text() or ""
            # keep only pages that can contain the figure -- caps prompt size
            if re.search(r"CO2|CO\u2082|GHG|avoid|emission|green bond|proceeds", t, re.I):
                out.append(t)
    return "\n".join(out)


def _html_text(content: bytes) -> str:
    s = content.decode("utf-8", errors="ignore")
    s = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", s, flags=re.S | re.I)
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"\s+", " ", s)


async def _fetch_doc(client: httpx.AsyncClient, url: str) -> str | None:
    try:
        r = await client.get(url)
        r.raise_for_status()
    except httpx.HTTPError as e:
        print(f"[co2_refresh] fetch failed {url}: {e!r}")
        return None
    text = _pdf_text(r.content) if r.content[:4] == b"%PDF" else _html_text(r.content)
    return text or None


async def _discover_docs(client: httpx.AsyncClient, meta: dict) -> list[str]:
    """Docs list + any newer report PDFs found on the issuer's listing page."""
    docs = list(meta.get("docs", []))
    d = meta.get("discover")
    if d:
        try:
            r = await client.get(d["page"])
            hits = sorted(set(re.findall(d["pattern"], r.text)), reverse=True)
            if hits:
                base = re.match(r"https?://[^/]+", d["page"]).group(0)
                newest = hits[0] if hits[0].startswith("http") else base + hits[0]
                if newest not in docs:
                    docs.insert(0, newest)
        except httpx.HTTPError as e:
            print(f"[co2_refresh] discover failed {d['page']}: {e!r}")
    return docs


async def _extract(issuer_name: str, hint: str, docs: dict[str, str]) -> dict | None:
    """Tool-less extraction call over fetched document text -> validated record or None."""
    from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient

    body = "\n\n".join(
        f"SOURCE URL: {url}\n---\n{text[:_MAX_DOC_CHARS // max(1, len(docs))]}"
        for url, text in docs.items()
    )
    prompt = _EXTRACT_PROMPT.format(issuer=issuer_name, hint=hint, docs=body)
    opts = ClaudeAgentOptions(
        model=os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"),
        allowed_tools=[],  # extraction only -- deterministic input, no browsing
        max_turns=1,
    )
    text = ""
    async with ClaudeSDKClient(options=opts) as client:
        await client.query(prompt)
        async for msg in client.receive_response():
            for block in getattr(msg, "content", []) or []:
                if getattr(block, "text", None):
                    text += block.text
    try:
        rec = json.loads(text[text.index("{"): text.rindex("}") + 1])
    except (ValueError, json.JSONDecodeError):
        return None
    if not _valid(rec, list(docs)):
        return None
    return {
        "value_t": rec["value_t"],
        "period": rec["period"],
        "scope": rec["scope"],
        "as_of": rec.get("as_of"),
        "source_url": rec["source_url"],
        "note": "Issuer-published; auto-refreshed from issuer disclosure.",
    }


async def refresh_co2(force: bool = False) -> dict:
    """Fetch every tracked issuer's disclosures, extract figures, merge validated
    records into the JSON. Keeps existing records on any per-issuer failure."""
    if not force and not _stale():
        return _load()

    data = _load()
    async with httpx.AsyncClient(headers=_HEADERS, timeout=httpx.Timeout(90.0, connect=15.0),
                                 follow_redirects=True) as client:
        for stem, meta in _ISSUERS.items():
            try:
                urls = await _discover_docs(client, meta)
                docs = {}
                for url in urls:
                    text = await _fetch_doc(client, url)
                    if text:
                        docs[url] = text
                if not docs:
                    print(f"[co2_refresh] {stem}: no documents fetched -- keeping existing")
                    continue
                rec = await _extract(meta["name"], meta.get("hint", ""), docs)
            except Exception as e:  # never let one issuer break the run
                print(f"[co2_refresh] {stem}: error {e!r} -- keeping existing")
                continue
            if rec:
                data[stem] = rec
                print(f"[co2_refresh] {stem}: {rec['value_t']} tCO2e ({rec['period']}) <- {rec['source_url']}")
            else:
                print(f"[co2_refresh] {stem}: no attributable figure in disclosures -- keeping existing")

    tmp = _DATA_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    tmp.replace(_DATA_PATH)  # atomic
    _STAMP_PATH.write_text(str(time.time()))
    print(f"[co2_refresh] wrote {len(data)} issuers -> {_DATA_PATH}")
    return data


def maybe_refresh_in_background() -> None:
    """Fire-and-forget refresh if the cache is stale. Safe to call at app startup."""
    if not _stale():
        return
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(refresh_co2())
    except RuntimeError:
        pass  # no running loop; next in-loop trigger will handle it


if __name__ == "__main__":
    import sys
    asyncio.run(refresh_co2(force="--force" in sys.argv))
