"""
Authoritative green-bond registry, sourced from PDS itself.

PDS publishes a "Listed Securities Database" PDF on its public S3 bucket. Each
row has an ISSUE description column; officially green-labeled issues say
"Green Bonds" / "ASEAN Green Bonds" there. We parse that PDF, collect the
SERIES CODE (== our PDS Local ID) of every green row, and cache the set.

This REPLACES issuer heuristics: a bond is green iff PDS labels its issue green.
Source of truth, auto-refreshing — no hardcoded issuer guesses.
"""

import asyncio
import io
import re
import time

import httpx
import pdfplumber

from .sources._common import normalize_id

LISTING_PAGE = "https://www.pds.com.ph/listing-and-enrollment/"
_PDF_RE = r"https://pdswordpressbucket[^\"']*Listed-Securities-Database[^\"']*\.pdf"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": LISTING_PAGE}
_TTL = 6 * 3600  # PDS Listed Securities DB updates ~daily; refresh a few times a day

_cache: dict = {"ts": 0.0, "as_of": None, "source_url": None, "ids": set(), "detail": {}}
_lock = asyncio.Lock()


def _latest_pdf_url(html: str) -> str | None:
    hits = sorted(set(re.findall(_PDF_RE, html)))
    return hits[-1] if hits else None


def _parse_green(pdf_bytes: bytes) -> tuple[set[str], dict]:
    ids: set[str] = set()
    detail: dict[str, str] = {}
    header = None
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for tbl in page.extract_tables() or []:
                for row in tbl:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if cells and cells[0] == "ISSUER":
                        header = cells
                        continue
                    if not header or not any(cells):
                        continue
                    d = dict(zip(header, cells))
                    issue = d.get("ISSUE", "")
                    series = d.get("SERIES CODE", "")
                    if series and re.search(r"green", issue, re.I):
                        nid = normalize_id(series)
                        ids.add(nid)
                        detail[nid] = issue
    return ids, detail


async def _load() -> dict:
    if time.time() - _cache["ts"] < _TTL and _cache["ids"]:
        return _cache
    async with _lock:
        if time.time() - _cache["ts"] < _TTL and _cache["ids"]:
            return _cache
        async with httpx.AsyncClient(headers=_HEADERS, timeout=httpx.Timeout(40.0, connect=10.0),
                                     follow_redirects=True) as client:
            html = (await client.get(LISTING_PAGE)).text
            url = _latest_pdf_url(html)
            if not url:
                # keep any stale cache rather than wiping green flags on a transient failure
                return _cache
            pdf_bytes = (await client.get(url)).content
        ids, detail = _parse_green(pdf_bytes)
        m = re.search(r"as-of-([\d.]+)\.pdf", url)
        if ids:
            _cache.update(ts=time.time(), as_of=(m.group(1) if m else None),
                          source_url=url, ids=ids, detail=detail)
        return _cache


async def green_ids() -> set[str]:
    """Normalized Local IDs of every PDS green-labeled security. Empty set only
    if PDS is unreachable and nothing was ever cached."""
    return set((await _load())["ids"])


async def green_detail(local_id: str) -> str | None:
    """The PDS ISSUE text for a green security (e.g. 'ASEAN Green Bonds Due 2027')."""
    c = await _load()
    return c["detail"].get(normalize_id(local_id))
