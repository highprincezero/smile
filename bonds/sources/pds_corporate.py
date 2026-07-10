"""
PDS corporate-securities source adapter.

The public PDS site is JS-rendered behind a WAF, so the old HTML scrape returns
nothing. Instead we read the official month-end "Corporate Board Summary"
PDFs that PDS publishes on a public S3 bucket (no WAF), parse the Price and
Yield tables with pdfplumber, and build a BondAnalysis directly — deterministic,
no LLM normalization needed.
"""

import asyncio
import io
import re
import time
from datetime import datetime, timezone

import httpx
import pdfplumber

from ..enrich import classify
from ..green_registry import green_detail, green_ids
from ..schema import BondAnalysis
from ._common import SecurityNotFound, SourceFetchError, normalize_id

REPORTS_PAGE = "https://www.pds.com.ph/downloadable-reports/"
_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.pds.com.ph/"}
_TTL_SECONDS = 6 * 3600  # board summary is month-end; refresh a few times a day is plenty

# Module-level cache of the parsed board so we don't re-download the PDFs per lookup.
_cache: dict = {"ts": 0.0, "date": None, "source_url": None, "rows": []}
_lock = asyncio.Lock()


def _num(s: str | None) -> float | None:
    if not s:
        return None
    s = s.replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return None


def _date(s: str | None):
    if not s:
        return None
    for fmt in ("%m/%d/%Y", "%m/%d/%y"):
        try:
            return datetime.strptime(s.strip(), fmt).date()
        except ValueError:
            continue
    return None


def _rows_from_pdf(pdf: bytes) -> tuple[list[str] | None, list[list[str]]]:
    header, out = None, []
    with pdfplumber.open(io.BytesIO(pdf)) as p:
        for page in p.pages:
            for table in page.extract_tables() or []:
                for row in table:
                    cells = [(c or "").replace("\n", " ").strip() for c in row]
                    if cells and cells[0] == "Global ID":
                        header = cells
                        continue
                    if cells and cells[0]:
                        out.append(cells)
    return header, out


async def _latest_report_urls(client: httpx.AsyncClient) -> tuple[str, str, str]:
    """Return (price_pdf_url, yield_pdf_url, as_of_label)."""
    html = (await client.get(REPORTS_PAGE)).text

    def latest(pat: str) -> str | None:
        files = sorted(set(re.findall(pat, html)))
        return files[-1] if files else None

    price = latest(r'href=["\']([^"\']*Corporate-Board-Summary-Price[^"\']*\.pdf)["\']')
    yld = latest(r'href=["\']([^"\']*Corporate-Board-Summary-Yield[^"\']*\.pdf)["\']')
    if not price or not yld:
        raise SourceFetchError("could not locate Corporate Board Summary PDFs on PDS")
    m = re.search(r"as-of-([A-Za-z]+-\d+-\d+)", price)
    as_of = m.group(1) if m else ""
    return price, yld, as_of


async def _load_board() -> dict:
    """Download + parse the latest Price/Yield board summaries; cache in-process."""
    if time.time() - _cache["ts"] < _TTL_SECONDS and _cache["rows"]:
        return _cache

    async with _lock:
        if time.time() - _cache["ts"] < _TTL_SECONDS and _cache["rows"]:
            return _cache
        try:
            async with httpx.AsyncClient(
                headers=_HEADERS, timeout=httpx.Timeout(40.0, connect=10.0), follow_redirects=True
            ) as client:
                price_url, yield_url, as_of = await _latest_report_urls(client)
                price_pdf = (await client.get(price_url)).content
                yield_pdf = (await client.get(yield_url)).content
        except httpx.HTTPError as e:
            raise SourceFetchError(f"network error fetching PDS board summary: {e}") from e

        ph, pr = _rows_from_pdf(price_pdf)
        yh, yr = _rows_from_pdf(yield_pdf)
        if not ph or not pr:
            raise SourceFetchError("PDS Corporate Board Summary (price) had no parseable rows")

        yidx = {}
        for r in yr:
            d = dict(zip(yh, r))
            yidx[(d.get("ISIN", "-"), d.get("Local ID", ""))] = d

        rows = []
        for r in pr:
            d = dict(zip(ph, r))
            y = yidx.get((d.get("ISIN", "-"), d.get("Local ID", "")), {})
            rows.append({"price": d, "yield": y})

        _cache.update(ts=time.time(), date=as_of, source_url=price_url, rows=rows)
        return _cache


def _matches(target: str, price: dict) -> bool:
    fields = [price.get("Local ID", ""), price.get("Global ID", ""),
              price.get("Domestic No.", ""), price.get("ISIN", "")]
    norm = [normalize_id(f) for f in fields if f and f != "-"]
    return any(target == f or target in f for f in norm)


# Common issuer names → PDS ticker stems (the PDFs only carry tickers like ALIPM).
_ALIASES = {
    "AYALA": ["ALI", "AC", "ACEN"],
    "AYALA LAND": ["ALI"],
    "AC ENERGY": ["ACEN"],
    "ACEN": ["ACEN"],
    "BANK OF THE PHILIPPINE ISLANDS": ["BPI"],
    "BPI": ["BPI"],
    "BDO": ["BDO"],
    "BANCO DE ORO": ["BDO"],
    "SM PRIME": ["SMPH"],
    "SM": ["SMPH", "SMCGP", "SMIC", "SMC"],
    "ABOITIZ": ["AP", "AEV"],
    "ROBINSONS": ["RLC"],
    "RCBC": ["RCB"],
    "METROBANK": ["MBT"],
    "SECURITY BANK": ["SECB"],
    "ENERGY DEVELOPMENT": ["EDC"],
    "EDC": ["EDC"],
    "PETRON": ["PCOR"],
    "SAN MIGUEL": ["SMC", "SMCGP", "PCOR"],
    "MERALCO": ["MER"],
    "FILINVEST": ["FLI", "FDC"],
}


def _alias_prefixes(q: str) -> set[str]:
    prefixes: set[str] = set()
    for name, tickers in _ALIASES.items():
        if name in q or q in name:
            prefixes.update(normalize_id(t) for t in tickers)
    return prefixes


async def board_as_of() -> str | None:
    """The 'as of' date label of the current PDS Corporate Board Summary."""
    return (await _load_board()).get("date")


async def list_securities(
    query: str | None = None, limit: int = 500, green_only: bool = False
) -> list[dict]:
    """Return corporate securities (real PDS Local IDs). Returns ALL bonds by
    default, each carrying a `green` flag (authoritative, from the PDS Listed
    Securities Database). Pass green_only=True to filter to green only."""
    board = await _load_board()
    greens = await green_ids()
    q = normalize_id(query) if query else None
    prefixes = _alias_prefixes(q) if q else set()
    out: list[dict] = []
    for row in board["rows"]:
        p = row["price"]
        local = p.get("Local ID", "")
        if not local:
            continue
        hay = normalize_id(" ".join([p.get("Local ID", ""), p.get("Global ID", ""), p.get("ISIN", "")]))
        if q and q not in hay and not any(pref in hay for pref in prefixes):
            continue
        issuer = (p.get("Global ID", "").split() or [""])[0]
        intel = classify(local, issuer, green_ids=greens, green_issue=await green_detail(local))
        is_green = intel.theme == "green"
        if green_only and not is_green:
            continue
        out.append({
            "local_id": local,
            "issuer": issuer,
            "isin": (p.get("ISIN") if p.get("ISIN") not in ("", "-") else None),
            "coupon": _num(p.get("CPN")),
            "maturity": p.get("Maturity", ""),
            "sector": intel.sector,
            "sub_sector": intel.sub_sector,
            "theme": intel.theme,
            "green": is_green,
            "taxonomies": intel.taxonomies.model_dump(),
            "confidence": intel.confidence,
        })
        if len(out) >= limit:
            break
    return out


async def fetch_analysis(security_id: str) -> BondAnalysis:
    board = await _load_board()
    target = normalize_id(security_id)

    match = next((row for row in board["rows"] if _matches(target, row["price"])), None)
    if match is None:
        raise SecurityNotFound(
            f"'{security_id}' not found in PDS Corporate Board Summary "
            f"(as of {board['date']})"
        )

    p, y = match["price"], match["yield"]
    vol_mm = _num(p.get("D. Vol (MM)"))
    issuer = (p.get("Global ID", "").split() or [None])[0]
    local = p.get("Local ID")
    return BondAnalysis(
        intelligence=classify(local, issuer, green_ids=await green_ids(),
                              green_issue=await green_detail(local)),
        security_id=security_id,
        type="corporate",
        issuer=issuer,
        name=p.get("Local ID") or None,
        isin=(p.get("ISIN") if p.get("ISIN") not in ("", "-") else None),
        coupon=_num(p.get("CPN")),
        maturity=_date(p.get("Maturity")),
        tenor_years=_num(p.get("YRS")),
        currency="PHP",
        last_price=_num(p.get("Last Price")),
        last_yield=_num(y.get("Last Yield")),
        bid_yield=_num(y.get("Bid Yield")),
        ask_yield=_num(y.get("Offer Yield")),
        trade_date=None,
        volume=(vol_mm * 1_000_000 if vol_mm is not None else None),
        source_url=board["source_url"],
        fetched_at=datetime.now(timezone.utc),
        normalized_by="pds-board-summary-parser",
        raw_excerpt=(
            f"{p.get('Local ID','')} | ISIN {p.get('ISIN','')} | CPN {p.get('CPN','')} | "
            f"Maturity {p.get('Maturity','')} | Last Price {p.get('Last Price','')} | "
            f"Last Yield {y.get('Last Yield','')} (PDS Board Summary as of {board['date']})"
        ),
    )
