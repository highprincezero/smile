"""
PDS government-securities source adapter.

Locates the row matching `security_id` on the government securities page and
returns a focused snippet for downstream LLM normalization.
"""

from bs4 import BeautifulSoup, Tag

from ._common import SourceResult, SecurityNotFound, fetch_html, normalize_id

URL = "https://pds.com.ph/government-securities/"

# Cap snippet size so the LLM call stays cheap. A single row + header is plenty.
MAX_SNIPPET_CHARS = 4000


async def fetch_snippet(security_id: str) -> SourceResult:
    html = await fetch_html(URL)
    soup = BeautifulSoup(html, "html.parser")

    target = normalize_id(security_id)
    matched_row = _find_row(soup, target)
    if matched_row is None:
        raise SecurityNotFound(f"'{security_id}' not found on {URL}")

    snippet = _extract_snippet(matched_row)[:MAX_SNIPPET_CHARS]
    return SourceResult(snippet=snippet, source_url=URL)


def _find_row(soup: BeautifulSoup, target: str) -> Tag | None:
    """Find the <tr> whose text contains the (loose-normalized) security id."""
    for tr in soup.find_all("tr"):
        if target in normalize_id(tr.get_text(" ", strip=True)):
            return tr
    return None


def _extract_snippet(row: Tag) -> str:
    """Return the matched row plus its enclosing table's header row, if any."""
    table = row.find_parent("table")
    if table is None:
        return str(row)

    header = table.find("tr")  # first <tr> is typically the header
    parts: list[str] = []
    if header is not None and header is not row:
        parts.append(str(header))
    parts.append(str(row))
    return "\n".join(parts)
