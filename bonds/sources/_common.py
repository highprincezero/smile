"""
Shared helpers for PDS source adapters.
"""

from dataclasses import dataclass

import httpx

USER_AGENT = "Greenfy-Smile/1.0 (+https://greenfy.ai)"
DEFAULT_TIMEOUT = httpx.Timeout(15.0, connect=5.0)


@dataclass(frozen=True)
class SourceResult:
    snippet: str       # smallest fragment that contains the security's row(s)
    source_url: str    # exact URL the snippet came from


class SourceFetchError(Exception):
    """Upstream page could not be fetched (network, non-200, etc.)."""


class SecurityNotFound(Exception):
    """Page fetched OK but no row matched the security_id."""


async def fetch_html(url: str) -> str:
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml"},
        timeout=DEFAULT_TIMEOUT,
        follow_redirects=True,
    ) as client:
        try:
            resp = await client.get(url)
        except httpx.HTTPError as e:
            raise SourceFetchError(f"network error fetching {url}: {e}") from e
        if resp.status_code != 200:
            raise SourceFetchError(f"upstream returned {resp.status_code} for {url}")
        return resp.text


def normalize_id(s: str) -> str:
    """Loose matcher: collapse whitespace + uppercase. Lets 'fxtn 10-65' match 'FXTN  10-65'."""
    return " ".join(s.split()).upper()
