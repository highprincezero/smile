"""
Source adapters. Each adapter exposes:

    async def fetch_snippet(security_id: str) -> SourceResult

returning the smallest HTML/text fragment containing the security's row,
plus the resolved source URL. The normalizer takes it from there.
"""

from .pds_government import fetch_snippet as fetch_government
from .pds_corporate import fetch_snippet as fetch_corporate

__all__ = ["fetch_government", "fetch_corporate"]
