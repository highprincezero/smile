"""
Source adapters.

- Government: `fetch_government(security_id) -> SourceResult` (HTML snippet → LLM normalize).
- Corporate:  `fetch_corporate_analysis(security_id) -> BondAnalysis` (PDS Board Summary
  PDFs parsed deterministically — no LLM normalization).
"""

from .pds_government import fetch_snippet as fetch_government
from .pds_corporate import fetch_analysis as fetch_corporate_analysis

__all__ = ["fetch_government", "fetch_corporate_analysis"]
