"""
Pydantic schema for /api/analyze-bond — single source of truth for the
request and response shapes. The normalizer LLM is forced into BondAnalysis.
"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SecurityType = Literal["government", "corporate"]


class AnalyzeBondRequest(BaseModel):
    security_id: str = Field(..., min_length=1, description="Symbol/name as it appears on PDS, e.g. 'FXTN 10-65'.")
    type: SecurityType = Field(..., description="'government' or 'corporate'.")
    refresh: bool = Field(False, description="Bypass the in-memory TTL cache.")


class BondAnalysis(BaseModel):
    """Normalized representation of a single Philippine fixed-income security."""

    security_id: str
    type: SecurityType

    issuer: Optional[str] = None
    name: Optional[str] = None
    isin: Optional[str] = None

    coupon: Optional[float] = Field(None, description="Annual coupon rate, percent.")
    maturity: Optional[date] = None
    tenor_years: Optional[float] = None
    currency: Optional[str] = "PHP"

    last_price: Optional[float] = None
    last_yield: Optional[float] = Field(None, description="Last traded yield, percent.")
    bid_yield: Optional[float] = None
    ask_yield: Optional[float] = None

    trade_date: Optional[date] = None
    volume: Optional[float] = Field(None, description="Face-value volume in PHP.")

    source_url: str
    fetched_at: datetime
    normalized_by: str = Field(..., description="Model id used for normalization.")
    raw_excerpt: str = Field(..., description="Short snippet from the source for traceability.")


class BondError(BaseModel):
    code: str
    message: str


class BondErrorResponse(BaseModel):
    error: BondError
