"""
Pydantic schema for /api/analyze-bond — single source of truth for the
request and response shapes. The normalizer LLM is forced into BondAnalysis.
"""

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field

SecurityType = Literal["government", "corporate"]

AlignmentStatus = Literal["aligned", "partial", "not_aligned", "unknown"]
Theme = Literal["green", "social", "sustainability", "transition", "blue", "conventional", "unknown"]


class TaxonomyAlignment(BaseModel):
    """Per-framework alignment verdict for one security."""

    eu: AlignmentStatus = "unknown"
    cbi_mitigation: AlignmentStatus = "unknown"
    cbi_resilience: AlignmentStatus = "unknown"
    ph_sftg: AlignmentStatus = "unknown"
    asean_gss: AlignmentStatus = "unknown"


class CO2Disclosure(BaseModel):
    """A single issuer-published CO2e-avoided figure. NOT a constant: it is a
    point-in-time disclosure, meaningful only with its period + scope + source,
    and refreshed whenever the issuer publishes a newer report."""

    value_t: float = Field(..., description="Tonnes CO2e avoided, exactly as disclosed.")
    period: str = Field(..., description="Reporting period the figure covers, e.g. '9M 2024', 'FY2022', 'cumulative to Dec 2025'.")
    scope: str = Field(..., description="'bond-level' or 'programme-level'.")
    as_of: Optional[str] = Field(None, description="Disclosure/report date (YYYY-MM-DD).")
    source_url: Optional[str] = Field(None, description="Link to the issuer document the figure came from.")
    note: Optional[str] = Field(None, description="Any qualifier on how the figure is attributed.")


class BondIntelligence(BaseModel):
    """Classification + sustainability intelligence layered on top of raw PDS data.

    v1 is deterministic (issuer/programme rules) — every verdict carries a
    confidence and basis so downstream UIs can label indicative data honestly.
    """

    sector: Optional[str] = None
    sub_sector: Optional[str] = None
    theme: Theme = "unknown"
    taxonomies: TaxonomyAlignment = Field(default_factory=TaxonomyAlignment)
    co2: Optional["CO2Disclosure"] = Field(
        None,
        description="Issuer-published CO2e-avoided disclosure (value + period + scope + source). "
                    "None unless the issuer has published a bond- or programme-attributable figure.",
    )
    confidence: Literal["verified", "indicative", "unknown"] = "unknown"
    basis: Optional[str] = Field(None, description="Short human-readable rationale for the classification.")


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

    intelligence: Optional[BondIntelligence] = Field(
        None, description="Sector/theme/taxonomy classification (Intelligence layer)."
    )

    source_url: str
    fetched_at: datetime
    normalized_by: str = Field(..., description="Model id used for normalization.")
    raw_excerpt: str = Field(..., description="Short snippet from the source for traceability.")


class BondError(BaseModel):
    code: str
    message: str


class BondErrorResponse(BaseModel):
    error: BondError
