"""
Intelligence-layer enrichment: sector/theme/taxonomy classification for PDS
corporate securities.

v1 is fully deterministic — no LLM, no network:
  1. Issuer ticker stem -> sector / sub-sector (static map of PDS issuers).
  2. Issuer ESG bond PROGRAMMES we can name -> theme (marked "indicative";
     issuer runs a labeled programme, but per-series verification is pending).
  3. Theme -> per-framework alignment verdict via conservative rules.

Anything we cannot classify stays "unknown" — verdicts are never fabricated.
Every payload carries `confidence` + `basis` so the UI can label honestly.
"""

from .schema import BondIntelligence, TaxonomyAlignment

# ---------------------------------------------------------------------------
# 1. Issuer ticker stem -> (sector, sub_sector)
#    Stems match the leading letters of PDS Local IDs (e.g. "ALI 25-R24" -> ALI).
# ---------------------------------------------------------------------------
SECTOR_MAP: dict[str, tuple[str, str]] = {
    "AC":    ("Holding Firms", "Conglomerate"),
    "ACEN":  ("Energy", "Renewable Power"),
    "ALCO":  ("Property", "Sustainable Real Estate"),
    "CREIT": ("Property", "Renewable-Energy REIT"),
    "AEV":   ("Holding Firms", "Conglomerate"),
    "ALI":   ("Property", "Real Estate Development"),
    "AP":    ("Energy", "Power Generation"),
    "BDO":   ("Financials", "Banking"),
    "BPI":   ("Financials", "Banking"),
    "CBC":   ("Financials", "Banking"),
    "CLI":   ("Property", "Real Estate Development"),
    "DDMPR": ("Property", "REIT"),
    "DD":    ("Property", "Real Estate Development"),
    "EDC":   ("Energy", "Geothermal / Renewable Power"),
    "EW":    ("Financials", "Banking"),
    "FDC":   ("Holding Firms", "Conglomerate"),
    "FLI":   ("Property", "Real Estate Development"),
    "GLO":   ("Telecommunications", "Mobile & Broadband"),
    "GTCAP": ("Holding Firms", "Conglomerate"),
    "JGS":   ("Holding Firms", "Conglomerate"),
    "MBT":   ("Financials", "Banking"),
    "MER":   ("Utilities", "Power Distribution"),
    "NLEX":  ("Industrials", "Toll Roads / Infrastructure"),
    "PCOR":  ("Energy", "Oil Refining & Marketing"),
    "PNB":   ("Financials", "Banking"),
    "PSB":   ("Financials", "Banking"),
    "RCB":   ("Financials", "Banking"),
    "RLC":   ("Property", "Real Estate Development"),
    "SECB":  ("Financials", "Banking"),
    "SMC":   ("Holding Firms", "Conglomerate"),
    "SMCGP": ("Energy", "Power Generation"),
    "SMIC":  ("Holding Firms", "Conglomerate"),
    "SMPH":  ("Property", "Real Estate Development"),
    "TEL":   ("Telecommunications", "Mobile & Broadband"),
    "UBP":   ("Financials", "Banking"),
    "VLL":   ("Property", "Real Estate Development"),
}

# Green is NOT determined by issuer heuristics — it comes from the PDS Listed
# Securities Database (ISSUE column labeled "Green Bonds"). See green_registry.py.
# classify() receives the authoritative green-id set + optional PDS issue text.

# ---------------------------------------------------------------------------
# 3. Theme -> conservative per-framework alignment verdicts.
#    "partial" where a labeled programme plausibly maps but criteria differ;
#    conventional bonds are explicitly not_aligned; unknown stays unknown.
# ---------------------------------------------------------------------------
_THEME_TAXONOMY: dict[str, TaxonomyAlignment] = {
    "green": TaxonomyAlignment(
        eu="partial", cbi_mitigation="partial", cbi_resilience="unknown",
        ph_sftg="aligned", asean_gss="aligned",
    ),
    "sustainability": TaxonomyAlignment(
        eu="partial", cbi_mitigation="partial", cbi_resilience="unknown",
        ph_sftg="aligned", asean_gss="aligned",
    ),
    "social": TaxonomyAlignment(
        eu="not_aligned", cbi_mitigation="not_aligned", cbi_resilience="unknown",
        ph_sftg="partial", asean_gss="aligned",
    ),
    "conventional": TaxonomyAlignment(
        eu="not_aligned", cbi_mitigation="not_aligned", cbi_resilience="not_aligned",
        ph_sftg="not_aligned", asean_gss="not_aligned",
    ),
}


def _ticker_stem(local_id: str | None, issuer: str | None) -> str | None:
    """Best-effort ticker stem from a PDS Local ID (fallback: issuer field)."""
    for candidate in (local_id, issuer):
        if not candidate:
            continue
        stem = candidate.split()[0].strip().upper()
        # Longest-prefix match so SMCGP wins over SMC, ACEN over AC.
        hits = [t for t in SECTOR_MAP if stem == t or stem.startswith(t)]
        if hits:
            return max(hits, key=len)
    return None


def classify(
    local_id: str | None,
    issuer: str | None,
    green_ids: set[str] | None = None,
    green_issue: str | None = None,
) -> BondIntelligence:
    """Classify one security. `green_ids` = authoritative set of normalized PDS
    Local IDs labeled green (from green_registry); `green_issue` = the PDS ISSUE
    text for this security if green. Green is set ONLY when the security's own
    Local ID is in that PDS-sourced set — no issuer heuristics."""
    from .sources._common import normalize_id

    ticker = _ticker_stem(local_id, issuer)
    sector, sub_sector = SECTOR_MAP.get(ticker, (None, None)) if ticker else (None, None)

    is_green = bool(green_ids and local_id and normalize_id(local_id) in green_ids)

    if is_green:
        return BondIntelligence(
            sector=sector,
            sub_sector=sub_sector,
            theme="green",
            taxonomies=_THEME_TAXONOMY["green"].model_copy(),
            confidence="verified",
            basis=f"PDS-labeled green issue: {green_issue}" if green_issue
                  else "Labeled green in the PDS Listed Securities Database.",
        )

    return BondIntelligence(
        sector=sector,
        sub_sector=sub_sector,
        theme="conventional" if sector else "unknown",
        taxonomies=(_THEME_TAXONOMY["conventional"].model_copy() if sector else TaxonomyAlignment()),
        confidence="verified" if sector else "unknown",
        basis="Not labeled green in the PDS Listed Securities Database."
              if sector else "Issuer not in classification map.",
    )
