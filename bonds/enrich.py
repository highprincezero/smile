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

# ---------------------------------------------------------------------------
# 2. Issuers with publicly announced labeled-bond PROGRAMMES.
#    Theme is applied issuer-wide as "indicative" — per-series verification
#    (framework docs / SPOs) is a later, higher-confidence pass.
# ---------------------------------------------------------------------------
THEME_MAP: dict[str, tuple[str, str]] = {
    # ticker: (theme, basis)
    "ACEN": ("green", "ACEN issues under a green finance framework (renewables capex)."),
    "EDC":  ("green", "EDC geothermal issuances sit under its green bond framework."),
    "BPI":  ("sustainability", "BPI runs a sustainable funding framework (ASEAN-labeled issuances)."),
    "RCB":  ("sustainability", "RCBC has issued ASEAN sustainability bonds under its framework."),
    "BDO":  ("sustainability", "BDO has issued ASEAN sustainability bonds under its framework."),
    "SMPH": ("green", "SM Prime has issued under a green finance framework."),
}

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


def classify(local_id: str | None, issuer: str | None) -> BondIntelligence:
    """Deterministic v1 classification for one security."""
    ticker = _ticker_stem(local_id, issuer)
    if ticker is None:
        return BondIntelligence(confidence="unknown", basis="Issuer not in classification map.")

    sector, sub_sector = SECTOR_MAP[ticker]
    theme, basis = THEME_MAP.get(ticker, (None, None))

    if theme:
        return BondIntelligence(
            sector=sector,
            sub_sector=sub_sector,
            theme=theme,  # type: ignore[arg-type]
            taxonomies=_THEME_TAXONOMY[theme].model_copy(),
            confidence="indicative",
            basis=f"{basis} Issuer-level rule; per-series verification pending.",
        )

    return BondIntelligence(
        sector=sector,
        sub_sector=sub_sector,
        theme="unknown",
        taxonomies=TaxonomyAlignment(),
        confidence="indicative",
        basis="Sector from issuer ticker; no labeled-bond programme on record for this issuer.",
    )
