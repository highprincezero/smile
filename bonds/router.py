"""
FastAPI router for POST /api/analyze-bond.

Orchestration:
    request → cache lookup → source adapter (PDS) → LLM normalize → cache store → response

All errors return the BondErrorResponse shape so the calling agent can react.
"""

import os

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from . import cache
from .normalize import NormalizationError, normalize
from .schema import AnalyzeBondRequest, BondErrorResponse
from .analytics import compute_stats
from .sources import fetch_corporate_analysis, fetch_government
from .sources.pds_corporate import list_securities
from .sources._common import SecurityNotFound, SourceFetchError

router = APIRouter()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _error(status: int, code: str, message: str) -> JSONResponse:
    payload = BondErrorResponse(error={"code": code, "message": message}).model_dump()
    return JSONResponse(status_code=status, content=payload)


@router.get("/api/list-bonds")
async def list_bonds(type: str = "corporate", query: str | None = None, include_all: bool = False):
    """List available securities with their real PDS Local IDs.

    Platform coverage is GREEN bonds only by default; include_all=true returns
    the full PDS corporate board (internal/analyst use)."""
    if type != "corporate":
        return {"type": type, "available": False, "securities": []}
    try:
        securities = await list_securities(query, green_only=not include_all)
    except SourceFetchError as e:
        return _error(502, "upstream_unavailable", str(e))
    return {"type": "corporate", "available": True, "count": len(securities),
            "coverage": "all" if include_all else "green", "securities": securities}


@router.get("/api/bond-stats")
async def bond_stats(issuer: str | None = None):
    """Statistical analysis (summary, correlation, variance, trend, by-issuer) over the
    corporate board, optionally filtered to one issuer/keyword."""
    try:
        return await compute_stats(issuer)
    except SourceFetchError as e:
        return _error(502, "upstream_unavailable", str(e))


@router.post("/api/analyze-bond")
async def analyze_bond(request: Request):
    body = await request.json()

    try:
        req = AnalyzeBondRequest.model_validate(body)
    except ValidationError as e:
        return _error(400, "invalid_request", e.errors().__str__())

    if not req.refresh:
        cached = cache.get(req.type, req.security_id)
        if cached is not None:
            return cached.model_dump(mode="json")

    if req.type == "corporate":
        # Corporate: deterministic parse of the PDS Board Summary PDFs (no LLM).
        try:
            result = await fetch_corporate_analysis(req.security_id)
        except SecurityNotFound as e:
            return _error(404, "not_found", str(e))
        except SourceFetchError as e:
            return _error(502, "upstream_unavailable", str(e))
    else:
        # Government: HTML snippet → LLM normalize.
        try:
            source = await fetch_government(req.security_id)
        except SecurityNotFound as e:
            return _error(404, "not_found", str(e))
        except SourceFetchError as e:
            return _error(502, "upstream_unavailable", str(e))

        try:
            result = await normalize(
                security_id=req.security_id,
                type_=req.type,
                snippet=source.snippet,
                source_url=source.source_url,
                api_key=API_KEY,
            )
        except NormalizationError as e:
            return _error(500, "normalization_failed", str(e))

    cache.put(req.type, req.security_id, result)
    return result.model_dump(mode="json")
