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
from .sources import fetch_corporate, fetch_government
from .sources._common import SecurityNotFound, SourceFetchError

router = APIRouter()

API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENROUTER_API_KEY")


def _error(status: int, code: str, message: str) -> JSONResponse:
    payload = BondErrorResponse(error={"code": code, "message": message}).model_dump()
    return JSONResponse(status_code=status, content=payload)


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

    fetcher = fetch_government if req.type == "government" else fetch_corporate

    try:
        source = await fetcher(req.security_id)
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
