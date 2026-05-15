"""
TTL cache for normalized bond responses. PDS pages don't change intra-day for
a given security — caching avoids hammering the upstream and re-paying for
the LLM normalization on repeated agent tool calls.
"""

from typing import Optional
from cachetools import TTLCache

from .schema import BondAnalysis, SecurityType

# 256 entries × 10 min TTL — small footprint, plenty for a chat workload.
_cache: TTLCache = TTLCache(maxsize=256, ttl=600)


def _key(type_: SecurityType, security_id: str) -> str:
    return f"{type_}:{security_id.strip().upper()}"


def get(type_: SecurityType, security_id: str) -> Optional[BondAnalysis]:
    return _cache.get(_key(type_, security_id))


def put(type_: SecurityType, security_id: str, value: BondAnalysis) -> None:
    _cache[_key(type_, security_id)] = value


def clear() -> None:
    _cache.clear()
