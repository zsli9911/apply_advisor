"""Hybrid retrieval scoring: vector + keyword + metadata + authority.

retrieval_score = 0.45·vector + 0.30·keyword + 0.15·metadata + 0.10·authority
Official sources outrank third-party ones via source_authority.
"""
from __future__ import annotations

import re
from typing import Any

_W_VEC, _W_KW, _W_META, _W_AUTH = 0.45, 0.30, 0.15, 0.10


def _tokens(text: Any) -> set[str]:
    return {t for t in re.findall(r"\w+", str(text or "").lower()) if len(t) > 2}


def keyword_score(query: str, text: str) -> float:
    q, d = _tokens(query), _tokens(text)
    return len(q & d) / len(q) if q else 0.0


def metadata_match(query: str, row: dict[str, Any]) -> float:
    q = _tokens(query)
    if not q:
        return 0.0
    meta = _tokens(" ".join(str(row.get(k) or "") for k in ("country", "document_type", "chunk_type")))
    meta |= _tokens(" ".join(row.get("heading_path") or []))
    return len(q & meta) / len(q)


def source_authority(row: dict[str, Any]) -> float:
    official = row.get("official")
    return 1.0 if official is True else 0.4 if official is False else 0.6


def combine(query: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for r in rows:
        vec = max(0.0, float(r.get("similarity") or 0.0))
        kw = keyword_score(query, r.get("content"))
        meta = metadata_match(query, r)
        auth = source_authority(r)
        r["keyword_score"] = round(kw, 3)
        r["metadata_match"] = round(meta, 3)
        r["source_authority"] = auth
        r["retrieval_score"] = round(_W_VEC * vec + _W_KW * kw + _W_META * meta + _W_AUTH * auth, 3)
    return sorted(rows, key=lambda r: -r["retrieval_score"])
