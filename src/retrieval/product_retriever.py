"""Retrieve products with BM25 scoring and a keyword-overlap fallback."""

from __future__ import annotations

from typing import List, Dict, Any
import re

from .product_catalog import get_product_catalog


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", str(text).lower()) if text else []


def _build_corpus(catalog) -> List[List[str]]:
    docs = []
    for _, row in catalog.iterrows():
        text = " ".join([
            str(row.get("product_name", "")),
            str(row.get("description", "")),
            str(row.get("target_needs", "")),
            str(row.get("typical_customer", "")),
            str(row.get("business_notes", "")),
        ])
        docs.append(_tokenize(text))
    return docs


def bm25_search(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """Search the product catalog using BM25 (rank_bm25).

    Falls back to keyword overlap scoring if rank_bm25 is not installed.

    Returns
    -------
    List[Dict]
        Each dict has ``product_code``, ``score_bm25`` and ``product_info``.
    """
    catalog = get_product_catalog()
    if catalog.empty or not query:
        return []

    corpus = _build_corpus(catalog)
    query_tokens = _tokenize(query)

    try:
        from rank_bm25 import BM25Okapi
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
    except ImportError:
        # Fallback: fraction of query tokens present in doc
        scores = []
        for doc in corpus:
            matches = sum(1 for t in query_tokens if t in doc)
            scores.append(matches / max(len(query_tokens), 1))

    import numpy as np
    scores = list(scores)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:topk]

    results: List[Dict[str, Any]] = []
    for idx in ranked_idx:
        row = catalog.iloc[idx].to_dict()
        results.append({
            "product_code": row["product_code"],
            "score_bm25": float(scores[idx]),
            "product_info": row,
        })
    return results


# Keep backward-compatible alias
def search_products(query: str, topk: int = 3):
    """Backward-compatible wrapper returning (product_code, score) tuples."""
    results = bm25_search(query, topk=topk)
    return [(r["product_code"], r["score_bm25"]) for r in results]

