"""Rerank retrieval candidates with lightweight recommendation-aware heuristics."""

from __future__ import annotations

from typing import List, Dict, Any, Optional


def rerank_results(
    query: str,
    candidates: List[Dict[str, Any]],
    recommended_products: Optional[List[str]] = None,
    topk: int = 5,
) -> List[Dict[str, Any]]:
    """Rerank hybrid search candidates using lightweight heuristics.

    Parameters
    ----------
    query: str
        Original query string.
    candidates: List[Dict]
        Output of :func:`hybrid_search_products`.
    recommended_products: Optional[List[str]]
        Product codes returned by the ML recommender.  Products in this
        list receive a score boost of +0.2.
    topk: int
        Number of results to return.

    Returns
    -------
    List[Dict]
        Candidates sorted by ``rerank_score``, limited to ``topk``.
    """
    recommended_products = recommended_products or []
    query_lower = query.lower()

    reranked = []
    for c in candidates:
        score = c.get("score", 0.0)
        info = c.get("product_info", {})

        # Boost if the ML recommender also suggested this product
        if c["product_code"] in recommended_products:
            score += 0.2

        # Boost if the product's target_needs overlaps with the query
        target_needs = str(info.get("target_needs", "")).lower()
        if target_needs and any(word in query_lower for word in target_needs.split(",")):
            score += 0.1

        c = dict(c)
        c["rerank_score"] = score
        reranked.append(c)

    reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
    return reranked[:topk]

