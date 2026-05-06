"""
Simple keyword based product retrieval.

This module implements a naive bag‑of‑words search over the product
catalog.  It is designed to illustrate how retrieval can be used to
provide context to explanations, but it is not intended for
production use.  For better performance and relevance consider using
BM25 (via ``rank_bm25``) or vector embeddings (via sentence
transformers and FAISS or Chroma).
"""

from __future__ import annotations

from typing import List, Tuple
import re

from .product_catalog import get_product_catalog


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\w+", text.lower()) if text else []


def search_products(query: str, topk: int = 3) -> List[Tuple[str, float]]:
    """Search the product catalog using a simple token overlap metric.

    Parameters
    ----------
    query: str
        The free text query, e.g. "family protection".
    topk: int
        Number of results to return.

    Returns
    -------
    List[Tuple[str, float]]
        A list of tuples ``(product_code, score)`` sorted by descending
        score.  The score is the fraction of query tokens appearing in
        the product name or description.
    """
    catalog = get_product_catalog()
    if catalog.empty or not query:
        return []
    q_tokens = _tokenize(query)
    results: List[Tuple[str, float]] = []
    for _, row in catalog.iterrows():
        text = f"{row.get('product_name', '')} {row.get('description', '')}"
        doc_tokens = _tokenize(text)
        if not doc_tokens:
            score = 0.0
        else:
            matches = sum(1 for t in q_tokens if t in doc_tokens)
            score = matches / len(q_tokens)
        if score > 0:
            results.append((row["product_code"], score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:topk]
