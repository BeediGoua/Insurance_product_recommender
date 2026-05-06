"""
Hybrid search combines keyword and vector based retrieval.

This is a placeholder implementation that simply forwards to the
keyword search implemented in ``product_retriever.py``.  In a full
implementation you would combine BM25 and dense vector similarity
scores and then rerank the results.
"""

from __future__ import annotations

from typing import List, Tuple

from .product_retriever import search_products


def hybrid_search_products(query: str, topk: int = 3) -> List[Tuple[str, float]]:
    """Perform hybrid search over the product catalog.

    Currently this function delegates to keyword search.  Replace or
    extend this with vector search as needed.
    """
    return search_products(query, topk=topk)
