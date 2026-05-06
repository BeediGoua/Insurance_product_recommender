"""Combine BM25 and vector retrieval scores with graceful fallback behavior."""

from __future__ import annotations

from typing import List, Dict, Any, Optional

import numpy as np

from .product_catalog import get_product_catalog
from .product_retriever import bm25_search, _tokenize


# ---------------------------------------------------------------------------
# Vector index (lazy, built once per process)
# ---------------------------------------------------------------------------

_vector_index_cache: Optional[tuple] = None  # (df, model, faiss_index)


def _build_vector_index(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Build a FAISS inner-product index over product descriptions."""
    global _vector_index_cache
    if _vector_index_cache is not None:
        return _vector_index_cache

    from sentence_transformers import SentenceTransformer
    import faiss

    df = get_product_catalog()
    model = SentenceTransformer(model_name)

    texts = []
    for _, row in df.iterrows():
        text = " ".join([
            str(row.get("product_name", "")),
            str(row.get("description", "")),
            str(row.get("target_needs", "")),
            str(row.get("typical_customer", "")),
            str(row.get("business_notes", "")),
        ])
        texts.append(text)

    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    _vector_index_cache = (df, model, index)
    return _vector_index_cache


def vector_search(query: str, topk: int = 5) -> List[Dict[str, Any]]:
    """Dense vector search using FAISS.

    Returns an empty list when sentence-transformers or faiss-cpu are
    not installed rather than raising.
    """
    try:
        df, model, index = _build_vector_index()
    except ImportError:
        return []

    q = model.encode([query], normalize_embeddings=True)
    q = np.array(q, dtype="float32")
    scores, indices = index.search(q, topk)

    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        row = df.iloc[idx].to_dict()
        results.append({
            "product_code": row["product_code"],
            "score_vector": float(score),
            "product_info": row,
        })
    return results


def hybrid_search_products(query: str, topk: int = 5, alpha: float = 0.5) -> List[Dict[str, Any]]:
    """Combine BM25 and vector scores with linear interpolation.

    Parameters
    ----------
    query: str
        Free text query.
    topk: int
        Number of products to return.
    alpha: float
        Weight for BM25 score (1-alpha applied to vector score).

    Returns
    -------
    List[Dict]
        Each dict contains ``product_code``, ``score``, ``score_bm25``,
        ``score_vector`` and ``product_info``, sorted by descending ``score``.
    """
    bm25_results = bm25_search(query, topk=topk)
    vector_results = vector_search(query, topk=topk)

    scores: Dict[str, Dict] = {}
    for r in bm25_results:
        code = r["product_code"]
        scores.setdefault(code, {"bm25": 0.0, "vector": 0.0, "info": r["product_info"]})
        scores[code]["bm25"] = r["score_bm25"]
    for r in vector_results:
        code = r["product_code"]
        scores.setdefault(code, {"bm25": 0.0, "vector": 0.0, "info": r["product_info"]})
        scores[code]["vector"] = r["score_vector"]

    max_bm25 = max((v["bm25"] for v in scores.values()), default=1.0) or 1.0
    max_vector = max((v["vector"] for v in scores.values()), default=1.0) or 1.0

    final: List[Dict[str, Any]] = []
    for code, v in scores.items():
        bm25_norm = v["bm25"] / max_bm25
        vector_norm = v["vector"] / max_vector
        final_score = alpha * bm25_norm + (1 - alpha) * vector_norm
        final.append({
            "product_code": code,
            "score": final_score,
            "score_bm25": v["bm25"],
            "score_vector": v["vector"],
            "product_info": v["info"],
        })

    final.sort(key=lambda x: x["score"], reverse=True)
    return final[:topk]

