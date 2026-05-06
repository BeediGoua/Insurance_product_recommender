"""Tests for hybrid retrieval pipeline."""
import pytest


def test_bm25_search_returns_results():
    """BM25 search should return results when the catalog is non-empty."""
    from src.retrieval.product_retriever import bm25_search
    from src.retrieval.product_catalog import get_product_catalog

    if get_product_catalog().empty:
        pytest.skip("Product catalog not available")

    results = bm25_search("family protection", topk=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "product_code" in results[0]
    assert "score_bm25" in results[0]


def test_bm25_search_empty_query():
    from src.retrieval.product_retriever import bm25_search
    results = bm25_search("", topk=3)
    assert results == []


def test_hybrid_search_returns_results():
    """hybrid_search_products should return scored results."""
    from src.retrieval.hybrid_search import hybrid_search_products
    from src.retrieval.product_catalog import get_product_catalog

    if get_product_catalog().empty:
        pytest.skip("Product catalog not available")

    results = hybrid_search_products("family protection", topk=3)
    assert isinstance(results, list)
    assert len(results) > 0
    assert "score" in results[0]
    assert "product_code" in results[0]


def test_reranker_boosts_recommended():
    from src.retrieval.reranker import rerank_results

    candidates = [
        {"product_code": "A", "score": 0.5, "product_info": {"target_needs": "retirement"}},
        {"product_code": "B", "score": 0.6, "product_info": {"target_needs": "family protection"}},
    ]
    reranked = rerank_results("family protection", candidates, recommended_products=["A"], topk=2)
    # A gets +0.2 boost for being in recommended_products, bringing it to 0.7
    assert reranked[0]["product_code"] == "A"
    assert "rerank_score" in reranked[0]


def test_reranker_target_needs_boost():
    from src.retrieval.reranker import rerank_results

    candidates = [
        {"product_code": "X", "score": 0.4, "product_info": {"target_needs": "education"}},
        {"product_code": "Y", "score": 0.4, "product_info": {"target_needs": "retirement"}},
    ]
    reranked = rerank_results("retirement planning", candidates, topk=2)
    # Y target_needs matches query → +0.1
    assert reranked[0]["product_code"] == "Y"
