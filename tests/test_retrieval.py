import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.retrieval.product_retriever import search_products


def test_search_products():
    results = search_products("family protection", topk=2)
    # Should return at least one result
    assert len(results) >= 1
    # First result should include Life insurance in sample catalog
    codes = [code for code, _ in results]
    assert "Life" in codes