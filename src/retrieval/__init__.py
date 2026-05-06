"""
Retrieval utilities for providing product context.

These modules implement lightweight retrieval‑augmented generation
capabilities.  They are used by the explanation agent to enrich
recommendations with additional information such as product
descriptions and target customer segments.  The retrieval layer is
optional: if the product knowledge files are missing the functions
return empty results.
"""

from .product_catalog import get_product_catalog, get_product_info  # noqa: F401
from .product_retriever import search_products  # noqa: F401
from .hybrid_search import hybrid_search_products  # noqa: F401
from .reranker import rerank_results  # noqa: F401
