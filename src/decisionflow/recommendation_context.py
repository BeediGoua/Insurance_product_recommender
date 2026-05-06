"""
Wrapper around the existing recommendation engine.

This module exposes a simple API for generating recommendations from a
``ClientProfile`` without having to know about CatBoost, the baseline
model or the various helper functions in ``src.inference``.  It
produces a :class:`RecommendationResult` with raw scores, filtered
scores and a top‑k list.  Business rules (already owned products) are
applied here in a deterministic manner.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Optional

import pandas as pd

from .schemas import ClientProfile, RecommendationResult

# We reuse the existing inference logic.  Note: import inside functions to
# avoid heavy modules loading at import time.


def load_existing_model():
    """Load the hybrid recommendation model (CatBoost + Baseline).

    Returns a tuple of ``(hybrid_model, alpha)`` or ``None`` if
    loading fails.  It relies on ``src.inference.load_hybrid_model``.
    """
    from src.inference import load_hybrid_model

    try:
        return load_hybrid_model()
    except Exception:
        return None


def run_statistical_baseline(profile: ClientProfile, topk: int = 5) -> RecommendationResult:
    """Generate recommendations using the pure statistical baseline.

    The baseline is deterministic and does not require any user features
    beyond the list of currently owned products.  If the baseline
    artifact cannot be loaded an empty result is returned.
    """
    from src.inference import load_baseline
    from src.pipelines.baseline_pipeline import recommend_from_selection

    artifact = load_baseline()
    if not artifact:
        return RecommendationResult(
            client_id=profile.client_id,
            raw_scores={},
            filtered_scores={},
            top_k=[],
        )

    # Compute baseline scores as a pandas Series
    s = recommend_from_selection(artifact, profile.current_products, topk=topk)
    raw_scores = s.to_dict()
    filtered_scores = raw_scores.copy()
    top_k = list(s.index[:topk])
    return RecommendationResult(
        client_id=profile.client_id,
        raw_scores=raw_scores,
        filtered_scores=filtered_scores,
        top_k=top_k,
    )


def run_catboost_model(profile: ClientProfile, topk: int = 5, alpha_override: Optional[float] = None) -> RecommendationResult:
    """Generate recommendations using the hybrid CatBoost + baseline model.

    This function prepares the input for the model, calls the hybrid
    predictor and then applies the anti‑cheat filter (remove owned
    products).  If the model cannot be loaded an empty result is
    returned.
    """
    from src.inference import get_recommendations

    # Build context dict for inference.get_recommendations
    context = {
        "user_features": profile.extra_info,
        "owned_products": profile.current_products,
    }
    # Use the CatBoost strategy string; get_recommendations handles loading
    s = get_recommendations("CatBoost", context, topk=topk, alpha_override=alpha_override)
    # When inference fails it returns an empty Series
    if s is None or len(s) == 0:
        return RecommendationResult(
            client_id=profile.client_id,
            raw_scores={},
            filtered_scores={},
            top_k=[],
        )
    # s is a pandas Series mapping product->score
    raw_scores = s.to_dict()
    # After get_recommendations the owned products are already removed.  We
    # still copy into filtered_scores for completeness.
    filtered_scores = raw_scores.copy()
    top_k = list(s.index[:topk])
    return RecommendationResult(
        client_id=profile.client_id,
        raw_scores=raw_scores,
        filtered_scores=filtered_scores,
        top_k=top_k,
    )


def combine_scores(catboost_scores: Dict[str, float], baseline_scores: Dict[str, float], alpha: float = 0.5) -> Dict[str, float]:
    """Combine CatBoost and baseline scores using a weighted average.

    This helper is provided for completeness but is not used directly
    because the hybrid model in ``src.models.catboost.predictor`` already
    performs the fusion.  It remains here for potential future use.
    """
    combined: Dict[str, float] = {}
    for product, cb_score in catboost_scores.items():
        bl_score = baseline_scores.get(product, 0.0)
        # Weighted geometric mean is equivalent to cb_score * (bl_score**alpha)
        combined[product] = cb_score * (bl_score ** alpha)
    return combined


def apply_already_owned_filter(scores: Dict[str, float], owned: List[str]) -> Dict[str, float]:
    """Filter out products already owned by setting their score to -inf.

    Parameters
    ----------
    scores: Dict[str, float]
        Mapping of product codes to scores.
    owned: List[str]
        List of product codes that the client already owns.

    Returns
    -------
    Dict[str, float]
        New dictionary with scores for owned products replaced by
        ``float('-inf')``.  Unowned products retain their scores.
    """
    filtered = {}
    for product, score in scores.items():
        if product in owned:
            filtered[product] = float("-inf")
        else:
            filtered[product] = score
    return filtered


def get_top_k_recommendations(scores: Dict[str, float], k: int = 5) -> List[str]:
    """Select the top‑k products from a score mapping.

    Products with ``-inf`` score are automatically excluded.  Ties are
    broken arbitrarily by the underlying sort.
    """
    # Filter out -inf values and sort descending
    sorted_products = sorted([
        (p, s) for p, s in scores.items() if s != float("-inf")
    ], key=lambda x: x[1], reverse=True)
    return [p for p, _ in sorted_products[:k]]


def run_recommender(profile: ClientProfile, topk: int = 5, use_hybrid: bool = True):
    """Convenience wrapper that dispatches to the hybrid or baseline recommender.

    Parameters
    ----------
    profile: ClientProfile
        Client profile for which to generate recommendations.
    topk: int
        Number of recommendations to return.
    use_hybrid: bool
        If True use the CatBoost+baseline hybrid model when available,
        otherwise fall back to the pure baseline.

    Returns
    -------
    RecommendationResult
        Structured recommendation result.
    """
    if use_hybrid:
        rec = run_catboost_model(profile, topk=topk)
        if not rec.top_k:
            rec = run_statistical_baseline(profile, topk=topk)
    else:
        rec = run_statistical_baseline(profile, topk=topk)
    return rec
