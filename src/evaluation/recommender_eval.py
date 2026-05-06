"""
Simple evaluation metrics for recommendation rankings.

The functions in this module operate on lists of recommended products
and corresponding ground truth items.  They compute familiar metrics
such as Hit@K, Mean Reciprocal Rank (MRR) and coverage.
"""

from __future__ import annotations

from typing import List, Dict, Any


def evaluate_recommender(
    predictions: List[List[str]],
    truths: List[str],
    ks: List[int] = [1, 3, 5],
) -> Dict[str, Any]:
    """Compute Hit@K and MRR for a set of predictions.

    Parameters
    ----------
    predictions: List[List[str]]
        A list where each element is an ordered list of product codes
        recommended for a single client.
    truths: List[str]
        The true held‑out product for each client.
    ks: List[int]
        Values of k at which to compute Hit@K.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing metrics and per‑k values.
    """
    assert len(predictions) == len(truths), "Predictions and truths must be of same length"
    n = len(predictions)
    hit_at_k = {k: 0 for k in ks}
    mrr_total = 0.0
    for recs, truth in zip(predictions, truths):
        # Hit@K
        for k in ks:
            if truth in recs[:k]:
                hit_at_k[k] += 1
        # MRR
        if truth in recs:
            rank = recs.index(truth) + 1
            mrr_total += 1.0 / rank
    metrics: Dict[str, Any] = {}
    for k in ks:
        metrics[f"Hit@{k}"] = hit_at_k[k] / n
    metrics["MRR"] = mrr_total / n if n > 0 else 0.0
    return metrics
