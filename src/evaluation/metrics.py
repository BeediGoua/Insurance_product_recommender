from __future__ import annotations

import numpy as np


def hit_at_k(ranked_idx: np.ndarray, y_true: np.ndarray, k: int) -> float:
    """Hit@k = % où y_true est dans le top-k."""
    return float(np.mean([y_true[i] in ranked_idx[i, :k] for i in range(len(y_true))]))


def mrr(ranked_idx: np.ndarray, y_true: np.ndarray) -> float:
    """Mean Reciprocal Rank."""
    rr = []
    for i in range(len(y_true)):
        pos = np.where(ranked_idx[i] == y_true[i])[0]
        rr.append(1.0 / (pos[0] + 1) if pos.size else 0.0)
    return float(np.mean(rr))
