import numpy as np
import pandas as pd

def evaluate_masking(
    X_full: np.ndarray,
    score_fn,                       # score_fn(X_row_obs, idx) -> scores (P,)
    k_list=(1,3,5,10),
    hide_k=1,                       # int, si hide_pct est None
    hide_pct=None,                  # ex: 0.1, 0.3, 0.5
    seed=42,
    min_observed=1                  # garder au moins 1 produit observé
):
    rng = np.random.default_rng(seed)
    N, P = X_full.shape
    basket = X_full.sum(axis=1)

    # déterminer k à cacher pour chaque client
    if hide_pct is None:
        k_hide = np.full(N, hide_k, dtype=int)
    else:
        k_hide = np.maximum(1, np.floor(hide_pct * basket).astype(int))

    # clients éligibles : avoir assez de produits pour cacher k et garder min_observed
    eligible = np.where(basket >= (k_hide + min_observed))[0]

    X_obs = X_full[eligible].copy()
    hidden_sets = []
    
    # Pre-calculate masking to ensure repro
    for t, i in enumerate(eligible):
        owned = np.where(X_full[i] == 1)[0]
        k = k_hide[i]
        hidden = rng.choice(owned, size=k, replace=False)
        X_obs[t, hidden] = 0
        hidden_sets.append(set(hidden.tolist()))

    # scoring loop
    # We pass 'i' (original index in X_full) to score_fn
    # score_fn signature update: (x_vector, original_index)
    scores_list = []
    for t in range(len(eligible)):
        idx_original = eligible[t]
        x_row = X_obs[t]
        
        # Try-catch to support both signatures (backward compatibility)
        try:
             s = score_fn(x_row, idx_original)
        except TypeError:
             # Fallback for old score_fn that only takes x_row
             s = score_fn(x_row)
        scores_list.append(s)
        
    scores_mat = np.vstack(scores_list)
    ranked = np.argsort(-scores_mat, axis=1)

    # métriques
    def hit_at_k_single(ranked_row, hidden_set, k):
        # version LOO (k=1 caché) -> bool ; version multi -> au moins 1 retrouvé
        return int(any(item in ranked_row[:k] for item in hidden_set))

    def recall_at_k(ranked_row, hidden_set, k):
        topk = set(ranked_row[:k].tolist())
        return len(hidden_set & topk) / max(1, len(hidden_set))

    metrics = {}
    for k in k_list:
        if all(len(s)==1 for s in hidden_sets):
            metrics[f"Hit@{k}"] = float(np.mean([hit_at_k_single(ranked[i], hidden_sets[i], k) for i in range(len(hidden_sets))]))
        metrics[f"Recall@{k}"] = float(np.mean([recall_at_k(ranked[i], hidden_sets[i], k) for i in range(len(hidden_sets))]))

    # résumé
    info = {
        "n_eval": int(len(eligible)),
        "avg_basket": float(basket[eligible].mean()),
        "avg_hidden": float(np.mean([len(s) for s in hidden_sets])),
        "mode": "hide_pct" if hide_pct is not None else "hide_k",
        "hide_k": int(hide_k),
        "hide_pct": float(hide_pct) if hide_pct is not None else None
    }

    return pd.Series(info), pd.Series(metrics)
