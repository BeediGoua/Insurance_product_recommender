from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .metrics import hit_at_k, mrr


@dataclass(frozen=True)
class EvalReport:
    metrics: dict[str, float]
    by_basket_size: pd.DataFrame
    by_masked_product: pd.DataFrame


def evaluate_ranking(
    *,
    scores_mat: np.ndarray,       # (n_eval, P)
    y_true: np.ndarray,           # (n_eval,)
    product_cols: list[str],
    basket_size_obs: np.ndarray | None = None,   # (n_eval,)
    k_list: list[int] = [1, 3, 5, 10],
    min_group_n: int = 200,
    min_prod_n: int = 100,
    prevalence: np.ndarray | None = None,        # (P,)
) -> EvalReport:
    ranked = np.argsort(-scores_mat, axis=1)
    metrics = {f"Hit@{k}": hit_at_k(ranked, y_true, k) for k in k_list}
    metrics["MRR"] = mrr(ranked, y_true)

    # breakdown basket size
    by_bs_rows = []
    if basket_size_obs is not None:
        for bs in sorted(np.unique(basket_size_obs)):
            sel = np.where(basket_size_obs == bs)[0]
            if sel.size < min_group_n:
                continue
            by_bs_rows.append(
                {
                    "basket_size_observe": int(bs),
                    "n": int(sel.size),
                    "Hit@1": hit_at_k(ranked[sel], y_true[sel], 1),
                    "Hit@5": hit_at_k(ranked[sel], y_true[sel], 5),
                    "MRR": mrr(ranked[sel], y_true[sel]),
                }
            )
    by_basket_size = pd.DataFrame(by_bs_rows).sort_values("basket_size_observe") if by_bs_rows else pd.DataFrame()

    # breakdown by product masked
    by_prod_rows = []
    for j, prod in enumerate(product_cols):
        sel = np.where(y_true == j)[0]
        if sel.size < min_prod_n:
            continue
        by_prod_rows.append(
            {
                "masked_product": prod,
                "n": int(sel.size),
                "prev_train": float(prevalence[j]) if prevalence is not None else np.nan,
                "Hit@1": hit_at_k(ranked[sel], y_true[sel], 1),
                "Hit@5": hit_at_k(ranked[sel], y_true[sel], 5),
                "MRR": mrr(ranked[sel], y_true[sel]),
            }
        )
    by_masked_product = pd.DataFrame(by_prod_rows).sort_values("Hit@5", ascending=False) if by_prod_rows else pd.DataFrame()

    return EvalReport(metrics=metrics, by_basket_size=by_basket_size, by_masked_product=by_masked_product)
