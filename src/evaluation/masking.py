from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class MaskedEvalSet:
    X_obs: np.ndarray   # (n_eval, P) observé après masquage
    y_true: np.ndarray  # (n_eval,) index du produit masqué
    idx_src: np.ndarray # (n_eval,) indices clients dans train original


def make_masked_eval_set(
    X_full: np.ndarray,
    *,
    rng: np.random.Generator,
    min_basket_size: int = 2,
) -> MaskedEvalSet:
    """
    Masque 1 produit à 1 par client éligible (basket_size>=min_basket_size),
    et conserve y_true = index du produit masqué.
    """
    N, P = X_full.shape
    basket_size = X_full.sum(axis=1)
    eligible = np.where(basket_size >= min_basket_size)[0]

    idx_src = eligible
    X_obs = X_full[idx_src].copy()
    y_true = np.empty(len(idx_src), dtype=int)

    for t, i in enumerate(idx_src):
        owned = np.where(X_full[i] == 1)[0]
        hide = int(rng.choice(owned))
        y_true[t] = hide
        X_obs[t, hide] = 0

    return MaskedEvalSet(X_obs=X_obs, y_true=y_true, idx_src=idx_src)
