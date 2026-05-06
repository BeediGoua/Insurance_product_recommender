from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConditionalBaseline:
    """
    Baseline 'forward' : score(B) = somme_A w(A) * P(B|A), puis masque des produits déjà détenus.
    """
    product_cols: list[str]
    cond: np.ndarray  # (P,P) cond(A->B)=P(B|A)
    support_A: np.ndarray  # (P,) support(A) = diag(co_counts)
    product_to_idx: dict[str, int]

    @classmethod
    def from_stats(cls, product_cols: list[str], cond: np.ndarray, support_A: np.ndarray) -> "ConditionalBaseline":
        return cls(
            product_cols=product_cols,
            cond=cond,
            support_A=support_A.astype(float),
            product_to_idx={p: i for i, p in enumerate(product_cols)},
        )

    def score_one(self, x_obs: np.ndarray) -> np.ndarray:
        """
        x_obs: (P,) binaire observé (après masquage), 1=possédé, 0=absent/masqué.
        Retour: scores (P,), -inf sur produits déjà possédés.
        """
        owned_idx = np.where(x_obs == 1)[0]
        P = len(self.product_cols)
        scores = np.zeros(P, dtype=float)
        if owned_idx.size:
            w = self.support_A[owned_idx].astype(float)
            w = w / w.sum() if w.sum() > 0 else np.ones_like(w) / owned_idx.size
            # IMPORTANT : forward P(B|A) => cond[owned, :] puis somme sur A => axis=0
            scores = (self.cond[owned_idx, :] * w.reshape(-1, 1)).sum(axis=0)

        scores[owned_idx] = -np.inf
        return scores

    def score_dataframe_row(self, row_products: pd.Series) -> pd.Series:
        x = row_products[self.product_cols].astype(int).to_numpy()
        scores = self.score_one(x)
        return pd.Series(scores, index=self.product_cols)
