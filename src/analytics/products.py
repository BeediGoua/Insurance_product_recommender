from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProductStats:
    product_cols: list[str]
    N: int
    co_counts: np.ndarray  # (P,P) supports(A,B)
    prevalence: np.ndarray  # (P,) P(A)
    lift: np.ndarray  # (P,P) lift(A,B), diag=0
    cond: np.ndarray  # (P,P) cond(A->B)=P(B|A), diag=1 par construction mais peut rester ~1


def build_product_stats(train_df: pd.DataFrame, product_cols: list[str]) -> ProductStats:
    """
    Construit :
    - co_counts(A,B)
    - prevalence P(A)
    - lift(A,B) = P(A,B)/(P(A)P(B))
    - cond(A->B) = P(B|A)
    """
    X = train_df[product_cols].astype(int).to_numpy()
    N = int(X.shape[0])
    P = int(X.shape[1])

    co_counts = X.T @ X  # supports
    co_prob = co_counts / max(1, N)

    prev = train_df[product_cols].mean().to_numpy()  # aligné à product_cols
    p_outer = np.outer(prev, prev)
    lift = np.divide(co_prob, p_outer, out=np.zeros_like(co_prob), where=(p_outer > 0))
    np.fill_diagonal(lift, 0.0)

    support_A = np.diag(co_counts).astype(float)
    cond = np.divide(
        co_counts,
        support_A.reshape(-1, 1),
        out=np.zeros_like(co_prob),
        where=(support_A.reshape(-1, 1) > 0),
    )

    return ProductStats(
        product_cols=product_cols,
        N=N,
        co_counts=co_counts,
        prevalence=prev,
        lift=lift,
        cond=cond,
    )


def pair_table(
    stats: ProductStats,
    *,
    min_support: int,
    min_expected: float | None = None,
) -> pd.DataFrame:
    """Table A-B (i<j) avec support, lift, P(B|A), P(A|B), attendu_indep."""
    P = len(stats.product_cols)
    rows = []
    for i in range(P):
        for j in range(i + 1, P):
            sup_ab = int(stats.co_counts[i, j])
            if sup_ab < min_support:
                continue

            expected = float(stats.N * stats.prevalence[i] * stats.prevalence[j])
            if min_expected is not None and expected < min_expected:
                continue

            sup_a = int(stats.co_counts[i, i])
            sup_b = int(stats.co_counts[j, j])

            rows.append(
                {
                    "A": stats.product_cols[i],
                    "B": stats.product_cols[j],
                    "support(A)": sup_a,
                    "support(B)": sup_b,
                    "support(A,B)": sup_ab,
                    "expected_indep": expected,
                    "prevA": float(stats.prevalence[i]),
                    "prevB": float(stats.prevalence[j]),
                    "lift": float(stats.lift[i, j]),
                    "P(B|A)": float(sup_ab / sup_a) if sup_a > 0 else 0.0,
                    "P(A|B)": float(sup_ab / sup_b) if sup_b > 0 else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values(["lift", "support(A,B)"], ascending=False)


def support_threshold_quantile(stats: ProductStats, q: float = 0.75, floor: int = 20) -> int:
    """
    Seuil 'data-driven' : quantile q des supports hors diagonale (non nuls),
    avec un plancher floor pour éviter trop petit sur dataset petit.
    """
    P = len(stats.product_cols)
    off = stats.co_counts[np.triu_indices(P, k=1)]
    off_nz = off[off > 0]
    if off_nz.size == 0:
        return floor
    return int(max(floor, np.quantile(off_nz, q)))
