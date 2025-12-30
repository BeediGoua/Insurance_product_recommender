from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PairAuditConfig:
    min_support: int
    min_expected: float = 5.0


def pair_stats(
    *,
    A: str,
    B: str,
    product_cols: list[str],
    co_counts: np.ndarray,
    N: int,
    prevalence: np.ndarray,
) -> dict:
    product_to_idx = {p: i for i, p in enumerate(product_cols)}
    i, j = product_to_idx[A], product_to_idx[B]

    supA = int(co_counts[i, i])
    supB = int(co_counts[j, j])
    supAB = int(co_counts[min(i, j), max(i, j)]) if i != j else supA

    expected = float(N * prevalence[i] * prevalence[j]) if i != j else float(supA)
    pB_given_A = float(supAB / supA) if supA > 0 else 0.0
    pA_given_B = float(supAB / supB) if supB > 0 else 0.0

    return {
        "A": A, "B": B,
        "support(A)": supA,
        "support(B)": supB,
        "support(A,B)": supAB,
        "expected_indep": expected,
        "P(B|A)": pB_given_A,
        "P(A|B)": pA_given_B,
    }


def verdict_pair(supAB: int, expected: float, cfg: PairAuditConfig) -> Tuple[str, str]:
    """
    Retour: (verdict, why)
    - verdict: OK (solide) / fragile (...)
    - why: raison lisible
    """
    reasons = []
    if supAB < cfg.min_support:
        reasons.append(f"support<{cfg.min_support}")
    if expected < cfg.min_expected:
        reasons.append(f"attendu<{cfg.min_expected:.1f}")

    if not reasons:
        return "OK (solide)", "support et attendu suffisants"
    if len(reasons) == 2:
        return "fragile (rare + peu observé)", ", ".join(reasons)
    if "support" in reasons[0]:
        return "fragile (support faible)", reasons[0]
    return "fragile (attendu faible)", reasons[0]


def audit_topk_for_client(
    *,
    owned_products: list[str],
    topk_scores: pd.Series,  # index=product_cols, values=scores
    product_cols: list[str],
    co_counts: np.ndarray,
    N: int,
    prevalence: np.ndarray,
    cfg: PairAuditConfig,
) -> pd.DataFrame:
    rows = []
    for B, score in topk_scores.items():
        for A in owned_products:
            st = pair_stats(A=A, B=B, product_cols=product_cols, co_counts=co_counts, N=N, prevalence=prevalence)
            verdict, why = verdict_pair(st["support(A,B)"], st["expected_indep"], cfg)
            rows.append(
                {
                    "produit_observe_A": A,
                    "produit_candidat_B": B,
                    "score_baseline": float(score),
                    **st,
                    "verdict": verdict,
                    "pourquoi": why,
                }
            )
    return pd.DataFrame(rows).sort_values(["score_baseline", "produit_candidat_B"], ascending=[False, True])
