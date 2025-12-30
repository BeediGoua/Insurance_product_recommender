from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.data.schema import DEFAULT_PROFILE_COLS
from src.data.io import load_train_test, infer_product_cols
from src.preprocessing.cleaning import apply_basic_cleaning
from src.analytics.products import build_product_stats
from src.baselines.conditional import ConditionalBaseline
from src.evaluation.masking import make_masked_eval_set
from src.evaluation.evaluate import evaluate_ranking


CAT_COLS = ["sex", "marital_status", "branch_code", "occupation_code", "occupation_category_code"]


@dataclass(frozen=True)
class BaselineArtifact:
    product_cols: list[str]
    cond: np.ndarray
    support_A: np.ndarray
    prevalence: np.ndarray
    N: int

    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / "baseline_conditional.npz",
            product_cols=np.array(self.product_cols, dtype=object),
            cond=self.cond.astype(np.float32),
            support_A=self.support_A.astype(np.float32),
            prevalence=self.prevalence.astype(np.float32),
            N=np.array([self.N], dtype=np.int64),
        )
        meta = {"type": "conditional_baseline", "version": 1}
        (out_dir / "baseline_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    @staticmethod
    def load(in_dir: Path) -> "BaselineArtifact":
        blob = np.load(in_dir / "baseline_conditional.npz", allow_pickle=True)
        product_cols = list(blob["product_cols"])
        cond = blob["cond"]
        support_A = blob["support_A"]
        prevalence = blob["prevalence"]
        N = int(blob["N"][0])
        return BaselineArtifact(
            product_cols=product_cols,
            cond=cond,
            support_A=support_A,
            prevalence=prevalence,
            N=N,
        )


def fit_baseline_from_csv(train_csv: Path, test_csv: Path) -> tuple[pd.DataFrame, pd.DataFrame, BaselineArtifact]:
    train, test = load_train_test(train_csv, test_csv)

    product_cols = infer_product_cols(train, DEFAULT_PROFILE_COLS)

    train_c = apply_basic_cleaning(train, cat_cols=CAT_COLS)
    test_c = apply_basic_cleaning(test, cat_cols=CAT_COLS)

    stats = build_product_stats(train_c, product_cols)
    support_A = np.diag(stats.co_counts).astype(float)

    artifact = BaselineArtifact(
        product_cols=product_cols,
        cond=stats.cond,
        support_A=support_A,
        prevalence=stats.prevalence,
        N=stats.N,
    )
    return train_c, test_c, artifact


def evaluate_baseline(train_c: pd.DataFrame, artifact: BaselineArtifact, seed: int = 42, min_basket_size: int = 2) -> dict:
    baseline = ConditionalBaseline.from_stats(artifact.product_cols, artifact.cond, artifact.support_A)

    rng = np.random.default_rng(seed)
    X_full = train_c[artifact.product_cols].astype(int).to_numpy()

    evalset = make_masked_eval_set(X_full, rng=rng, min_basket_size=min_basket_size)
    scores = np.vstack([baseline.score_one(evalset.X_obs[i]) for i in range(evalset.X_obs.shape[0])])

    # --- Standard Aggregate Report ---
    report = evaluate_ranking(
        scores_mat=scores,
        y_true=evalset.y_true,
        product_cols=artifact.product_cols,
        basket_size_obs=evalset.X_obs.sum(axis=1),
        prevalence=artifact.prevalence,
    )

    # --- Granular Report with Metadata ---
    # 1. Calculate Ranks per row
    # rank of true item = # items with score > true_score + 1
    # Efficient way: argsort
    ranked_indices = np.argsort(-scores, axis=1) # indices of products sorted by score desc
    
    ranks = []
    for i in range(len(evalset.y_true)):
        true_idx = evalset.y_true[i]
        # find where true_idx is in ranked_indices[i]
        r = np.where(ranked_indices[i] == true_idx)[0][0] + 1
        ranks.append(r)
    
    ranks = np.array(ranks)
    
    # 2. Build DataFrame
    granular = pd.DataFrame({
        "rank": ranks,
        "is_hit1": (ranks <= 1).astype(int),
        "is_hit5": (ranks <= 5).astype(int),
        "mrr": 1.0 / ranks,
        "target_product": [artifact.product_cols[i] for i in evalset.y_true],
        "basket_size": evalset.X_obs.sum(axis=1),
        "original_idx": evalset.idx_src
    })
    
    # 3. Join with Metadata from train_c
    # We need to map original_idx (integer position in X_full/train_c)
    # Be careful if train_c index is not RangeIndex.
    # We'll use iloc to be safe since X_full was built from train_c order.
    meta_cols = ["sex", "marital_status", "occupation_category_code"]
    # Check present cols
    valid_meta = [c for c in meta_cols if c in train_c.columns]
    
    meta_df = train_c.iloc[evalset.idx_src][valid_meta].reset_index(drop=True)
    granular = pd.concat([granular, meta_df], axis=1)

    return {
        "metrics": report.metrics,
        "by_basket_size": report.by_basket_size,
        "by_masked_product": report.by_masked_product,
        "granular": granular, # New !
    }


def recommend_from_selection(artifact: BaselineArtifact, owned_products: list[str], topk: int = 5) -> pd.Series:
    baseline = ConditionalBaseline.from_stats(artifact.product_cols, artifact.cond, artifact.support_A)

    x = np.zeros(len(artifact.product_cols), dtype=int)
    prod_to_idx = {p: i for i, p in enumerate(artifact.product_cols)}
    for p in owned_products:
        if p in prod_to_idx:
            x[prod_to_idx[p]] = 1

    scores = baseline.score_one(x)
    s = pd.Series(scores, index=artifact.product_cols).replace([-np.inf, np.inf], np.nan).dropna()
    return s.sort_values(ascending=False).head(topk)
