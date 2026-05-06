from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp


def ks_report(train_s: pd.Series, test_s: pd.Series) -> dict:
    t = train_s.dropna()
    u = test_s.dropna()
    res = ks_2samp(t, u)
    return {"ks_stat": float(res.statistic), "p_value": float(res.pvalue)}


def chi2_train_test(train_s: pd.Series, test_s: pd.Series) -> dict:
    tr = train_s.astype(str)
    te = test_s.astype(str)
    cats = sorted(set(tr.unique()) | set(te.unique()))
    tr_counts = tr.value_counts().reindex(cats, fill_value=0)
    te_counts = te.value_counts().reindex(cats, fill_value=0)
    table = np.vstack([tr_counts.values, te_counts.values])
    chi2, p, dof, exp = chi2_contingency(table)
    return {"chi2": float(chi2), "p_value": float(p), "n_categories": len(cats)}
