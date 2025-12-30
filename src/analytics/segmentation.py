from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def tfidf_products(X_bin: np.ndarray) -> np.ndarray:
    """TF-IDF simple sur matrice binaire produits (réduit domination des produits fréquents)."""
    n, p = X_bin.shape
    df = X_bin.sum(axis=0)  # doc frequency
    idf = np.log((n + 1) / (df + 1)) + 1
    return X_bin * idf.reshape(1, -1)


def kmeans_segments(X: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    km = KMeans(n_clusters=k, random_state=seed, n_init=20)
    return km.fit_predict(X)


def scan_k(X: np.ndarray, k_min: int = 2, k_max: int = 12, seed: int = 42) -> pd.DataFrame:
    """Retourne inertie + silhouette pour choisir k (usage EDA)."""
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init=20)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels) if k > 1 else np.nan
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(sil)})
    return pd.DataFrame(rows)
