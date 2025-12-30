from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd


def load_train_test(train_csv: Path, test_csv: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Charge Train.csv et Test.csv."""
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    return train, test


def infer_product_cols(train: pd.DataFrame, profile_cols: list[str]) -> list[str]:
    """Déduit les colonnes produits (= toutes les colonnes hors profil)."""
    return [c for c in train.columns if c not in profile_cols]
