from __future__ import annotations

import pandas as pd


def normalize_cat(s: pd.Series) -> pd.Series:
    """Normalisation prod-safe : strip + casefold (corrige 'F' vs 'f', espaces, etc.)."""
    return s.astype(str).str.strip().str.casefold()


def extract_join_year(join_date: pd.Series) -> pd.Series:
    """
    Extrait l'année depuis join_date, sans interpréter jour/mois (ambigus).
    Ex: '12/3/2018' -> 2018
    """
    return pd.to_numeric(join_date.astype(str).str.split("/").str[-1], errors="coerce")


def add_join_year_and_age(
    df: pd.DataFrame,
    *,
    age_min: int = 18,
    age_max: int = 90,
) -> pd.DataFrame:
    """Ajoute join_year, age_raw, age (clippé) + flags qualité."""
    out = df.copy()

    out["join_year"] = extract_join_year(out["join_date"])
    out["age_raw"] = out["join_year"] - out["birth_year"]
    out["age"] = out["age_raw"].clip(age_min, age_max)

    out["join_year_missing"] = out["join_year"].isna()
    out["age_missing"] = out["age_raw"].isna()
    out["age_was_clipped"] = (out["age_raw"] != out["age"]) & (~out["age_raw"].isna())

    return out


def apply_basic_cleaning(
    df: pd.DataFrame,
    *,
    cat_cols: list[str],
    age_min: int = 18,
    age_max: int = 90,
) -> pd.DataFrame:
    """
    Pipeline minimal et réutilisable :
    - normalisation des catégorielles (case/espaces)
    - join_year + age + flags
    """
    out = df.copy()
    for c in cat_cols:
        out[c] = normalize_cat(out[c])

    out = add_join_year_and_age(out, age_min=age_min, age_max=age_max)
    return out
