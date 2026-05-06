"""Provide client data loading and lookup utilities for DecisionFlow."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.io import infer_product_cols
from src.data.schema import DEFAULT_PROFILE_COLS

# Ordered list of candidate paths – the first one that exists will be used.
_CANDIDATE_PATHS: list[Path] = [
    Path("data/train_cleaned.parquet"),
    Path("artifacts/baseline_v0/train_cleaned.parquet"),
    Path("data/Train.csv"),
]

# Possible column names for the client identifier
_ID_COL_CANDIDATES: list[str] = ["ID", "client_id", "Customer ID", "customer_id"]


def _resolve_data_path() -> Optional[Path]:
    """Return the first existing candidate data path."""
    for p in _CANDIDATE_PATHS:
        if p.exists():
            return p
    return None


def load_clients_dataframe(path: Optional[Path] = None) -> pd.DataFrame:
    """Load the client dataframe from disk.

    Parameters
    ----------
    path: Optional[Path]
        Explicit path to a CSV or Parquet file.  When *None* the
        function tries the candidate paths defined in this module.

    Returns
    -------
    pd.DataFrame
        The loaded dataframe.

    Raises
    ------
    FileNotFoundError
        If no data file can be found.
    """
    if path is None:
        path = _resolve_data_path()
    if path is None or not path.exists():
        raise FileNotFoundError(
            "No client data file found. Tried: "
            + ", ".join(str(p) for p in _CANDIDATE_PATHS)
        )
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _detect_id_col(df: pd.DataFrame) -> str:
    """Return the name of the client identifier column."""
    for candidate in _ID_COL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"No client id column found. Expected one of: {_ID_COL_CANDIDATES}. "
        f"Got: {list(df.columns[:10])}"
    )


def get_client_row(
    client_id: str,
    df: Optional[pd.DataFrame] = None,
) -> tuple[dict, list[str]]:
    """Retrieve a single client row and the list of product columns.

    Parameters
    ----------
    client_id: str
        The client identifier to look up.
    df: Optional[pd.DataFrame]
        Pre-loaded dataframe.  When *None* the dataframe is loaded from
        disk via :func:`load_clients_dataframe`.

    Returns
    -------
    tuple[dict, list[str]]
        A ``(row_dict, product_cols)`` pair where ``row_dict`` is the
        client's raw attribute mapping and ``product_cols`` is the list
        of product column names detected in the dataset.

    Raises
    ------
    ValueError
        If the client is not found in the dataset.
    FileNotFoundError
        If the dataset cannot be loaded.
    """
    if df is None:
        df = load_clients_dataframe()

    id_col = _detect_id_col(df)
    matched = df[df[id_col].astype(str) == str(client_id)]
    if matched.empty:
        raise ValueError(
            f"Client '{client_id}' not found. "
            f"Check the ID or the dataset path."
        )
    row = matched.iloc[0].to_dict()
    product_cols = infer_product_cols(df, DEFAULT_PROFILE_COLS)
    return row, product_cols
