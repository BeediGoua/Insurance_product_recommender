"""
Functions for constructing a :class:`~decisionflow.schemas.ClientProfile` from
raw data.  In the existing project the raw data comes from CSV files
loaded via ``src/data/io``, but the functions here do not depend on
those internals – callers are expected to supply whatever information
they have about a client.

This module provides a thin layer of validation and inference on top of
the raw user attributes.  For example it can infer a client segment
based on their age and occupation, detect missing critical fields and
normalise product lists.  Keeping these transformations in one place
allows the decision engine to remain simple and focused on orchestration.
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from datetime import datetime

import pandas as pd

from .schemas import ClientProfile

def extract_current_products(row: Dict[str, Any], product_cols: Optional[List[str]] = None) -> List[str]:
    """Extract the list of currently owned products from a row of data.

    Parameters
    ----------
    row: Dict[str, Any]
        A mapping of column names to values for a single client.
    product_cols: Optional[List[str]]
        An optional list of column names corresponding to product
        indicators.  If provided the function will use these names;
        otherwise it will treat any key starting with ``"P"`` as a
        product column.

    Returns
    -------
    List[str]
        List of product codes (column names) for which the value in
        ``row`` is truthy (non‑zero, non‑empty).
    """
    products = []
    for col, val in row.items():
        if product_cols is not None:
            if col not in product_cols:
                continue
        else:
            # Heuristic: treat columns that look like P1, P2... as products
            if not (isinstance(col, str) and col.lower().startswith("p")):
                continue
        try:
            if bool(int(val)):
                products.append(col)
        except Exception:
            # non numeric values (e.g. blank strings) are treated as not owned
            pass
    return products


def infer_client_segment(row: Dict[str, Any]) -> Optional[str]:
    """Infer a simple customer segment from the row's attributes.

    This is a very naive implementation intended purely for
    demonstration.  A real segmentation would use statistical
    clustering or business rules.  If age is over 50 the segment is
    ``"senior"``; if under 30 it's ``"young"``; otherwise ``"adult"``.
    Occupation keywords can also influence the segment.
    """
    age = None
    if "age" in row and row["age"]:
        try:
            age = int(row["age"])
        except Exception:
            pass
    elif "birth_year" in row and row["birth_year"]:
        try:
            birth_year = int(row["birth_year"])
            current_year = datetime.now().year
            age = current_year - birth_year
        except Exception:
            pass
    if age is not None:
        if age >= 50:
            return "senior"
        if age <= 30:
            return "young"
    # fallback
    return "adult"


def detect_missing_fields(row: Dict[str, Any], required_fields: Optional[List[str]] = None) -> str:
    """Assess the data quality by checking for missing fields.

    If any of the ``required_fields`` is missing or falsy in the row the
    returned string will be ``"incomplete"``.  Otherwise ``"complete"``.

    Parameters
    ----------
    row: Dict[str, Any]
        Data for a single client.
    required_fields: Optional[List[str]]
        List of keys that are considered mandatory.  Defaults to a
        typical set of demographic features.
    """
    if required_fields is None:
        required_fields = ["age", "sex", "marital_status", "branch_code"]
    for field in required_fields:
        if not row.get(field):
            return "incomplete"
    return "complete"


def build_client_profile(client_id: str, row: Optional[Dict[str, Any]] = None, product_cols: Optional[List[str]] = None) -> ClientProfile:
    """Construct a :class:`ClientProfile` from raw attributes.

    Parameters
    ----------
    client_id: str
        Unique identifier of the client.
    row: Optional[Dict[str, Any]]
        If provided, a mapping of the client's attributes.  When not
        provided the function will return an empty profile with just the
        identifier set.
    product_cols: Optional[List[str]]
        Explicit list of product column names.  Only relevant when
        ``row`` is provided.

    Returns
    -------
    ClientProfile
        A populated profile dataclass.
    """
    if row is None:
        return ClientProfile(client_id=client_id)

    # Extract owned products
    owned = extract_current_products(row, product_cols=product_cols)
    # Infer segment
    segment = infer_client_segment(row)
    # Detect missing fields
    data_quality = detect_missing_fields(row)

    # Additional user features for ML model – exclude product columns
    extra_info = {k: v for k, v in row.items() if product_cols is None or k not in product_cols}

    return ClientProfile(
        client_id=client_id,
        segment=segment,
        current_products=owned,
        needs_signals=[],
        data_quality=data_quality,
        extra_info=extra_info,
    )
