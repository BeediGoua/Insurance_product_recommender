"""
Load and access product metadata.

The product catalog is stored as a CSV file under
``data/product_knowledge/product_descriptions.csv``.  Each row should
contain at minimum the following columns:

* ``product_code`` – unique identifier (e.g. "Life", "Motor").
* ``product_name`` – human friendly name.
* ``description`` – marketing description of the product.
* ``target_needs`` – comma separated list of needs this product
  addresses.
* ``typical_customer`` – short description of a typical buyer.
* ``exclusions`` – optional notes on who should not buy this product.
* ``business_notes`` – any additional information for internal use.

If the file does not exist or cannot be read the returned DataFrame
will be empty.
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

# Compute the path relative to the project root so that tests can find the data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PRODUCT_CATALOG_PATH = PROJECT_ROOT / "data/product_knowledge/product_descriptions.csv"


def get_product_catalog() -> pd.DataFrame:
    """Load the product catalog as a pandas DataFrame.

    Returns an empty DataFrame if the file does not exist.
    """
    if not PRODUCT_CATALOG_PATH.exists():
        return pd.DataFrame()
    try:
        # Use UTF‑8 explicitly to avoid issues with special characters
        df = pd.read_csv(PRODUCT_CATALOG_PATH, encoding="utf-8")
        return df
    except Exception:
        # On any parsing failure return an empty DataFrame
        return pd.DataFrame()


def get_product_info(product_code: str) -> dict:
    """Return the row for the specified product as a dict.

    If the product is not found or the catalog cannot be loaded an
    empty dictionary is returned.
    """
    df = get_product_catalog()
    if df.empty:
        return {}
    row = df[df["product_code"] == product_code]
    if row.empty:
        return {}
    return row.iloc[0].to_dict()
