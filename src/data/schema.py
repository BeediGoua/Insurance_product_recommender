from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Schema:
    profile_cols: List[str]
    product_cols: List[str]


DEFAULT_PROFILE_COLS: List[str] = [
    "ID",
    "join_date",
    "sex",
    "marital_status",
    "birth_year",
    "branch_code",
    "occupation_code",
    "occupation_category_code",
]
