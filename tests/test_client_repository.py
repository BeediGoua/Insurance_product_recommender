"""Tests for client_repository module."""
import pandas as pd
import pytest

from src.decisionflow.client_repository import get_client_row


def _make_fake_df():
    return pd.DataFrame({
        "ID": ["C001", "C002", "C123"],
        "join_date": ["2020-01-01", "2021-03-15", "2019-07-01"],
        "sex": ["M", "F", "M"],
        "marital_status": ["Married", "Single", "Married"],
        "birth_year": [1980, 1990, 1975],
        "branch_code": ["B01", "B02", "B01"],
        "occupation_code": [1, 2, 3],
        "occupation_category_code": [10, 20, 10],
        "P5DA": [1, 0, 1],
        "RIBP": [0, 1, 0],
        "8NN1": [0, 0, 1],
    })


def test_get_client_row_existing_client():
    df = _make_fake_df()
    row, product_cols = get_client_row("C123", df=df)
    assert row["ID"] == "C123"
    assert "P5DA" in product_cols
    assert "RIBP" in product_cols
    assert len(product_cols) == 3


def test_get_client_row_not_found():
    df = _make_fake_df()
    with pytest.raises(ValueError, match="C999"):
        get_client_row("C999", df=df)


def test_get_client_row_owned_products():
    """C123 owns P5DA and 8NN1 — product cols should include them."""
    df = _make_fake_df()
    row, product_cols = get_client_row("C123", df=df)
    assert row["P5DA"] == 1
    assert row["8NN1"] == 1
    assert row["RIBP"] == 0


def test_profile_built_from_real_row():
    """Profile built from real row must have non-empty current_products."""
    df = _make_fake_df()
    row, product_cols = get_client_row("C123", df=df)

    from src.decisionflow.profile_builder import build_client_profile
    profile = build_client_profile("C123", row=row, product_cols=product_cols)

    assert profile.client_id == "C123"
    assert len(profile.current_products) > 0
    assert "P5DA" in profile.current_products
