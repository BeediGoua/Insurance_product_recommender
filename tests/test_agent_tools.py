import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.agents.tools import run_decisionflow_tool
import json
import pandas as pd
import pytest


def test_run_decisionflow_tool_returns_json():
    try:
        output = run_decisionflow_tool("C3")
        result = json.loads(output)
        assert isinstance(result, dict)
        assert "client_profile" in result
    except (ValueError, FileNotFoundError):
        pytest.skip("Client data not available in this environment")


def test_get_client_profile_tool_structure():
    from src.agents.tools import get_client_profile_tool
    import src.decisionflow.client_repository as repo

    fake_df = pd.DataFrame({
        "ID": ["C123"],
        "join_date": ["2019-07-01"],
        "sex": ["M"],
        "marital_status": ["Married"],
        "birth_year": [1975],
        "branch_code": ["B01"],
        "occupation_code": [3],
        "occupation_category_code": [10],
        "P5DA": [1],
        "RIBP": [0],
    })

    original = repo.load_clients_dataframe
    repo.load_clients_dataframe = lambda path=None: fake_df
    try:
        output = get_client_profile_tool("C123")
        data = json.loads(output)
        assert "client_id" in data
        assert "current_products" in data
        assert data["client_id"] == "C123"
        assert "P5DA" in data["current_products"]
    finally:
        repo.load_clients_dataframe = original