import os
import sys
import pathlib
import pandas as pd
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.decisionflow.decision_engine import run_decisionflow_for_client
from src.decisionflow.schemas import RecommendationResult


def test_decisionflow_returns_structure(monkeypatch):
    import src.decisionflow.client_repository as repo
    import src.decisionflow.decision_engine as engine

    # Fake single-row dataset for client lookup
    fake_df = pd.DataFrame({
        "ID": ["TEST_CLIENT"],
        "join_date": ["2020-01-01"],
        "sex": ["M"],
        "marital_status": ["Single"],
        "birth_year": [1990],
        "branch_code": ["B1"],
        "occupation_code": ["O1"],
        "occupation_category_code": ["C1"],
        "P5DA": [1],
        "RIBP": [0],
    })

    # Mock data loader
    monkeypatch.setattr(repo, "load_clients_dataframe", lambda path=None: fake_df)

    # Mock recommender to avoid model artifacts dependency
    def _fake_baseline(profile, topk=2):
        return RecommendationResult(
            client_id=profile.client_id,
            raw_scores={"P5DA": 0.9, "RIBP": 0.8},
            filtered_scores={"P5DA": 0.9, "RIBP": 0.8},
            top_k=["P5DA", "RIBP"],
        )

    monkeypatch.setattr(engine, "run_statistical_baseline", _fake_baseline)
    monkeypatch.setattr(engine, "save_audit_record", lambda record: None)

    result = run_decisionflow_for_client("TEST_CLIENT", topk=2, use_hybrid=False)

    # Contract checks across modules (profile -> rec -> policy -> risk -> explanation -> audit)
    assert "client_profile" in result
    assert "recommendations" in result
    assert "risk" in result
    assert "policy" in result
    assert "explanations" in result
    assert "audit_timestamp" in result

    # Policy guardrail: owned products must not be in allowed set
    allowed = set(result["policy"].allowed)
    owned = set(result["client_profile"].current_products)
    assert not (allowed & owned)