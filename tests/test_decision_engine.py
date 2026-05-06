import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.decisionflow.decision_engine import run_decisionflow_for_client


def test_decisionflow_returns_structure():
    result = run_decisionflow_for_client("TEST_CLIENT", topk=2, use_hybrid=False)
    # Expect certain keys
    assert "client_profile" in result
    assert "recommendations" in result
    assert "risk" in result
    assert "policy" in result