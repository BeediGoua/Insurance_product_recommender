import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.decisionflow.schemas import ClientProfile, RecommendationResult
from src.decisionflow.policy_rules import apply_policy_rules


def test_owned_product_is_blocked():
    profile = ClientProfile(client_id="C1", current_products=["Life"])
    rec = RecommendationResult(
        client_id="C1",
        raw_scores={"Life": 0.9, "Home": 0.8},
        filtered_scores={"Life": 0.9, "Home": 0.8},
        top_k=["Life", "Home"],
    )
    decision = apply_policy_rules(profile, rec)
    assert "Life" in decision.blocked
    assert "Home" in decision.allowed