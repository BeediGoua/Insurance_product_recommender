import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.decisionflow.schemas import ClientProfile, RecommendationResult


def test_decisionflow_module_contract(monkeypatch):
    import src.decisionflow.decision_engine as engine

    def _fake_baseline(profile, topk=3):
        return RecommendationResult(
            client_id=profile.client_id,
            raw_scores={"P5DA": 0.95, "RIBP": 0.75, "8NN1": 0.6},
            filtered_scores={"P5DA": 0.95, "RIBP": 0.75, "8NN1": 0.6},
            top_k=["P5DA", "RIBP", "8NN1"],
        )

    monkeypatch.setattr(engine, "run_statistical_baseline", _fake_baseline)
    monkeypatch.setattr(engine, "save_audit_record", lambda record: None)

    profile = ClientProfile(
        client_id="C-contract",
        current_products=["P5DA"],
        extra_info={"age": 40, "sex": "M"},
    )

    result = engine.run_decisionflow_from_profile(profile, topk=3, use_hybrid=False)

    # Pipeline outputs exist
    assert "client_profile" in result
    assert "recommendations" in result
    assert "raw_scores" in result
    assert "filtered_scores" in result
    assert "policy" in result
    assert "risk" in result
    assert "explanations" in result
    assert "audit_timestamp" in result

    # policy -> explanations contract: explanations only for allowed products
    allowed = set(result["policy"].allowed)
    explained = {e.product for e in result["explanations"]}
    assert explained.issubset(allowed)

    # no allowed product is already owned
    assert not (allowed & set(profile.current_products))
