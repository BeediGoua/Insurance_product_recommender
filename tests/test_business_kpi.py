import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.business_kpi import evaluate_business_kpis


def test_business_kpi_bounds_and_keys():
    rec = {"Hit@3": 0.8}
    policy = {"forbidden_violation_rate": 0.0, "owned_violation_rate": 0.05}
    expl = {"explanation_quality_score": 0.7}

    out = evaluate_business_kpis(rec, policy, expl, avg_latency_ms=120.0)

    assert "kpi_cross_sell_proxy" in out
    assert "kpi_invalid_recommendation_rate" in out
    assert "kpi_explanation_quality" in out
    assert "kpi_efficiency_latency_ms" in out
    assert "kpi_readiness_score" in out
    assert 0.0 <= out["kpi_readiness_score"] <= 1.0


def test_business_kpi_regression_signal():
    good = evaluate_business_kpis(
        {"Hit@3": 0.9},
        {"forbidden_violation_rate": 0.0, "owned_violation_rate": 0.0},
        {"explanation_quality_score": 0.9},
    )
    bad = evaluate_business_kpis(
        {"Hit@3": 0.3},
        {"forbidden_violation_rate": 0.2, "owned_violation_rate": 0.2},
        {"explanation_quality_score": 0.3},
    )

    assert good["kpi_readiness_score"] > bad["kpi_readiness_score"]
