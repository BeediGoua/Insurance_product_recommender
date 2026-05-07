import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

from src.evaluation.agent_eval import evaluate_agent


def test_evaluate_agent_empty():
    m = evaluate_agent([])
    assert m["n_runs"] == 0
    assert m["correct_tool_call_rate"] == 0.0
    assert m["error_rate"] == 0.0


def test_evaluate_agent_with_expected_tools():
    conversations = [
        {
            "tool_calls": ["get_client_profile_tool", "apply_policy_rules_tool"],
            "expected_tools": ["get_client_profile_tool", "apply_policy_rules_tool"],
            "latency_ms": 120,
            "cost_usd": 0.01,
            "error": False,
        },
        {
            "tool_calls": ["get_client_profile_tool"],
            "expected_tools": ["get_client_profile_tool", "apply_policy_rules_tool"],
            "latency_ms": 180,
            "cost_usd": 0.02,
            "error": True,
        },
    ]

    m = evaluate_agent(conversations)
    assert m["n_runs"] == 2
    assert 0.0 <= m["correct_tool_call_rate"] <= 1.0
    assert m["error_rate"] == 0.5
    assert m["total_cost_usd"] == 0.03
    assert m["avg_latency_ms"] == 150.0


def test_evaluate_agent_unnecessary_tool_calls():
    conversations = [
        {
            "tools_called": ["get_client_profile_tool", "unknown_tool"],
            "required_tools": ["get_client_profile_tool"],
        }
    ]
    m = evaluate_agent(conversations)
    assert m["coverage_rate"] == 1.0
    assert m["correct_tool_call_rate"] == 1.0
    assert m["unnecessary_tool_rate"] == 1.0
