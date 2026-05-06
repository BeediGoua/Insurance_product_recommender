"""
Tools exposed to language model agents.

Each tool is decorated with ``@tool`` from smolagents.  Tools are
lightweight wrappers around deterministic functions defined in the
``decisionflow`` package.  They must not implement any logic on their
own, they simply bridge the gap between natural language requests and
Python code.
"""

from __future__ import annotations

import json
from typing import Any

try:
    from smolagents import tool
except ImportError:
    # Define a no‑op decorator to allow import without smolagents
    def tool(func=None, *, name=None, description=None):
        def decorator(f):
            return f
        if func is None:
            return decorator
        return decorator(func)

from src.decisionflow.decision_engine import run_decisionflow_for_client
from src.decisionflow.policy_rules import apply_policy_rules
from src.decisionflow.risk_scoring import compute_recommendation_risk
from src.decisionflow.explanation import build_recommendation_explanation
from src.decisionflow.audit import create_audit_record, save_audit_record
from src.decisionflow.schemas import ClientProfile, RecommendationResult, PolicyDecision


@tool
def run_decisionflow_tool(client_id: str) -> str:
    """Run the full insurance decision workflow for one client.

    Args
    ----
    client_id: str
        Client identifier.

    Returns
    -------
    str
        JSON serialisation of the decisionflow result.
    """
    result = run_decisionflow_for_client(client_id)
    return json.dumps(result, default=str)


@tool
def apply_policy_rules_tool(client_profile_json: str, recommendations_json: str) -> str:
    """Apply insurance business rules to recommendations.

    The input strings must be JSON serialisations of a ClientProfile
    and a RecommendationResult (with at least ``filtered_scores``).
    """
    import json as _json
    from src.decisionflow.schemas import ClientProfile, RecommendationResult
    profile_dict = _json.loads(client_profile_json)
    rec_dict = _json.loads(recommendations_json)
    # Reconstruct dataclasses
    profile = ClientProfile(
        client_id=profile_dict.get("client_id"),
        segment=profile_dict.get("segment"),
        current_products=profile_dict.get("current_products", []),
        needs_signals=profile_dict.get("needs_signals", []),
        data_quality=profile_dict.get("data_quality"),
        extra_info=profile_dict.get("extra_info", {}),
    )
    rec = RecommendationResult(
        client_id=rec_dict.get("client_id"),
        raw_scores=rec_dict.get("raw_scores", {}),
        filtered_scores=rec_dict.get("filtered_scores", {}),
        top_k=rec_dict.get("top_k", []),
    )
    decision = apply_policy_rules(profile, rec)
    return _json.dumps(decision.__dict__, default=str)


@tool
def compute_risk_tool(recommendations_json: str) -> str:
    """Compute risk metrics for a recommendation result.

    The input must be a JSON serialisation of a RecommendationResult.
    """
    import json as _json
    from src.decisionflow.schemas import RecommendationResult
    rec_dict = _json.loads(recommendations_json)
    rec = RecommendationResult(
        client_id=rec_dict.get("client_id"),
        raw_scores=rec_dict.get("raw_scores", {}),
        filtered_scores=rec_dict.get("filtered_scores", {}),
        top_k=rec_dict.get("top_k", []),
    )
    risk = compute_recommendation_risk(rec)
    return _json.dumps(risk, default=str)


@tool
def generate_explanation_tool(client_profile_json: str, recommendations_json: str, policy_decision_json: str, risk_json: str) -> str:
    """Generate explanations for a recommendation.

    Args
    ----
    client_profile_json: JSON representation of a ClientProfile
    recommendations_json: JSON representation of a RecommendationResult
    policy_decision_json: JSON representation of a PolicyDecision
    risk_json: JSON representation of the risk metrics
    """
    import json as _json
    from src.decisionflow.schemas import ClientProfile, RecommendationResult, PolicyDecision
    profile_dict = _json.loads(client_profile_json)
    rec_dict = _json.loads(recommendations_json)
    policy_dict = _json.loads(policy_decision_json)
    risk = _json.loads(risk_json)
    profile = ClientProfile(
        client_id=profile_dict.get("client_id"),
        segment=profile_dict.get("segment"),
        current_products=profile_dict.get("current_products", []),
        needs_signals=profile_dict.get("needs_signals", []),
        data_quality=profile_dict.get("data_quality"),
        extra_info=profile_dict.get("extra_info", {}),
    )
    rec = RecommendationResult(
        client_id=rec_dict.get("client_id"),
        raw_scores=rec_dict.get("raw_scores", {}),
        filtered_scores=rec_dict.get("filtered_scores", {}),
        top_k=rec_dict.get("top_k", []),
    )
    policy = PolicyDecision(
        allowed=policy_dict.get("allowed", []),
        blocked=policy_dict.get("blocked", []),
        reasons=policy_dict.get("reasons", {}),
    )
    explanations = build_recommendation_explanation(profile, rec, policy, risk)
    # Serialise explanation dataclasses
    exp_list = [e.__dict__ for e in explanations]
    return _json.dumps(exp_list, default=str)


@tool
def create_audit_tool(client_profile_json: str, recommendations_json: str, policy_decision_json: str, explanations_json: str, risk_json: str) -> str:
    """Create and save an audit record for a recommendation.

    Returns the timestamp of the audit event.
    """
    import json as _json
    from src.decisionflow.schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult
    profile_dict = _json.loads(client_profile_json)
    rec_dict = _json.loads(recommendations_json)
    policy_dict = _json.loads(policy_decision_json)
    explanations_list = _json.loads(explanations_json)
    risk = _json.loads(risk_json)
    profile = ClientProfile(
        client_id=profile_dict.get("client_id"),
        segment=profile_dict.get("segment"),
        current_products=profile_dict.get("current_products", []),
        needs_signals=profile_dict.get("needs_signals", []),
        data_quality=profile_dict.get("data_quality"),
        extra_info=profile_dict.get("extra_info", {}),
    )
    rec = RecommendationResult(
        client_id=rec_dict.get("client_id"),
        raw_scores=rec_dict.get("raw_scores", {}),
        filtered_scores=rec_dict.get("filtered_scores", {}),
        top_k=rec_dict.get("top_k", []),
    )
    policy = PolicyDecision(
        allowed=policy_dict.get("allowed", []),
        blocked=policy_dict.get("blocked", []),
        reasons=policy_dict.get("reasons", {}),
    )
    explanations = [ExplanationResult(**exp) for exp in explanations_list]
    from src.decisionflow.audit import create_audit_record, save_audit_record
    audit_record = create_audit_record(profile, rec, policy, explanations, risk)
    save_audit_record(audit_record)
    return audit_record.timestamp
