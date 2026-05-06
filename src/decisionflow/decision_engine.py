"""
High level orchestrator for the DecisionFlow system.

This module exposes functions to run the full insurance decision flow
either from a client identifier or from an already constructed
``ClientProfile``.  The flow consists of:

1. Building or receiving a client profile.
2. Generating recommendations using the ML engine.
3. Applying policy and eligibility rules.
4. Computing risk/confidence metrics.
5. Generating human explanations.
6. Creating and saving an audit record.

The returned value is a serialisable dictionary containing all of the
above outputs.  Users of the library or the Streamlit UI can rely on
this module to encapsulate the decision logic in a single call.
"""

from __future__ import annotations

from typing import Dict, Any, Optional, List

from .schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult
from .profile_builder import build_client_profile
from .recommendation_context import run_catboost_model, run_statistical_baseline
from .policy_rules import apply_policy_rules, load_product_rules, load_exclusions
from .risk_scoring import compute_recommendation_risk
from .explanation import build_recommendation_explanation
from .audit import create_audit_record, save_audit_record


def run_decisionflow_for_client(client_id: str, topk: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
    """Run the full DecisionFlow pipeline for a given client ID.

    If the client cannot be found in any underlying dataset a minimal
    profile will be constructed.  For demonstration purposes the
    function does not attempt to read from disk.  Real applications
    should supply a row of client attributes to
    :func:`build_client_profile`.
    """
    # Build a minimal profile – this could be extended to fetch data
    profile = build_client_profile(client_id)
    return run_decisionflow_from_profile(profile, topk=topk, use_hybrid=use_hybrid)


def run_decisionflow_from_profile(profile: ClientProfile, topk: int = 5, use_hybrid: bool = True) -> Dict[str, Any]:
    """Run the DecisionFlow pipeline given a pre‑constructed profile.

    Parameters
    ----------
    profile: ClientProfile
        The client's profile.  Must include the list of currently
        owned products and any additional features expected by the
        underlying ML model.
    topk: int
        Number of recommendations to return.
    use_hybrid: bool
        Whether to use the hybrid CatBoost+baseline model or fall back
        to the pure statistical baseline.  Hybrid is recommended when
        the model artefacts are available.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the profile, recommendations, policy
        decision, risk scores, explanations and audit ID.
    """
    # 1. Generate recommendations
    if use_hybrid:
        rec: RecommendationResult = run_catboost_model(profile, topk=topk)
        # If we failed to load the hybrid model fallback to baseline
        if not rec.top_k:
            rec = run_statistical_baseline(profile, topk=topk)
    else:
        rec = run_statistical_baseline(profile, topk=topk)

    # 2. Apply policy rules
    policy: PolicyDecision = apply_policy_rules(profile, rec)
    # 3. Compute risk metrics
    risk = compute_recommendation_risk(rec)
    # 4. Generate explanations
    explanations: List[ExplanationResult] = build_recommendation_explanation(profile, rec, policy, risk)
    # 5. Audit
    audit_record = create_audit_record(profile, rec, policy, explanations, risk)
    save_audit_record(audit_record)
    # 6. Return aggregated result
    return {
        "client_profile": profile,
        "recommendations": rec.top_k,
        "raw_scores": rec.raw_scores,
        "filtered_scores": rec.filtered_scores,
        "policy": policy,
        "risk": risk,
        "explanations": explanations,
        "audit_timestamp": audit_record.timestamp,
    }
