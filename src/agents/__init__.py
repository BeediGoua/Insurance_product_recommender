"""Expose agent factories and tools for the DecisionFlow agent layer."""

from .model_provider import build_model  # noqa: F401
from .tools import (
    run_decisionflow_tool,
    apply_policy_rules_tool,
    compute_risk_tool,
    generate_explanation_tool,
    create_audit_tool,
    get_client_profile_tool,
    run_recommender_tool,
)  # noqa: F401
from .manager import (
    build_recommendation_agent,
    build_profiling_agent,
    build_policy_agent,
    build_risk_agent,
    build_explanation_agent,
    build_audit_agent,
    build_manager_agent,
    run_agentic_recommendation,
)  # noqa: F401
