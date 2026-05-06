"""
Agents layer for the DecisionFlow system.

This package contains the glue code between the deterministic
recommendation pipeline and language‑model based agents.  Agents are
optional: the core recommendation logic resides in the
``decisionflow`` package.  When enabled the agents coordinate calls
into the deterministic code via registered tools and provide a more
natural language interface for users and developers.
"""

from .model_provider import build_model  # noqa: F401
from .tools import (
    run_decisionflow_tool,
    apply_policy_rules_tool,
    compute_risk_tool,
    generate_explanation_tool,
    create_audit_tool,
)  # noqa: F401
from .manager import (
    build_recommendation_agent,
    build_risk_agent,
    build_explanation_agent,
    build_audit_agent,
    build_manager_agent,
    run_agentic_recommendation,
)  # noqa: F401
