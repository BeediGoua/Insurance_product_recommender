"""
This package contains the building blocks for the DecisionFlow layer.  The
modules defined here wrap the existing recommendation engine with a set of
clearly defined abstractions (schemas, profiling, policy checks, risk
assessment, explainability and audit).  By isolating these concerns in
separate modules the project can grow from a simple recommender to a
decision‑support system without rewriting the core machine learning code.

The DecisionFlow API is deliberately lightweight: each function receives
plain Python objects (typically dataclasses defined in ``schemas.py``) and
returns new objects of the same family.  This makes the system easy to
reason about, test and extend.  See ``decision_engine.py`` for the
orchestrator that ties everything together.
"""

from .schemas import (  # noqa: F401
    ClientProfile,
    ProductCandidate,
    RecommendationResult,
    PolicyDecision,
    ExplanationResult,
    AuditRecord,
)
from .profile_builder import build_client_profile
from .recommendation_context import run_recommender
from .policy_rules import apply_policy_rules, load_product_rules
from .risk_scoring import compute_recommendation_risk
from .explanation import build_recommendation_explanation
from .audit import create_audit_record, save_audit_record
from .decision_engine import run_decisionflow_for_client, run_decisionflow_from_profile
