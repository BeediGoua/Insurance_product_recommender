"""
Standard data structures used throughout the DecisionFlow system.

Using dataclasses here helps to enforce a clear contract between the
different layers of the system (profiling, recommendation, policy,
risk/explainability and audit).  These objects are intentionally kept
simple and serialisable so they can be easily logged, inspected or
passed between processes.  See the documentation in the README for
further details on each field.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class ClientProfile:
    """A representation of a client's state relevant to the recommender.

    Attributes
    ----------
    client_id: str
        Unique identifier for the client.
    segment: Optional[str]
        A high‑level grouping for the client (e.g. "family_protection_candidate").
    current_products: List[str]
        List of product codes the client already owns.  These must be
        excluded from any recommendation.
    needs_signals: List[str]
        Optional list of detected needs (e.g. "home_protection", "life_protection").
        This can be used by explainability modules or policy checks.
    data_quality: Optional[str]
        Indicates whether the underlying source data is complete or if
        some important fields are missing.  For example "complete",
        "incomplete" or "unknown".
    extra_info: Dict[str, Any]
        A catch‑all for any additional features collected about the client
        (age, income, occupation, etc.).  When passed to the underlying
        ML model these keys must match the expected feature names.
    """

    client_id: str
    segment: Optional[str] = None
    current_products: List[str] = field(default_factory=list)
    needs_signals: List[str] = field(default_factory=list)
    data_quality: Optional[str] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProductCandidate:
    """A potential product recommendation with an associated score.

    Attributes
    ----------
    product_code: str
        Identifier of the recommended product.
    score: float
        Score or probability assigned by the underlying model.
    is_eligible: bool
        Flag indicating whether the product passed all policy checks.
    """

    product_code: str
    score: float
    is_eligible: bool = True


@dataclass
class RecommendationResult:
    """Container for raw and filtered recommendation results.

    Attributes
    ----------
    client_id: str
        Identifier of the client these recommendations were generated for.
    raw_scores: Dict[str, float]
        Mapping of product codes to raw scores produced by the ML engine.
    filtered_scores: Dict[str, float]
        Mapping of product codes to scores after business rules and
        anti‑cheat filters have been applied (e.g. already owned products
        set to ``-inf`` or removed).
    top_k: List[str]
        Final list of recommended product codes (sorted by descending
        filtered score) to present to the user.  The length of this
        list is determined by the caller.
    """

    client_id: str
    raw_scores: Dict[str, float]
    filtered_scores: Dict[str, float]
    top_k: List[str]


@dataclass
class PolicyDecision:
    """Result of applying policy and eligibility rules to a set of products.

    Attributes
    ----------
    allowed: List[str]
        List of product codes that passed all eligibility and business rules.
    blocked: List[str]
        List of products that were filtered out due to policy constraints.
    reasons: Dict[str, str]
        A mapping from product codes to a textual reason for why they
        were blocked.  Products that are allowed may also have entries
        in this dictionary if they triggered non‑blocking rules.
    """

    allowed: List[str]
    blocked: List[str]
    reasons: Dict[str, str] = field(default_factory=dict)


@dataclass
class ExplanationResult:
    """Explanation for a single recommended product.

    Attributes
    ----------
    product: str
        The code of the product being explained.
    reasons: List[str]
        A list of human‑readable reasons for why this product was
        recommended.  The first reason should be the most important.
    limitations: List[str]
        A list of caveats or limitations associated with the
        recommendation (e.g. "based on historical averages only").
    """

    product: str
    reasons: List[str]
    limitations: List[str] = field(default_factory=list)


@dataclass
class AuditRecord:
    """Audit information for a recommendation decision.

    This record captures all inputs and outputs of the recommendation
    process and is intended to be persisted for traceability and
    governance.  Depending on deployment requirements audit records can
    be written to a database, message queue or a simple JSON log file.

    Attributes
    ----------
    timestamp: str
        ISO formatted timestamp when the recommendation was made.
    client_id: str
        Identifier of the client involved in this decision.
    input_profile: Dict[str, Any]
        A serialisable representation of the client profile used by the
        engine.  This should match the fields in :class:`ClientProfile`.
    raw_model_scores: Dict[str, float]
        The unfiltered scores from the underlying ML engine.
    rules_triggered: List[str]
        Names of the policy rules that were triggered for this client.
    final_recommendations: List[str]
        Final list of recommended products after all filtering.
    explanation: Optional[str]
        Optional free text explanation summarising the reasoning.
    model_version: str
        Identifier for the underlying model version used.
    policy_version: str
        Identifier for the policy ruleset version used.
    """

    timestamp: str
    client_id: str
    input_profile: Dict[str, Any]
    raw_model_scores: Dict[str, float]
    rules_triggered: List[str]
    final_recommendations: List[str]
    explanation: Optional[str] = None
    model_version: str = ""
    policy_version: str = ""
