"""
Evaluation suite for DecisionFlow.

This package provides simple metrics and evaluation routines to
quantify the performance of the recommender, the policy rules,
explanations and the agent layer.  These metrics are illustrative and
intended for experimentation rather than strict benchmarking.
"""

from .recommender_eval import evaluate_recommender  # noqa: F401
from .policy_eval import evaluate_policy  # noqa: F401
from .explanation_eval import evaluate_explanations  # noqa: F401
from .agent_eval import evaluate_agent  # noqa: F401
from .eval_runner import run_all_evaluations  # noqa: F401
