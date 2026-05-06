"""
Evaluation runner for the DecisionFlow system.

This script ties together the various evaluation routines and can be
used from within notebooks or scripts to produce a unified report.  It
expects a dataset of clients with ground truth missing products and a
DecisionFlow implementation.
"""

from __future__ import annotations

from typing import Dict, Any, List, Tuple
import json

from src.decisionflow.decision_engine import run_decisionflow_from_profile
from src.decisionflow.profile_builder import build_client_profile
from .recommender_eval import evaluate_recommender
from .policy_eval import evaluate_policy
from .explanation_eval import evaluate_explanations
from src.decisionflow.schemas import PolicyDecision, ExplanationResult


def run_all_evaluations(dataset: List[Tuple[str, str, dict]]) -> Dict[str, Any]:
    """Run recommender, policy and explanation evaluations on a dataset.

    Parameters
    ----------
    dataset: List[Tuple[str, str, dict]]
        A list of tuples ``(client_id, true_product, raw_row)``.  ``raw_row``
        should contain at least the product indicators and user features.

    Returns
    -------
    Dict[str, Any]
        Aggregated metrics for each component.
    """
    predictions: List[List[str]] = []
    truths: List[str] = []
    decisions: List[PolicyDecision] = []
    expls: List[List[ExplanationResult]] = []
    for client_id, true_product, row in dataset:
        profile = build_client_profile(client_id, row)
        result = run_decisionflow_from_profile(profile)
        predictions.append(result["recommendations"])
        truths.append(true_product)
        decisions.append(result["policy"])
        expls.append(result["explanations"])
    metrics = {}
    metrics["recommender"] = evaluate_recommender(predictions, truths)
    metrics["policy"] = evaluate_policy(decisions)
    metrics["explanation"] = evaluate_explanations(expls)
    return metrics
