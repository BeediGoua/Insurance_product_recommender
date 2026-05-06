"""
Build and run multi‑agent orchestration for insurance recommendation.

Agents are composed hierarchically using smolagents' ``CodeAgent``.  A
top‑level manager agent delegates tasks to specialised sub‑agents such
as the decision agent (which calls the deterministic DecisionFlow
pipeline) and the explanation agent.  If smolagents is not installed
the factory functions will raise an ImportError.
"""

from __future__ import annotations

from typing import Optional, List

from .model_provider import build_model
from .prompts import (
    MANAGER_SYSTEM_PROMPT,
    RECOMMENDATION_AGENT_PROMPT,
    EXPLANATION_AGENT_PROMPT,
    RISK_AGENT_PROMPT,
    AUDIT_AGENT_PROMPT,
)
from .tools import (
    run_decisionflow_tool,
    compute_risk_tool,
    generate_explanation_tool,
    create_audit_tool,
)


def _import_codeagent():
    try:
        from smolagents import CodeAgent
        return CodeAgent
    except ImportError as e:
        raise ImportError(
            "smolagents package is required to build agents. "
            "Please install smolagents[toolkit] to use agentic functionality."
        ) from e


def build_recommendation_agent(provider: str = "huggingface", model_id: Optional[str] = None):
    """Create an agent responsible for running the deterministic decisionflow.

    The agent uses the run_decisionflow_tool to call into Python code.
    """
    CodeAgent = _import_codeagent()
    model = build_model(provider, model_id)
    return CodeAgent(
        model=model,
        tools=[run_decisionflow_tool],
        name="recommendation_agent",
        description="Runs the deterministic insurance recommendation workflow.",
        max_steps=5,
        verbosity_level=1,
    )


def build_risk_agent(provider: str = "huggingface", model_id: Optional[str] = None):
    """Create an agent that computes risk metrics."""
    CodeAgent = _import_codeagent()
    model = build_model(provider, model_id)
    return CodeAgent(
        model=model,
        tools=[compute_risk_tool],
        name="risk_agent",
        description="Computes risk and confidence for recommendations.",
        max_steps=5,
        verbosity_level=1,
    )


def build_explanation_agent(provider: str = "huggingface", model_id: Optional[str] = None):
    """Create an agent that generates explanations."""
    CodeAgent = _import_codeagent()
    model = build_model(provider, model_id)
    return CodeAgent(
        model=model,
        tools=[generate_explanation_tool],
        name="explanation_agent",
        description="Explains recommendations using deterministic context.",
        max_steps=5,
        verbosity_level=1,
    )


def build_audit_agent(provider: str = "huggingface", model_id: Optional[str] = None):
    """Create an agent responsible for creating audit records."""
    CodeAgent = _import_codeagent()
    model = build_model(provider, model_id)
    return CodeAgent(
        model=model,
        tools=[create_audit_tool],
        name="audit_agent",
        description="Persists audit logs for recommendations.",
        max_steps=3,
        verbosity_level=1,
    )


def build_manager_agent(provider: str = "huggingface", model_id: Optional[str] = None):
    """Create the top level manager agent which orchestrates sub‑agents."""
    CodeAgent = _import_codeagent()
    model = build_model(provider, model_id)
    rec_agent = build_recommendation_agent(provider, model_id)
    risk_agent = build_risk_agent(provider, model_id)
    expl_agent = build_explanation_agent(provider, model_id)
    audit_agent = build_audit_agent(provider, model_id)
    return CodeAgent(
        model=model,
        tools=[],
        managed_agents=[rec_agent, risk_agent, expl_agent, audit_agent],
        name="insurance_manager_agent",
        description="Coordinates insurance recommendation, risk assessment, explanation and audit.",
        max_steps=10,
        verbosity_level=2,
    )


def run_agentic_recommendation(
    prompt: str,
    provider: str = "huggingface",
    model_id: Optional[str] = None,
) -> str:
    """Run the manager agent with a free form prompt.

    This helper builds a manager agent on the fly and executes it.  It
    returns whatever string the agent produces as its final answer.  If
    smolagents is not installed or any error occurs a RuntimeError is
    raised.
    """
    manager = build_manager_agent(provider, model_id)
    # The CodeAgent API defines a ``run`` method which accepts a prompt
    # and returns the final output.  We pass through the prompt
    # directly.  Additional keywords such as chat history could be
    # supplied here if needed.
    return manager.run(prompt)
