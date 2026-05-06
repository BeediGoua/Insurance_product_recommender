"""
Abstraction for constructing language models for agents.

Agents in this project can use either a hosted model on Hugging Face
(via ``InferenceClientModel``) or a local model served by Ollama (via
``LiteLLMModel``).  This module wraps the construction behind a
single ``build_model`` function so that the rest of the agent code
doesn't need to know about the underlying provider details.  If
smolagents or the relevant provider classes are not installed the
function will raise an informative error.
"""

from __future__ import annotations

from typing import Optional

def build_model(provider: str = "huggingface", model_id: Optional[str] = None):
    """Construct a language model instance for the given provider.

    Parameters
    ----------
    provider: str
        One of ``"huggingface"`` or ``"ollama"``.  Additional providers
        may be supported in the future.
    model_id: Optional[str]
        Identifier of the model to load.  If ``None`` a sensible
        default will be chosen based on the provider.

    Returns
    -------
    An instance of the corresponding smolagents model class.

    Raises
    ------
    ImportError
        If smolagents is not installed.
    ValueError
        If ``provider`` is unknown.
    """
    try:
        from smolagents import InferenceClientModel, LiteLLMModel
    except ImportError as e:
        raise ImportError(
            "smolagents package is required for agent functionality. "
            "Please install smolagents[toolkit] to use language model agents."
        ) from e
    if provider == "huggingface":
        return InferenceClientModel(
            model_id=model_id or "Qwen/Qwen2.5-Coder-32B-Instruct",
            max_tokens=4096,
            temperature=0.2,
        )
    if provider == "ollama":
        import os
        return LiteLLMModel(
            model_id=model_id or "ollama/llama3.1",
            api_base=os.environ.get("OLLAMA_API_BASE", "http://localhost:11434"),
            temperature=0.2,
        )
    raise ValueError(f"Unknown provider: {provider}")
