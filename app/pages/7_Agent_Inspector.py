import streamlit as st

st.set_page_config(page_title="Agent Inspector", layout="wide", initial_sidebar_state="expanded")
st.title("Agent Inspector")
st.markdown("""
This experimental page exposes the agentic interface built on top of the
deterministic DecisionFlow.  It allows you to interact with the
manager agent using natural language prompts.  Under the hood the
agent will call Python tools to fetch profiles, run recommendations,
apply policy and risk checks and generate explanations.

**Note:** Agents require the `smolagents` package.  If it is not
installed this page will display an error.
""")

try:
    from src.agents.manager import run_agentic_recommendation
    smolagents_available = True
except Exception as e:
    smolagents_available = False
    error_message = str(e)

if not smolagents_available:
    st.error(
        f"Agent functionality is unavailable: {error_message}.\n"
        "Install smolagents[toolkit] and ensure the chosen model provider is configured."
    )
else:
    provider = st.selectbox("Model Provider", options=["huggingface", "ollama"], index=0)
    user_prompt = st.text_area(
        "Prompt",
        value="Analyse le client C123. Donne les recommandations d’assurance, applique les règles métier, explique les raisons, et précise les limites.",
        height=150,
    )
    if st.button("Run Agent"):
        with st.spinner("Running agent..."):
            try:
                response = run_agentic_recommendation(user_prompt, provider=provider)
                st.write(response)
            except Exception as e:
                st.error(f"Agent execution failed: {e}")
