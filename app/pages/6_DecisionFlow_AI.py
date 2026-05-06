import json
import streamlit as st

from src.decisionflow.decision_engine import run_decisionflow_for_client
from src.decisionflow.profile_builder import build_client_profile


st.set_page_config(page_title="DecisionFlow AI", layout="wide", initial_sidebar_state="expanded")
st.title("Insurance DecisionFlow AI")
st.markdown("""
This page runs the full decision‑support workflow on a single client.  Enter
a client identifier below to receive a personalised recommendation along
with policy checks, risk assessment and explanations.
""")

client_id = st.text_input("Client ID", value="C123")
use_hybrid = st.checkbox("Use Hybrid Model (CatBoost + Baseline)", value=True)
topk = st.slider("Number of recommendations", min_value=1, max_value=10, value=3, step=1)

if st.button("Run DecisionFlow"):
    with st.spinner("Running DecisionFlow..."):
        try:
            result = run_decisionflow_for_client(client_id, topk=topk, use_hybrid=use_hybrid)
        except ValueError as e:
            st.error(f"Client lookup failed: {e}")
            st.stop()
        except FileNotFoundError as e:
            st.error(f"Dataset not found: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            st.stop()
    # Display client profile
    st.subheader("Client Profile")
    profile = result.get("client_profile")
    st.json({
        "client_id": profile.client_id,
        "segment": profile.segment,
        "current_products": profile.current_products,
        "data_quality": profile.data_quality,
    })
    # Display recommendations
    st.subheader("Top Recommendations")
    st.write(result.get("recommendations", []))
    # Display risk
    st.subheader("Risk Assessment")
    st.json(result.get("risk", {}))
    # Display explanations
    st.subheader("Explanations")
    explanations = result.get("explanations", [])
    if explanations:
        for expl in explanations:
            st.markdown(f"**{expl.product}**")
            for reason in expl.reasons:
                st.markdown(f"- {reason}")
            if expl.limitations:
                st.markdown("_Limitations:_")
                for lim in expl.limitations:
                    st.markdown(f"  - {lim}")
    else:
        st.write("No explanations available.")
    # Display raw scores optionally
    with st.expander("Show Raw Scores"):
        st.json(result.get("raw_scores", {}))
