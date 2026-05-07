import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.evaluation.eval_runner import run_all_evaluations, EvalCase


st.set_page_config(page_title="Evaluation", layout="wide", initial_sidebar_state="expanded")
st.title("Evaluation Dashboard")
st.markdown("""
This page runs offline evaluation of the recommendation system using
benchmark datasets.  It computes standard ranking metrics (Hit@K,
MRR), policy compliance rates and explanation statistics.

To add your own evaluation sets, place CSV files in
``data/evaluation`` and adjust the loading logic below.
""")

# Load evaluation data
bench_path = Path("data/evaluation/benchmark_clients.csv")
expected_path = Path("data/evaluation/expected_recommendations.csv")
agent_logs_path = Path("data/evaluation/agent_conversations.json")

if not bench_path.exists() or not expected_path.exists():
    st.warning("Evaluation datasets not found.  Please provide benchmark and expected files in data/evaluation.")
else:
    bench_df = pd.read_csv(bench_path)
    expected_df = pd.read_csv(expected_path)

    def _parse_product_set(value) -> set[str]:
        if pd.isna(value):
            return set()
        items = [x.strip() for x in str(value).split(";") if x and x.strip()]
        return set(items)

    # Merge on client_id
    merged = pd.merge(expected_df, bench_df, on="client_id", how="left")
    dataset = []
    for _, row in merged.iterrows():
        client_id = row["client_id"]
        true_product = row["expected_product"]
        acceptable = _parse_product_set(row.get("acceptable_products"))
        forbidden = _parse_product_set(row.get("forbidden_products"))
        reason = row.get("reason")
        # Build a dict of the row excluding the expected columns
        raw = row.drop(["client_id", "expected_product", "acceptable_products", "forbidden_products", "reason"]).to_dict()
        dataset.append(
            EvalCase(
                client_id=client_id,
                true_product=true_product,
                raw_row=raw,
                acceptable_products=acceptable,
                forbidden_products=forbidden,
                reason=reason,
            )
        )
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            agent_conversations = []
            if agent_logs_path.exists():
                try:
                    with open(agent_logs_path, "r", encoding="utf-8") as f:
                        agent_conversations = json.load(f)
                except Exception:
                    agent_conversations = []
            metrics = run_all_evaluations(dataset, agent_conversations=agent_conversations)

        st.subheader("Business KPI")
        st.json(metrics.get("business_kpi", {}))

        st.subheader("Recommender Metrics")
        st.json(metrics.get("recommender", {}))
        st.subheader("Policy Metrics")
        st.json(metrics.get("policy", {}))
        st.subheader("Explanation Metrics")
        st.json(metrics.get("explanation", {}))
        st.subheader("Agent Metrics")
        st.json(metrics.get("agent", {}))
