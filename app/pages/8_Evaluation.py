import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.evaluation.eval_runner import run_all_evaluations
from src.decisionflow.profile_builder import build_client_profile


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

if not bench_path.exists() or not expected_path.exists():
    st.warning("Evaluation datasets not found.  Please provide benchmark and expected files in data/evaluation.")
else:
    bench_df = pd.read_csv(bench_path)
    expected_df = pd.read_csv(expected_path)
    # Merge on client_id
    merged = pd.merge(expected_df, bench_df, on="client_id", how="left")
    dataset = []
    for _, row in merged.iterrows():
        client_id = row["client_id"]
        true_product = row["expected_product"]
        # Build a dict of the row excluding the expected columns
        raw = row.drop(["client_id", "expected_product", "acceptable_products", "forbidden_products", "reason"]).to_dict()
        dataset.append((client_id, true_product, raw))
    if st.button("Run Evaluation"):
        with st.spinner("Evaluating..."):
            metrics = run_all_evaluations(dataset)
        st.subheader("Recommender Metrics")
        st.json(metrics.get("recommender", {}))
        st.subheader("Policy Metrics")
        st.json(metrics.get("policy", {}))
        st.subheader("Explanation Metrics")
        st.json(metrics.get("explanation", {}))
