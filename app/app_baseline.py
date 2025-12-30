import streamlit as st
from pathlib import Path
from src.pipelines.baseline_pipeline import BaselineArtifact, recommend_from_selection

st.title("Zimnat — Baseline Basket Completion (Simulateur client)")

from src.config import ARTIFACTS_DIR, BASELINE_VERSION
artifact_path = ARTIFACTS_DIR / BASELINE_VERSION
artifact = BaselineArtifact.load(artifact_path)

st.write("Sélectionne les produits que le client possède :")
owned = []
cols = st.columns(3)
for i, p in enumerate(artifact.product_cols):
    with cols[i % 3]:
        if st.checkbox(p, value=False):
            owned.append(p)

topk = st.slider("Top-K", 1, 10, 5)

if st.button("Recommander"):
    recs = recommend_from_selection(artifact, owned_products=owned, topk=topk)
    st.dataframe(recs.reset_index().rename(columns={"index": "produit", 0: "score"}))
