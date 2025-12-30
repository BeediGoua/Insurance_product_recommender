import streamlit as st
import pandas as pd
import numpy as np
from src.config import ARTIFACTS_DIR, BASELINE_VERSION
from src.pipelines.baseline_pipeline import BaselineArtifact, recommend_from_selection

st.set_page_config(page_title="Simulator - Zimnat", layout="wide")

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load artifact
artifact_path = ARTIFACTS_DIR / BASELINE_VERSION
try:
    artifact = BaselineArtifact.load(artifact_path)
except FileNotFoundError:
    st.error("Artefact non trouvé. Veuillez lancer le script d'entraînement.")
    st.stop()

st.title("Simulateur Interactif")
st.markdown("Construisez un panier de produits. Le modèle mettra à jour ses recommandations en temps réel.")

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("1. Panier Client")
    st.caption("Sélectionnez les produits déjà détenus par ce client fictif.")
    
    # Multiselect is cleaner than grid for many items
    all_products = artifact.product_cols
    owned = st.multiselect(
        "Produits détenus :", 
        all_products,
        default=[],
        placeholder="Ajoutez des produits..."
    )
    
    st.info(f"Produits dans le panier : {len(owned)}")
    
    if owned:
        st.success("Panier actif")
    else:
        st.warning("Panier vide. Ajoutez un produit pour voir les effets.")

with col_right:
    st.subheader("2. Recommandations (Temps Réel)")
    
    if not owned:
        st.write("En attente de produits...")
    else:
        # Instant Prediction
        topk_slider = st.slider("Nombre de recommandations", 1, 10, 5)
        recs = recommend_from_selection(artifact, owned_products=owned, topk=topk_slider)
        
        # Display Results
        st.markdown("##### Top Produits Suggérés")
        
        # Visualization loop
        for p, score in recs.items():
            c1, c2 = st.columns([3, 1])
            c1.progress(min(score, 1.0), text=f"**{p}**")
            c2.caption(f"Score: {score:.4f}")
            
        # --- EXPLAINABILITY (Focus on Top 1) ---
        st.divider()
        top_prod = recs.index[0]
        top_score = recs.iloc[0]
        
        st.subheader(f"3. Pourquoi '{top_prod}' ?")
        
        # Logic copied/adapted from Inspector
        prod_map = {name: i for i, name in enumerate(artifact.product_cols)}
        t_idx = prod_map.get(top_prod)
        
        if t_idx is not None:
            # 1. Weights Calculation
            # Filter owned to valid indices
            input_indices = [prod_map.get(p) for p in owned if prod_map.get(p) is not None]
            
            if input_indices:
                supports = artifact.support_A[input_indices]
                total_support = supports.sum()
                weights = supports / total_support if total_support > 0 else np.ones(len(supports))/len(supports)
                
                # 2. Build Explanation Data (Product, Weight, Prob)
                explanation_data = []
                for i, p_name in enumerate(owned):
                    if prod_map.get(p_name) is not None:
                        p_idx = prod_map.get(p_name)
                        w = weights[i]
                        prob = artifact.cond[p_idx, t_idx]
                        contribution = prob * w
                        explanation_data.append({
                            "Produit Source": p_name,
                            "Poids (Rareté)": w,
                            "Probabilité P(T|S)": prob,
                            "Contribution": contribution
                        })
                
                df_explain = pd.DataFrame(explanation_data).sort_values("Contribution", ascending=False)
                
                # Dynamic Text
                top_driver = df_explain.iloc[0]["Produit Source"]
                top_prob = df_explain.iloc[0]["Probabilité P(T|S)"]
                
                if top_prob > 0.7:
                    st.success(f"Lien très fort : Le produit **{top_driver}** implique presque toujours **{top_prod}** (Prob: {top_prob:.0%}).")
                elif top_prob > 0.4:
                    st.info(f"Influence majeure : **{top_driver}** est le facteur principal.")
                else:
                    st.write(f"Influence diffuse : **{top_prod}** est suggéré par une combinaison de facteurs faibles.")

                with st.expander("Voir le détail du calcul (Mathématiques)"):
                    st.latex(r"Score(Target) = \sum_{S \in Panier} P(Target | S) \times Poids(S)")
                    st.dataframe(
                        df_explain.style.format("{:.2%}", subset=["Poids (Rareté)", "Probabilité P(T|S)", "Contribution"])
                                        .background_gradient(cmap="Greens", subset=["Contribution"]),
                        use_container_width=True
                    )
                    st.caption(f"Score Total = {df_explain['Contribution'].sum():.4f}")
