
import streamlit as st
import pandas as pd
import numpy as np
from src.inference import load_baseline, get_recommendations

st.set_page_config(page_title="Simulator - Zimnat", layout="wide")

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- SIDEBAR MODEL SELECTION ---
model_choice = st.sidebar.radio("Modèle", ["Baseline (Stats)", "CatBoost V1 (Hybrid)"])

# Option de Comparaison (Uniquement pour CatBoost pour l'instant)
show_comparison = False
if model_choice.startswith("CatBoost"):
    st.sidebar.markdown("---")
    show_comparison = st.sidebar.checkbox("🧪 Mode Comparaison Avancée", value=False, help="Voir l'effet du paramètre Alpha (Mélange IA/Stats)")

# Load Resources (Pre-fetch for product list)
baseline = load_baseline()
if not baseline:
    st.error("Baseline non trouvée. Lancez run_baseline.py")
    st.stop()
    
# Get Product List
all_products = baseline.product_cols

st.title("Simulateur Interactif")
st.markdown(f"**Modèle Actif : {model_choice}**")
st.markdown("Construisez un panier et un profil. Le modèle mettra à jour ses recommandations en temps réel.")

col_left, col_right = st.columns([1, 1.5], gap="large")

with col_left:
    st.subheader("1. Panier Client")
    st.caption("Sélectionnez les produits déjà détenus par ce client fictif.")
    
    owned = st.multiselect(
        "Produits détenus :", 
        all_products,
        default=[],
        placeholder="Ajoutez des produits..."
    )
    
    st.info(f"Produits dans le panier : {len(owned)}")
    
    # --- PROFILE FORM (Only for CatBoost) ---
    user_features = {}
    if model_choice.startswith("CatBoost"):
        st.subheader("2. Profil Client (Requis pour l'IA)")
        with st.expander("Détails du Profil Client", expanded=True):
            sex = st.selectbox("Sexe", ["M", "F"], index=0)
            status = st.selectbox("Statut Marital", ["M", "U", "S", "W", "D", "P", "R", "f"], index=0)
            age = st.slider("Âge", 18, 90, 35)
            
            # Robust Dummies for all CAT_FEATURES
            user_features = {
                "sex": sex,
                "marital_status": status,
                "age": age,
                "birth_year": 2020 - age, 
                "join_date": "2018-01-01", 
                "join_year": 2018, # Feature engineered
                
                # Dummy values representing a "Standard" profile for technical fields we hide from UI
                "branch_code": "74280b18",
                "occupation_code": "2a7c15d9",
                "occupation_category_code": "T4"
            }
            # st.caption("Note: Agence, Métier et Catégorie sont fixés à des valeurs standards pour cette simulation.")
            
    if owned:
        st.success("Panier actif")
    else:
        st.warning("Panier vide. Ajoutez un produit pour voir les effets.")

with col_right:
    st.subheader("3. Recommandations (Temps Réel)")
    
    # Prepare Context
    context = {
        "owned_products": owned,
        "user_features": user_features
    }
    
    if not owned:
        st.write("En attente de produits...")
    else:
        topk_slider = st.slider("Nombre de recommandations", 1, 10, 5)
        
        # --- MODE COMPARAISON ---
        if show_comparison and model_choice.startswith("CatBoost"):
            c1, c2, c3 = st.columns(3)
            
            # 1. Pure AI (Alpha=0)
            with c1:
                st.markdown("##### 🤖 IA Pure (α=0)")
                st.caption("Profil uniquement")
                recs_ai = get_recommendations("CatBoost", context, topk=topk_slider, alpha_override=0.0)
                for p, s in recs_ai.items():
                    st.progress(min(float(s), 1.0), text=f"**{p}** ({s:.2f})")
            
            # 2. Hybrid (Alpha=0.5)
            with c2:
                st.markdown("##### ⚖️ Hybride (α=0.5)")
                st.caption("Mélange 50/50")
                recs_mix = get_recommendations("CatBoost", context, topk=topk_slider, alpha_override=0.5)
                for p, s in recs_mix.items():
                    st.progress(min(float(s), 1.0), text=f"**{p}** ({s:.2f})")
            
            # 3. Stats (Alpha=1.0)
            with c3:
                st.markdown("##### 📊 Stats (α=1.0)")
                st.caption("Panier uniquement")
                recs_stats = get_recommendations("CatBoost", context, topk=topk_slider, alpha_override=1.0)
                for p, s in recs_stats.items():
                    st.progress(min(float(s), 1.0), text=f"**{p}** ({s:.2f})")
                    
            st.divider()
            st.info("Observez comment l'IA (gauche) peut proposer des produits différents des Stats (droite) pour des profils atypiques.")

        # --- MODE STANDARD ---
        else:
            # Prediction Standard (Alpha optimisé par défaut)
            recs = get_recommendations(model_choice, context, topk=topk_slider)
            
            if recs.empty:
                st.warning("Pas de recommandation disponible.")
            else:
                st.markdown("##### Top Produits Suggérés")
                for p, score in recs.items():
                    c1_disp, c2_disp = st.columns([3, 1])
                    c1_disp.progress(min(float(score), 1.0), text=f"**{p}**")
                    c2_disp.caption(f"Score: {score:.4f}")
                
                # Explainability Block (Baseline Only)
                if model_choice.startswith("Baseline"):
                    st.divider()
                    top_prod = recs.index[0]
                    st.subheader(f"3. Pourquoi '{top_prod}' ? (Explication Stats)")
                    
                    artifact = baseline
                    prod_map = {name: i for i, name in enumerate(artifact.product_cols)}
                    t_idx = prod_map.get(top_prod)
                    
                    if t_idx is not None:
                        input_indices = [prod_map.get(p) for p in owned if prod_map.get(p) is not None]
                        if input_indices:
                            supports = artifact.support_A[input_indices]
                            total_support = supports.sum()
                            weights = supports / total_support if total_support > 0 else np.ones(len(supports))/len(supports)
                            
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
                            st.dataframe(
                                df_explain.style.format("{:.2%}", subset=["Poids (Rareté)", "Probabilité P(T|S)", "Contribution"])
                                                .background_gradient(cmap="Greens", subset=["Contribution"]),
                                use_container_width=True
                            )
                elif not show_comparison:
                     st.divider()
                     st.info("Note : Le modèle Hybride utilise une combinaison complexe. Activez le mode Comparaison pour voir l'influence de l'IA vs Stats.")
