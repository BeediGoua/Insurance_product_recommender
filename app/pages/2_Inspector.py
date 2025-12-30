import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from src.config import ARTIFACTS_DIR, BASELINE_VERSION, TRAIN_PATH
from src.pipelines.baseline_pipeline import BaselineArtifact, recommend_from_selection
from src.data.io import load_train_test

st.set_page_config(page_title="Inspector - Zimnat", layout="wide")

@st.cache_resource
def load_resources():
    artifact = BaselineArtifact.load(ARTIFACTS_DIR / BASELINE_VERSION)
    # Load CLEANED train data for inspection (consistent with model)
    df = pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")
    return artifact, df

artifact, df_train = load_resources()

st.title("Inspecteur de Client")
st.markdown("Analysez un client réel de la base d'entraînement et testez la capacité du modèle à deviner un produit caché.")

col_search, col_action = st.columns([1, 2])
with col_search:
    # Fix persistence: bind value to session_state
    default_id = st.session_state.get("client_id", "")
    client_id_input = st.text_input("Rechercher par ID (ex: ID_XXXX)", value=default_id, placeholder="ID_...")
    
    # Update session state if input changes manually
    if client_id_input != default_id:
        st.session_state["client_id"] = client_id_input
        client_id = client_id_input
    else:
        client_id = default_id

    st.markdown("---")
    st.write("**Client Aléatoire**")
    
    # 1. Advanced Filters UI
    with st.expander("Filtres Avancés (Optionnel)"):
        # Get unique values for dropdowns (sorted)
        unique_sex = sorted(df_train["sex"].dropna().unique().tolist())
        unique_marital = sorted(df_train["marital_status"].dropna().unique().tolist())
        unique_occ_cat = sorted(df_train["occupation_category_code"].dropna().unique().tolist())
        
        f_sex = st.selectbox("Sexe", ["Tous"] + unique_sex)
        f_marital = st.selectbox("Statut Marital", ["Tous"] + unique_marital)
        f_occ = st.selectbox("Catégorie Pro", ["Tous"] + unique_occ_cat)

    target_size = st.number_input("Taille de panier cible :", min_value=2, max_value=20, value=3)
    
    if st.button(f"Trouver un profil type ({target_size} pdts)"):
        # We need product cols. Using artifact ones.
        prod_cols = artifact.product_cols
        
        # Start with full dataframe
        pool = df_train.copy()
        
        # Apply Filters
        if f_sex != "Tous":
            pool = pool[pool["sex"] == f_sex]
        if f_marital != "Tous":
            pool = pool[pool["marital_status"] == f_marital]
        if f_occ != "Tous":
            pool = pool[pool["occupation_category_code"] == f_occ]
            
        if pool.empty:
            st.error("Aucun client ne correspond à ces critères (Sexe/Marital/Pro).")
        else:
            # Apply Basket Size Logic on the filtered pool
            basket_sizes = pool[prod_cols].sum(axis=1)
            
            # Exact match
            mask = basket_sizes == target_size
            valid_pool = pool[mask]
            
            # Fallback (+/- 1 product)
            if valid_pool.empty:
                 mask = (basket_sizes >= target_size - 1) & (basket_sizes <= target_size + 1)
                 valid_pool = pool[mask]
            
            if not valid_pool.empty:
                client_sample = valid_pool.sample(1).iloc[0]
                found_id = client_sample["ID"]
                st.session_state["client_id"] = found_id
                st.rerun()
            else:
                st.warning(f"Des clients existent avec ces critères démographiques, mais aucun n'a ~{target_size} produits. Essayez une autre taille.")

if client_id:
    client_row = df_train[df_train["ID"] == client_id]
    if not client_row.empty:
        client_data = client_row.iloc[0]
        
        # 1. Profile Display
        st.subheader(f"Profil : {client_id}")
        c1, c2, c3, c4 = st.columns(4)
        c1.info(f"Sexe: {client_data.get('sex', '?')}")
        c2.info(f"Age: {client_data.get('age', '?')}") # Pre-calc in cleaning
        c3.info(f"Métier: {client_data.get('occupation_code', '?')}")
        c4.info(f"Agence: {client_data.get('branch_code', '?')}")

        # 2. Products
        owned_mask = client_data[artifact.product_cols] == 1
        real_products = owned_mask[owned_mask].index.tolist()
        
        st.write("**Produits détenus (Réel / Vérité Terrain) :**")
        
        # Manual Input Selection
        st.info("Simulation Interactive : Décochez les produits que vous voulez CACHER au modèle.")
        
        # We start with all checked (all known)
        input_products = []
        c_cols = st.columns(4)
        for i, p in enumerate(real_products):
            with c_cols[i % 4]:
                # Default true = Model sees it. False = Model must guess it.
                if st.checkbox(p, value=True, key=f"mask_{p}"):
                    input_products.append(p)
        
        # Identify what was hidden (Targets)
        targets = [p for p in real_products if p not in input_products]
        
        if len(input_products) == len(real_products):
            st.warning("Vous donnez TOUS les produits au modèle. Il n'a rien à deviner ! Décochez-en au moins un.")
        else:
            st.divider()
            st.subheader("Résultat de la Prédiction")
            st.write(f"**Donné au modèle :** {input_products}")
            st.write(f"**À deviner (Masqué) :** `{targets}`")

            if st.button("Lancer la Prédiction"):
                # Predict
                recs = recommend_from_selection(artifact, input_products, topk=10)
                
                # Check ranks for each target
                prod_map = {name: i for i, name in enumerate(artifact.product_cols)}
                
                for t in targets:
                    rank = -1
                    if t in recs.index:
                        rank = recs.index.get_loc(t) + 1
                    
                    # 1. Visual Status
                    if rank == 1:
                        st.success(f"**{t}** : Trouvé en position #1 ! (Bravo)")
                        st.progress(1.0, text="Score : 100%")
                    elif 1 < rank <= 5:
                        st.success(f"**{t}** : Trouvé en Top-5 (Pos #{rank})")
                        st.progress(1.0/rank, text=f"Score : {100/rank:.0f}%")
                    elif rank > 5:
                        st.warning(f"**{t}** : Trouvé mais loin (Pos #{rank})")
                    else:
                        st.error(f"**{t}** : Non trouvé dans le Top-10.")

                    # 2. Explainability (WHY?)
                    if rank != -1: 
                        t_idx = prod_map.get(t)
                        with st.expander(f"Pourquoi le modèle propose {t} ?", expanded=True):
                            drivers = []
                            for p_in in input_products:
                                p_in_idx = prod_map.get(p_in)
                                if p_in_idx is not None and t_idx is not None:
                                    prob = artifact.cond[p_in_idx, t_idx]
                                    drivers.append((p_in, prob))
                            
                            drivers.sort(key=lambda x: x[1], reverse=True)
                            
                            if drivers:
                                top_driver, top_prob = drivers[0]
                                
                                # --- Dynamic Interpretation ---
                                st.markdown("##### Interprétation")
                                if top_prob > 0.7:
                                    st.write(f"Recommandation très forte. Dans l'historique, la quasi-totalité des clients ayant **{top_driver}** finissent par souscrire à **{t}**.")
                                elif top_prob > 0.4:
                                    st.write(f"Lien logique solide. Le produit **{top_driver}** est souvent un précurseur de l'achat de **{t}**.")
                                else:
                                    st.write(f"Opportunité de vente croisée (Cross-sell). Combinaison suggérant un intérêt pour **{t}**.")
                                
                                # --- Detailed Formula Application ---
                                st.markdown("##### Détails du Calcul (Preuve)")
                                st.write("Le score final est une moyenne pondérée par la popularité (Support) des produits possédés :")
                                st.latex(r"Score(T) = \sum_{i \in Input} P(T | Product_i) \times \frac{Support(i)}{\sum Support}")
                                
                                # 1. Calculate Weights
                                input_indices = [prod_map.get(p) for p in input_products if prod_map.get(p) is not None]
                                if input_indices:
                                    supports = artifact.support_A[input_indices]
                                    total_support = supports.sum()
                                    weights = supports / total_support if total_support > 0 else [1/len(supports)] * len(supports)
                                    
                                    # 2. Build Calculation String
                                    calc_str = []
                                    total_score = 0
                                    
                                    # We iterate over INPUTS to show contribution of each
                                    sorted_inputs = sorted(zip(input_products, weights, input_indices), key=lambda x: x[1], reverse=True)
                                    
                                    for p_name, w, p_idx in sorted_inputs:
                                        # Prob P(T|I)
                                        prob = artifact.cond[p_idx, t_idx]
                                        contrib = prob * w
                                        total_score += contrib
                                        calc_str.append(f"({prob:.2f} * {w:.2f})")
                                        
                                    calc_display = " + ".join(calc_str)
                                    st.code(f"Score({t}) = {calc_display}\n          = {total_score:.6f}", language="text")
                                    st.caption(f"Note: {t} reçoit une contribution de chaque produit possédé, pondérée par sa rareté/fréquence.")


                                # ... (Previous explainability code) ...
                                st.caption(f"Note: {t} reçoit une contribution de chaque produit possédé, pondérée par sa rareté/fréquence.")
                            else:
                                st.write("Pas de produits déclencheurs identifiés.")
                            
                            # --- 3. RADAR CHART (Profile Fit) ---
                            st.markdown("##### " + f"Analyse de Profil : Est-il un client type pour '{t}' ?")
                            
                            # A. Get population of existing owners of 't'
                            owners = df_train[df_train[t] == 1]
                            if not owners.empty:
                                # B. Calculate Axes
                                
                                # 1. Age Consistency
                                avg_age = owners["age"].mean()
                                client_age = client_data.get("age", avg_age)
                                # Score: 1.0 - normalized distance. Max diff ~30 years?
                                age_diff = abs(client_age - avg_age)
                                age_score = max(0, 1 - (age_diff / 30.0))
                                
                                # 2. Demographic Match (Sex + Marital)
                                top_sex = owners["sex"].mode()[0]
                                top_mar = owners["marital_status"].mode()[0]
                                match_count = 0
                                if client_data.get("sex") == top_sex: match_count += 0.5
                                if client_data.get("marital_status") == top_mar: match_count += 0.5
                                demo_score = match_count
                                
                                # 3. Occupation Match
                                # Does the client work in a top occupation for this product?
                                top_occs = owners["occupation_category_code"].value_counts().nlargest(3).index.tolist()
                                occ_score = 1.0 if client_data.get("occupation_category_code") in top_occs else 0.3
                                
                                # 4. Basket Similarity (Context)
                                # Do owners of T also have what I have?
                                # Check overlap of current input_products with average owner basket
                                # Simplified: What % of my current products are commonly held by owners of T?
                                # (Coverage of usage)
                                if input_products:
                                    # Frequency of my products in owners population
                                    # Mean ownership rate of my products among T-owners
                                    relevance = owners[input_products].mean().mean() 
                                    # relevance is between 0 and 1. 
                                    # If 1, everyone who has T has all my products.
                                else:
                                    relevance = 0.5
                                basket_score = relevance
                                
                                # Plot Radar
                                categories = ['Age Compatible', 'Démographie (Sexe/Marital)', 'Métier (Pro)', 'Cohérence Panier']
                                values = [age_score, demo_score, occ_score, basket_score]
                                # Close the loop
                                values += [values[0]]
                                categories += [categories[0]]
                                
                                fig = go.Figure()
                                fig.add_trace(go.Scatterpolar(
                                      r=values,
                                      theta=categories,
                                      fill='toself',
                                      name='Adéquation Profil',
                                      line_color='#E63946'
                                ))
                                fig.update_layout(
                                  polar=dict(
                                    radialaxis=dict(
                                      visible=True,
                                      range=[0, 1]
                                    )),
                                  margin=dict(l=20, r=20, t=20, b=20),
                                  height=300,
                                  showlegend=False
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                st.caption(f"Comparaison avec {len(owners)} clients possédant déjà {t}.")
                            else:
                                st.write("Pas assez de données historiques pour comparer.")

# (Rest of file)
                 # Show dataframe
                 df_recs = recs.reset_index().rename(columns={"index": "Produit", 0: "Score"})

                # Show dataframe
                df_recs = recs.reset_index().rename(columns={"index": "Produit", 0: "Score"})
                
                def highlight_targets(row):
                    return ['background-color: #d4edda' if row.Produit in targets else '' for _ in row]
                
                st.dataframe(df_recs.style.apply(highlight_targets, axis=1), height=400)

    else:
        st.error(f"Client ID {client_id} introuvable dans le Train set.")

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
