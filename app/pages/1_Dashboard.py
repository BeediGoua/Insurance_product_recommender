import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from src.config import OUT_DIR

st.set_page_config(layout="wide", page_title="Dashboard KPI")

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Performance du Modèle Baseline")

# --- GLOBAL METRICS ---
try:
    metrics = pd.read_json(OUT_DIR / "metrics.json", typ='series')
    c1, c2, c3 = st.columns(3)
    c1.metric("Hit@1 (Précision Top-1)", f"{metrics['Hit@1']:.1%}")
    c2.metric("Hit@5 (Précision Top-5)", f"{metrics['Hit@5']:.1%}")
    c3.metric("MRR (Qualité Rang)", f"{metrics['MRR']:.3f}")
except ValueError:
    st.error(f"Fichier de métriques introuvable ou invalide dans {OUT_DIR}")

# --- FILTERS ---
with st.sidebar:
    st.header("Filtres")
    try:
        df_granular = pd.read_parquet(OUT_DIR / "granular_metrics.parquet")
        
        # Unique values
        u_sex = sorted(df_granular["sex"].astype(str).unique().tolist())
        u_mar = sorted(df_granular["marital_status"].astype(str).unique().tolist())
        u_occ = sorted(df_granular["occupation_category_code"].astype(str).unique().tolist())
        
        sel_sex = st.multiselect("Sexe", u_sex, default=u_sex)
        sel_mar = st.multiselect("Statut Marital", u_mar, default=u_mar)
        sel_occ = st.multiselect("Catégorie Pro", u_occ, default=u_occ)
        
        # Apply Filters
        mask = (
            df_granular["sex"].astype(str).isin(sel_sex) & 
            df_granular["marital_status"].astype(str).isin(sel_mar) & 
            df_granular["occupation_category_code"].astype(str).isin(sel_occ)
        )
        df_filtered = df_granular[mask]
        
        st.metric("Clients analysés", f"{len(df_filtered)} / {len(df_granular)}")
        
    except FileNotFoundError:
        st.warning("Données détaillées manquantes. Filtres désactivés.")
        df_filtered = None

# --- DYNAMIC AGGREGATION ---
if df_filtered is not None and not df_filtered.empty:
    # 1. By Basket Size
    # Groupby 'basket_size' and compute mean of is_hit1, is_hit5, mrr
    df_basket = df_filtered.groupby("basket_size")[["is_hit1", "is_hit5", "mrr"]].mean()
    df_basket.columns = ["Hit@1", "Hit@5", "MRR"]
    
    # 2. By Product
    df_prod = df_filtered.groupby("target_product")[["is_hit1", "is_hit5", "mrr"]].mean()
    df_prod.columns = ["Hit@1", "Hit@5", "MRR"]
    df_prod["Count"] = df_filtered["target_product"].value_counts()
    
else:
    # Fallback/Empty
    df_basket = pd.DataFrame(columns=["Hit@1", "Hit@5", "MRR"])
    df_prod = pd.DataFrame(columns=["Hit@1", "Hit@5", "MRR"])


# --- SECTION 1: Performance par Taille de Panier ---
st.markdown("---")
st.subheader("1. Analyse par Taille de Panier (Filtrée)")

# Filtre Interactif (Visual)
all_sizes = sorted(df_basket.index.unique().tolist())
if all_sizes:
    sel_sizes = st.multiselect(
        "Focus tailles de panier :", 
        all_sizes, 
        default=all_sizes
    )
    
    if sel_sizes:
        st.dataframe(
            df_basket.loc[sel_sizes].style.format("{:.1%}", subset=["Hit@1", "Hit@5", "MRR"]), 
            use_container_width=True
        )
        st.download_button(
            label="Télécharger les données (CSV)",
            data=df_basket.loc[sel_sizes].to_csv().encode('utf-8'),
            file_name='basket_metrics.csv',
            mime='text/csv',
        )
else:
    st.warning("Aucune donnée pour ces filtres.")

# --- SECTION 2: Top / Flop Produits ---
st.markdown("---")
st.subheader("2. Performance par Produit (Difficulté)")

if not df_prod.empty:
    # Contrôles
    c1, c2 = st.columns([1, 2])
    with c1:
        mode = st.radio("Afficher :", ["Les plus difficiles (Flop)", "Les plus faciles (Top)"], index=0)
        ascending = True if "Flop" in mode else False
    with c2:
        top_n = st.slider("Nombre de produits :", 5, 50, 10, 5)
        
    # Tri et Affichage
    # Filter out products with low count if needed to avoid noise? Keep simple for now.
    df_sorted = df_prod.sort_values("MRR", ascending=ascending).head(top_n)
    st.dataframe(
        df_sorted.style.format("{:.1%}", subset=["Hit@1", "Hit@5", "MRR"])
                 .background_gradient(cmap="RdYlGn", subset=["MRR"]), 
        use_container_width=True
    )
    st.download_button(
        label="Télécharger les données (CSV)",
        data=df_sorted.to_csv().encode('utf-8'),
        file_name='product_metrics.csv',
        mime='text/csv',
    )


# ... (Previous code remains)

# --- SECTION 3: Matrice de Co-occurrence (Heatmap) ---
st.markdown("---")
st.subheader("3. Matrice de Co-occurrence (Relations Produits)")

# Load RAW Data for matrix calculation (Cached)
@st.cache_data
def load_raw_data():
    # Only needed columns to save memory
    return pd.read_parquet(OUT_DIR / "train_cleaned.parquet")

try:
    df_raw = load_raw_data()
    
    if df_filtered is not None and not df_filtered.empty:
        valid_indices = df_filtered["original_idx"].unique()
        df_subset = df_raw.loc[valid_indices]
        
        meta_cols = ["ID", "join_date", "sex", "marital_status", "birth_year", "branch_code", "occupation_code", "occupation_category_code", "join_year", "age_raw", "age", "join_year_missing", "age_missing", "age_was_clipped"]
        prod_cols = [c for c in df_subset.columns if c not in meta_cols]
        
        if not prod_cols:
            st.error("Impossible d'identifier les colonnes produits.")
        else:
            # Calculation
            X = df_subset[prod_cols].fillna(0).astype(int)
            co_matrix = X.T.dot(X)
            
            diag = np.diag(co_matrix)
            with np.errstate(divide='ignore', invalid='ignore'):
                cond_matrix = co_matrix.div(diag, axis=0).fillna(0)
            
            # Interactive Plot using Plotly
            fig = px.imshow(
                cond_matrix,
                labels=dict(x="Produit Cible (Recommandé)", y="Produit Possédé (Contexte)", color="Probabilité"),
                x=cond_matrix.columns,
                y=cond_matrix.index,
                color_continuous_scale="Reds",
                aspect="auto" # Square cells not enforced allows better fit
            )
            
            # Update tooltips formatting
            fig.update_traces(
                hovertemplate="Si j'ai : %{y}<br>Alors j'ai aussi : %{x}<br>Probabilité : %{z:.1%}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("Astuce : Passez la souris sur une case pour voir le détail exact de la probabilité.")

    else:
        st.warning("Filtres trop restrictifs (vide).")

except FileNotFoundError:
    st.warning("Données brutes introuvables (train_cleaned.parquet).")
