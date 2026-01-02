import sys
from pathlib import Path

# Fix Path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.config import ARTIFACTS_DIR, BASELINE_VERSION

st.set_page_config(layout="wide", page_title="Business Analytics - Zimnat")

# --- CSS ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Business Analytics Dashboard")
st.markdown("Strategic Insights on Portfolio, Demographics, and Sales Performance.")

# --- LOAD DATA ---
@st.cache_data
def load_biz_data():
    df = pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")
    
    # Identify Product Columns
    meta_cols = ["ID", "join_date", "sex", "marital_status", "birth_year", "branch_code", "occupation_code", "occupation_category_code", "join_year", "age_raw", "age", "join_year_missing", "age_missing", "age_was_clipped"]
    prod_cols = [c for c in df.columns if c not in meta_cols]
    
    # Pre-calc metrics
    df["basket_size"] = df[prod_cols].sum(axis=1)
    return df, prod_cols

try:
    df, prod_cols = load_biz_data()
except Exception as e:
    st.error(f"Data loading error: {e}")
    st.stop()

# --- 1. KPI CARDS ---
st.header("1. Portfolio Overview")

c1, c2, c3, c4 = st.columns(4)

total_clients = len(df)
avg_basket = df["basket_size"].mean()
top_prod = df[prod_cols].sum().idxmax()
top_prod_count = df[prod_cols].sum().max()
active_branches = df["branch_code"].nunique()

c1.metric("Total Client Base", f"{total_clients:,}")
c2.metric("Avg Products / Client", f"{avg_basket:.2f}", help="Cross-Sell Density")
c3.metric("Top Product", top_prod, f"Owned by {top_prod_count/total_clients:.1%}")
c4.metric("Active Branches", active_branches)

st.divider()

# --- 2. DEMOGRAPHICS & SEGMENTATION ---
st.header("2. Client Segmentation")

c_demo1, c_demo2 = st.columns(2)

with c_demo1:
    st.subheader("Occupation Landscape")
    # Treemap of Occupation Categories
    df_occ = df["occupation_category_code"].value_counts().reset_index()
    df_occ.columns = ["Category", "Count"]
    
    fig_tree = px.treemap(
        df_occ, 
        path=["Category"], 
        values="Count",
        title="Distribution by Job Category",
        color="Count",
        color_continuous_scale="Blues"
    )
    st.plotly_chart(fig_tree, use_container_width=True)

with c_demo2:
    st.subheader("Branch Performance (High Value)")
    # Branch performance by Avg Basket Size
    df_branch = df.groupby("branch_code")["basket_size"].mean().reset_index()
    # Filter top 10
    df_branch_top = df_branch.sort_values("basket_size", ascending=False).head(10)
    
    fig_bar = px.bar(
        df_branch_top,
        x="branch_code",
        y="basket_size",
        title="Top 10 Branches by Avg Basket Size",
        color="basket_size",
        color_continuous_scale="Viridis",
        labels={"basket_size": "Avg Products per Client"}
    )
    st.plotly_chart(fig_bar, use_container_width=True)

st.divider()

# --- 3. PRODUCT SYNERGIES (SUNBURST) ---
st.header("3. Product Bundles & Synergies")
st.markdown("Visualizing how products naturally group together in client portfolios.")

c_sun, c_heat = st.columns([1, 1])

with c_sun:
    st.subheader("Ownership Hierarchy (Sunburst)")
    # Create simple hierarchy: Most common product -> Second most common
    # This is a simplification for visual effect
    
    # Strategy: Take population, group by Top 2 products owned
    # We find the 'primary' product for each user (their first owned in list order, or random owned)
    # Ideally: Sort products by popularity global, then assign Primary/Secondary locally
    
    # Quick visual hack:
    # 1. Expand dataframe to (User, Product)
    # 2. But Sunburst needs hierarchy. Let's do:
    #    Level 1: Marital Status
    #    Level 2: Top Owned Product in Portfolio (e.g. if owns P5DA, put P5DA)
    
    # Let's try: Level 1 = Sex, Level 2 = Marital, Level 3 = Most Pop Product Owned
    
    def get_primary_product(row):
        # Return the most popular product they own
        owned = [p for p in prod_cols if row[p] == 1]
        if not owned: return "None"
        # We assume prod_cols matches some popularity order or we pick first
        return owned[0] # Simplification
    
    # We do this for a sample to save time
    df_samp = df.sample(min(1000, len(df))).copy()
    df_samp["Primary_Product"] = df_samp.apply(get_primary_product, axis=1)
    
    fig_sun = px.sunburst(
        df_samp,
        path=["sex", "marital_status", "Primary_Product"],
        title="Portfolio Segments (Sex > Status > Main Product)",
        color="basket_size",
        color_continuous_scale="RdBu"
    )
    st.plotly_chart(fig_sun, use_container_width=True)

with c_heat:
    st.subheader("Cross-Sell Heatmap")
    
    # Use Matrix calculation
    df_samp2 = df.sample(min(2000, len(df)))
    X = df_samp2[prod_cols].fillna(0).astype(int)
    co_matrix = X.T.dot(X)
    # Normalize by diagonal (Standard Jaccard-ish or just Conditional Prob)
    diag = np.diag(co_matrix)
    with np.errstate(divide='ignore', invalid='ignore'):
        cond_matrix = co_matrix.div(diag, axis=0).fillna(0) # P(Col|Row)
    
    fig_heat = px.imshow(
        cond_matrix,
        labels=dict(x="Has Also (Target)", y="Already Has (Context)", color="Prob"),
        color_continuous_scale="Mint",
        title="Product Affinity Matrix"
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    
st.caption("Power BI Style Analytics Module for Zimnat Management.")
