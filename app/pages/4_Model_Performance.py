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
import numpy as np
from src.config import OUT_DIR, ARTIFACTS_DIR, BASELINE_VERSION

st.set_page_config(layout="wide", page_title="Dashboard - Zimnat Model Lab")

# --- CSS ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Model Benchmark & Monitoring")
st.markdown("Global performance comparison between Statistical Baseline and Hybrid AI models.")

# --- 1. BENCHMARK COMPARISON ---
st.header("1. Model Leaderboard")

@st.cache_data
def load_benchmark():
    # Load Tuning Results
    path = ARTIFACTS_DIR / "tuning_results_grid.csv"
    if path.exists():
        return pd.read_csv(path)
    return None

df_bench = load_benchmark()

if df_bench is not None:
    # Get Best Model (Top Hit@1)
    # Columns expected: depth, learning_rate, Hit@1, Hit@3
    metric = "Hit@1" if "Hit@1" in df_bench.columns else df_bench.columns[2]
    
    best_model = df_bench.sort_values(metric, ascending=False).iloc[0]
    
    c1, c2, c3 = st.columns(3)
    
    top_score = best_model.get(metric, 0.0)
    best_depth = best_model.get("depth", 6)
    best_lr = best_model.get("learning_rate", 0.05)
    
    c1.metric("Best CatBoost Score", f"{metric}: {top_score:.2%}")
    c2.metric("Optimal Depth", f"{int(best_depth)}")
    c3.metric("Optimal Learning Rate", f"{best_lr}")
    
    # Visualization: Hyperparam Impact
    fig_bench = px.bar(
        df_bench, 
        x="depth", 
        y=metric, 
        color="learning_rate",
        barmode="group",
        title=f"Impact of Depth & LR on {metric}",
        labels={"depth": "Tree Depth", metric: "Accuracy (Hit@1)"}
    )
    
    st.plotly_chart(fig_bench, use_container_width=True)
    st.caption("Monitoring the impact of model complexity on accuracy.")

else:
    st.warning("Benchmark results not found (tuning_results_grid.csv).")


# --- 2. UNDER THE HOOD: STATS vs AI ---
st.divider()
st.header("2. Logic Comparison")

col_stats, col_sep, col_ai = st.columns([1, 0.1, 1])

# --- LEFT: STATS LOGIC ---
with col_stats:
    st.subheader("A. Statistical Logic (The Matrix)")
    st.markdown("Baseline relies on **Global Co-occurrence**. If products often go together, they are recommended.")
    
    # Load Co-occurrence Matrix (Calculated Live)
    @st.cache_data
    def load_matrix_data():
        return pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")
    
    try:
        df_raw = load_matrix_data()
        meta_cols = ["ID", "join_date", "sex", "marital_status", "birth_year", "branch_code", "occupation_code", "occupation_category_code", "join_year", "age_raw", "age", "join_year_missing", "age_missing", "age_was_clipped"]
        prod_cols = [c for c in df_raw.columns if c not in meta_cols]
        
        # Simple Matrix calc on sample for speed
        df_samp = df_raw.sample(min(5000, len(df_raw)))
        X = df_samp[prod_cols].fillna(0).astype(int)
        co_matrix = X.T.dot(X)
        diag = np.diag(co_matrix)
        with np.errstate(divide='ignore', invalid='ignore'):
            cond_matrix = co_matrix.div(diag, axis=0).fillna(0)
            
        fig_mat = px.imshow(
            cond_matrix,
            labels=dict(x="Recommended", y="Owned", color="Prob"),
            color_continuous_scale="Blues",
        )
        fig_mat.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_mat, use_container_width=True)
        st.caption("Visualizing 'Product Affinities'. Darker = Stronger Link.")
        
    except Exception as e:
        st.error(f"Could not load matrix: {e}")

with col_sep:
    st.markdown("<div style='border-left:1px solid #ddd;height:100%'></div>", unsafe_allow_html=True)

# --- RIGHT: AI LOGIC ---
with col_ai:
    st.subheader("B. Artificial Intelligence (The Profile)")
    st.markdown("CatBoost focuses on **User Features** (Age, Status...). It finds non-linear patterns.")
    
    # Feature Importance Mockup (or Load if available)
    feats = {
        "Feature": ["Age", "Marital Status", "Occupation Code", "Join Year", "Sex"],
        "Importance": [45, 25, 15, 10, 5]
    }
    df_feat = pd.DataFrame(feats)
    
    fig_feat = px.bar(
        df_feat, 
        x="Importance", 
        y="Feature", 
        orientation='h',
        color="Importance",
        color_continuous_scale="Purples",
        title="AI Driver Factors (Concept)"
    )
    fig_feat.update_layout(height=400, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_feat, use_container_width=True)
    st.plotly_chart(fig_feat, use_container_width=True)
    st.caption("The AI creates groups based on these profile attributes, creating 'Personalized' segments.")

st.markdown("---")
st.caption("Zimnat Insurance AI | **Goua Beedi** | [LinkedIn](https://www.linkedin.com/in/goua-beedi-henri-a152bb1b2/) | [GitHub](https://github.com/BeediGoua) | [Portfolio](https://beedigoua.github.io/)")

