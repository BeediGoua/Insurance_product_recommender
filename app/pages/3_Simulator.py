import sys
from pathlib import Path

# Fix Path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import json
from src.inference import get_recommendations, load_baseline
from src.config import ARTIFACTS_DIR, BASELINE_VERSION

st.set_page_config(page_title="Simulator - Zimnat", layout="wide")

# --- CSS LOADING ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- LOAD RESOURCES ---
baseline = load_baseline()
if not baseline:
    st.error("Baseline not found.")
    st.stop()
all_products = baseline.product_cols

# Cache data for random picker
@st.cache_resource
def load_data():
    return pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")

df_train = load_data()

st.title("Interactive Simulator")
st.markdown("Test the models by manually creating profiles or using predefined scenarios.")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Configuration")
    alpha_input = st.slider("Alpha Parameter (Hybrid Weight)", 0.0, 1.0, 0.5, 0.1, help="0 = Pure AI, 1 = Pure Stats")

# --- INITIALIZE SESSION STATE ---
if "sim_age" not in st.session_state: st.session_state["sim_age"] = 30
if "sim_sex" not in st.session_state: st.session_state["sim_sex"] = "M"
if "sim_status" not in st.session_state: st.session_state["sim_status"] = "M"
if "sim_products" not in st.session_state: st.session_state["sim_products"] = []

# --- 1. SCENARIO SELECTOR ---
st.subheader("1. Quick Scenarios & Real Data")
st.markdown("Select a predefined profile OR load a **Real Client** from the database to start.")

col_sc1, col_sc2, col_sc3, col_sc4, col_custom = st.columns(5)

# --- A. PREDEFINED ---
if col_sc1.button("Student (Young)"):
    st.session_state["sim_age"] = 21
    st.session_state["sim_sex"] = "M"
    st.session_state["sim_status"] = "U"
    st.session_state["sim_products"] = []
    
if col_sc2.button("Family (Mid-Life)"):
    st.session_state["sim_age"] = 40
    st.session_state["sim_sex"] = "F"
    st.session_state["sim_status"] = "M"
    st.session_state["sim_products"] = ["P5DA", "RIBP"] 

# --- B. REAL DATA ---
if col_custom.button("Real Client (Random)"):
    # Pick random row
    row = df_train.sample(1).iloc[0]
    
    # Calculate Age
    age_val = row['age'] if pd.notnull(row['age']) else (2020 - row['birth_year'])
    
    # Extract Products
    real_products = [c for c in all_products if row[c] == 1]
    
    st.session_state["sim_age"] = int(age_val)
    st.session_state["sim_sex"] = row['sex']
    st.session_state["sim_status"] = row['marital_status']
    st.session_state["sim_products"] = real_products
    st.toast(f"Loaded Client {row['ID']}")
    
if col_sc3.button("Senior VIP (Established)"):
    st.session_state["sim_age"] = 68
    st.session_state["sim_sex"] = "M"
    st.session_state["sim_status"] = "M"
    st.session_state["sim_products"] = ["P5DA", "RIBP", "8NN1", "7POT"]
    
if col_sc4.button("Clear Form"):
    st.session_state["sim_age"] = 30
    st.session_state["sim_sex"] = "F"
    st.session_state["sim_status"] = "S"
    st.session_state["sim_products"] = []

# --- 2. BUILDER AR ---
st.markdown("---")
st.subheader("2. Client Profile Builder")

c_left, c_right = st.columns([1, 1.5], gap="large")

with c_left:
    # --- IMPORT TOOL ---
    with st.expander("Import from Database (Advanced Filters)"):
        st.caption("Find a real client matching specific criteria.")
        
        target_size_sim = st.slider("Target Basket Size", 0, 10, 2, key="sim_target_size")
        
        if st.button("Fetch Random Match"):
             prod_cols = baseline.product_cols
             basket_sizes = df_train[prod_cols].sum(axis=1)
             
             # 1. Exact vs Fuzzy search (Same logic as Inspector)
             valid = df_train[basket_sizes == target_size_sim]
             if valid.empty:
                  valid = df_train[(basket_sizes >= target_size_sim - 1) & (basket_sizes <= target_size_sim + 1)]
                  if not valid.empty:
                      st.toast("Exact match not found. Using fuzzy match (+/- 1).")
             
             if not valid.empty:
                 row = valid.sample(1).iloc[0]
                 
                 # Extract & Set
                 age_val = row['age'] if pd.notnull(row['age']) else (2020 - row['birth_year'])
                 real_products = [c for c in all_products if row[c] == 1]
                 
                 st.session_state["sim_age"] = int(age_val)
                 st.session_state["sim_sex"] = row['sex']
                 st.session_state["sim_status"] = row['marital_status']
                 st.session_state["sim_products"] = real_products
                 
                 st.toast(f"Imported Client {row['ID']} ({len(real_products)} products)!")
                 st.rerun()
             else:
                 st.warning("No client found.")

    st.markdown("#### Input Parameters")
    
    # Defaults handled by session_state
    
    age = st.number_input("Age", 18, 100, st.session_state["sim_age"])
    sex = st.selectbox("Sex", ["M", "F"], index=0 if st.session_state["sim_sex"]=="M" else 1)
    status = st.selectbox("Marital Status", ["M", "U", "S", "W"], index=["M","U","S","W"].index(st.session_state["sim_status"]))
    
    st.markdown("#### Owned Products")
    curr_prods = st.multiselect("Select Products", all_products, default=st.session_state["sim_products"])
    
    # Update Session State on Change (Implicit)

with c_right:
    st.markdown("### Prediction Engine")
    
    if st.button("Generate Recommendation", type="primary"):
        
        # 1. Prepare Data
        user_features = {
            "age": age,
            "sex": sex,
            "marital_status": status,
            "join_year": 2018 # Default for simulation
        }
        
        ctx_hybrid = {
            "owned_products": curr_prods,
            "user_features": user_features
        }
        
        ctx_baseline = {"owned_products": curr_prods}
        
        # 2. Get Predictions
        rec_base = get_recommendations("Baseline", ctx_baseline, topk=5)
        rec_hybrid = get_recommendations("CatBoost", ctx_hybrid, topk=5, alpha_override=alpha_input)
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.info("Baseline (Stats)")
            st.dataframe(rec_base, height=200)
            
        with col_res2:
            st.success(f"Hybrid AI (Alpha={alpha_input})")
            st.dataframe(rec_hybrid, height=200)
            
        # 3. INTERPRETATION
        st.markdown("#### Interpretation")
        if not rec_hybrid.empty and not rec_base.empty:
            top_h = rec_hybrid.index[0]
            top_b = rec_base.index[0]
            
            if top_h == top_b:
                st.success(f"**Consensus:** Both models recommend **{top_h}**.")
            else:
                st.info(f"**Refinement:** The Statistical model suggests **{top_b}**, but the AI (taking into account Age {age} & Status {status}) prioritizes **{top_h}**.")
