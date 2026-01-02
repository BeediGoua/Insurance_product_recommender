import sys
from pathlib import Path

# Fix Path to allow importing 'src'
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from src.config import ARTIFACTS_DIR, BASELINE_VERSION
from src.inference import load_baseline, get_recommendations

st.set_page_config(page_title="Inspector - Zimnat", layout="wide")

# --- CSS LOADING ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    # Load Baseline Artifact (for product lists)
    artifact = load_baseline()
    
    # Load CLEANED train data for inspection (consistent with model)
    df = pd.read_parquet(ARTIFACTS_DIR / BASELINE_VERSION / "train_cleaned.parquet")
    return artifact, df

artifact, df_train = load_resources()

if not artifact:
    st.error("Baseline not found. Please run the pipeline.")
    st.stop()

st.title("Model Inspector")
st.markdown("""
This tool allows for a side-by-side comparison of recommendation strategies on real client data.
*   **Statistical Strategy (Baseline)**: Recommends based on products typical for current portfolio.
*   **Hybrid Strategy (AI)**: Recommends based on specific client profile features (Age, Job, etc.).
""")

# --- SIDEBAR: CONFIG ---
st.sidebar.header("Configuration")
alpha_input = st.sidebar.slider(
    "Hybrid Alpha Weight", 0.0, 1.0, 0.5, 0.1,
    help="Adjusts the balance between AI Profile scores and Statistical consensus."
)

# --- SEARCH BAR ---
col_search, col_action = st.columns([1, 2])
with col_search:
    default_id = st.session_state.get("client_id", "")
    client_id_input = st.text_input("Search Client ID", value=default_id, placeholder="Ex: ID_XXXX")
    
    # Session Sync
    if client_id_input != default_id:
        st.session_state["client_id"] = client_id_input
        client_id = client_id_input
    else:
        client_id = default_id

    st.markdown("---")
    st.write("**Find a random profile**")
    
    with st.expander("Search Filters"):
        target_size = st.slider("Target Basket Size", 0, 10, 2)
        if st.button("Random Client"):
            prod_cols = artifact.product_cols
            basket_sizes = df_train[prod_cols].sum(axis=1)
            
            # 1. Exact Match Priority
            valid = df_train[basket_sizes == target_size]
            
            # 2. Fallback (Fuzzy Match +/- 1)
            if valid.empty:
                 valid = df_train[(basket_sizes >= target_size -1) & (basket_sizes <= target_size + 1)]
                 if not valid.empty:
                     st.toast(f"Exact match not found. Found close match (+/- 1).")
            
            if not valid.empty:
                pick = valid.sample(1).iloc[0]["ID"]
                st.session_state["client_id"] = pick
                st.rerun()
            else:
                st.warning("No client found for this criteria.")

# --- MAIN INSPECTION ---
if client_id:
    client_row = df_train[df_train["ID"] == client_id]
    if not client_row.empty:
        client_data = client_row.iloc[0]
        
        # 1. DISPLAY PROFILE
        st.subheader(f"Client Profile : {client_id}")
        
        # Calcul Age si manquant (Ref 2020)
        age = client_data.get('age')
        if pd.isna(age) and 'birth_year' in client_data:
             age = 2020 - client_data['birth_year']
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Sex", client_data.get('sex', '?'))
        c2.metric("Age", f"{int(age)}" if pd.notnull(age) else "?")
        c3.metric("Marital Status", client_data.get('marital_status', '?'))
        c4.metric("Join Year", int(client_data.get('join_year', 0)))

        # 2. INTERACTIVE BASKET
        st.write("### Current Portfolio (Ground Truth)")
        st.info("Uncheck products to simulate 'hiding' them from the model (Simulation Mode).")
        
        real_products = [p for p in artifact.product_cols if client_data[p] == 1]
        
        # Dynamic Checkboxes
        input_products = []
        cols = st.columns(6)
        for i, p in enumerate(real_products):
            with cols[i % 6]:
                if st.checkbox(p, value=True, key=f"chk_{p}"):
                    input_products.append(p)
        
        targets_hidden = [p for p in real_products if p not in input_products]
        if targets_hidden:
            st.caption(f"Targets (Hidden): {', '.join(targets_hidden)}")
        
        st.divider()
        
        # 3. DUAL PREDICTION
        if st.button("Run Model Comparison", type="primary"):
            
            # A. Prepare Contexts
            ctx_baseline = {"owned_products": input_products}
            
            # CatBoost needs everything
            features = client_data.to_dict()
            ctx_hybrid = {
                "owned_products": input_products, 
                "user_features": features
            }
            
            # B. Get Predictions
            recs_base = get_recommendations("Baseline", ctx_baseline, topk=5)
            # Use Alpha from Sidebar
            recs_hybrid = get_recommendations("CatBoost", ctx_hybrid, topk=5, alpha_override=alpha_input) 
            
            # C. Display
            col_base, col_mid, col_ai = st.columns([1, 0.1, 1])
            
            with col_base:
                st.markdown("### Statistical Strategy")
                st.caption("Based on Population patterns only.")
                if not recs_base.empty:
                    st.dataframe(recs_base.rename("Score"), height=200)
                    top_b = recs_base.index[0]
                    st.success(f"Top Pick: **{top_b}**")
                    with st.expander("Explanation (Stats)"):
                        st.write("This product has the highest conditional probability given the current basket.")
                else:
                    st.warning("No sufficient data.")
            
            with col_mid:
                st.markdown("<div style='border-left:1px solid #ddd;height:100%'></div>", unsafe_allow_html=True)
            
            with col_ai:
                st.markdown(f"### Hybrid Strategy (Alpha={alpha_input})")
                st.caption(f"Micro-segmentation (Profile + Consensus).")
                if not recs_hybrid.empty:
                    st.dataframe(recs_hybrid.rename("Score"), height=200)
                    top_c = recs_hybrid.index[0]
                    st.markdown(f'<div class="top-pick-card">Top Pick: <b>{top_c}</b></div>', unsafe_allow_html=True)
                    
                    # DYNAMIC EXPLANATION
                    with st.expander("Explanation (Hybrid)"):
                         if top_c == top_b:
                             st.write(f"**Agreement:** The AI confirms the statistical trend. The user's profile ({int(age)} years old) aligns with the general population behavior for this product.")
                         else:
                             st.write(f"**Divergence:** The AI recommends **{top_c}** instead of {top_b}.")
                             st.write("Possible reasons for this shift:")
                             reasons = []
                             if age > 55: reasons.append(f"- Age ({int(age)}) typically shifts needs towards senior products.")
                             if client_data.get('join_year', 2020) < 2012: reasons.append("- Long-term loyalty influences the cross-sell potential.")
                             if client_data.get('sex') == 'F': reasons.append("- Gender-specific purchasing patterns detected.")
                             
                             if reasons:
                                 for r in reasons: st.write(r)
                             else:
                                 st.write("- Complex non-linear interactions in the profile detected by the Gradient Boosting trees.")

                else:
                    st.warning("Model score 0.")
                    
    else:
        st.error("Client not found.")
