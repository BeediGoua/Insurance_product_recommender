import json
from pathlib import Path
import streamlit as st

# --- CONFIG ---
BASELINE_METRICS_PATH = Path("artifacts/baseline_v0/metrics.json")
V1_CONFIG_PATH = Path("artifacts/best_config_v1.json")

# --- LOAD DATA ---
baseline_metrics = None
v1_metrics = None

try:
    if BASELINE_METRICS_PATH.exists():
        baseline_metrics = json.loads(BASELINE_METRICS_PATH.read_text())
except:
    pass

try:
    if V1_CONFIG_PATH.exists():
        v1_metrics = json.loads(V1_CONFIG_PATH.read_text())
except:
    pass

st.set_page_config(
    page_title="Zimnat Expert System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- HEADER ---
st.title("Zimnat Expert System")
st.caption("Advanced Insurance Recommendation Engine (Hybrid AI + Statistics)")

# --- KPI ROW ---
st.header("System Performance")
c1, c2, c3, c4 = st.columns(4)

# 1. Baseline Stats
with c1:
    val = baseline_metrics.get('Hit@1', 0) if baseline_metrics else 0
    st.metric("Baseline (Statistical)", f"{val:.1%}", help="Top-1 Accuracy (Population Rules)")

# 2. Hybrid AI
with c2:
    val_v1 = v1_metrics.get('Hit@1_max', 0) if v1_metrics else 0
    # Calculate delta
    delta = None
    if val > 0:
        diff = val_v1 - val
        delta = f"{diff:+.1%}"
    st.metric("Hybrid AI (Personalized)", f"{val_v1:.1%}", delta=delta, help="Top-1 Accuracy (CatBoost + Stats)")

# 3. Catalog
with c3:
    st.metric("Product Catalog", "21 Products", help="Active Insurance Products")

# 4. Status
with c4:
    st.metric("System Status", "Live", delta="Optimized", delta_color="normal")

st.markdown("---")

# --- CONTENT ---
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("System Architecture")
    st.info(
        """
        The **Zimnat Expert System** optimizes Customer Lifetime Value by predicting the next best product.
        It combines two powerful engines:
        1.  **Statistical Memory (Baseline)**: "What do people with a similar portfolio usually buy?"
        2.  **Contextual Intelligence (CatBoost)**: "What does this specific demographic profile need?"
        """
    )
    
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Arial", fontsize=10];
            
            Inputs [label="Client Data\n(Basket, Profile)"];
            
            subgraph cluster_0 {
                label = "Dual Engine";
                style=dashed;
                color=grey;
                Base [label="Statistical\nBaseline", fillcolor="#bbdefb"];
                AI [label="Hybrid AI\n(Gradient Boosting)", fillcolor="#ffccbc"];
            }
            
            Mixer [label="Hybridization\n(Alpha Control)", style=filled, fillcolor="#d1c4e9"];
            Output [label="Top-5\nRecommendations", shape=ellipse, style=filled, fillcolor="#c8e6c9"];
            
            Inputs -> Base;
            Inputs -> AI;
            Base -> Mixer [label="Consensus"];
            AI -> Mixer [label="Prediction"];
            Mixer -> Output;
        }
    """)
    
with c_right:
    st.subheader("Navigation")
    
    with st.expander("1. Model Monitoring (Technical)", expanded=True):
        st.write("Compare Baseline vs Hybrid models and monitor accuracy KPIs.")
        st.page_link("pages/1_Model_Monitoring.py", label="Open Monitoring")
        
    with st.expander("2. Inspector (Audit)", expanded=True):
        st.write("Explain 'Why' a decision was made for a specific client.")
        st.page_link("pages/2_Inspector.py", label="Open Inspector")
        
    with st.expander("3. Simulator (Lab)", expanded=True):
        st.write("Test 'What-If' scenarios with the Hybrid Alpha slider.")
        st.page_link("pages/3_Simulator.py", label="Open Simulator")

    with st.expander("4. Documentation", expanded=True):
        st.write("Technical details, Model Theory, and Metrics definitions.")
        st.page_link("pages/4_Documentation.py", label="Read Docs")

st.markdown("---")
st.caption("Zimnat Insurance AI | Version 2.0 (Dual Expert System)")
