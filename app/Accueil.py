import json
from pathlib import Path
import streamlit as st

# Load Metrics
try:
    metrics_path = Path("artifacts/baseline_v0/metrics.json")
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
    else:
        metrics = None
except Exception:
    metrics = None

st.set_page_config(
    page_title="Zimnat AI Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# --- HEADER ---
st.title("Zimnat Insurance AI")
st.caption("Plateforme de recommandation intelligente & Cross-sell Optimization")

# --- KPI ROW ---
if metrics:
    st.markdown("### Performance du Modèle (Global)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Précision (Top-5)", f"{metrics.get('Hit@5', 0):.1%}", delta="Baseline OK")
    with c2:
        st.metric("Pertinence (MRR)", f"{metrics.get('MRR', 0):.2f}", help="Mean Reciprocal Rank (1.0 = parfait)")
    with c3:
        st.metric("Produits Couverts", "21", help="Nombre total de produits au catalogue")
    st.markdown("---")

# --- CONTENT ---
c_left, c_right = st.columns([2, 1])

with c_left:
    st.subheader("Mission du Projet")
    st.info(
        """
        L'objectif est d'optimiser la valeur client (Customer Lifetime Value) en prédisant **le prochain produit d'assurance** 
        qu'un client est susceptible d'acheter, en se basant sur son portefeuille actuel.
        """
    )
    
    st.subheader("Comment ça marche ?")
    st.graphviz_chart("""
        digraph {
            rankdir=LR;
            node [shape=box, style=filled, fillcolor="#f0f2f6", fontname="Arial"];
            A [label="Client Portfolio\n(Produits A, B)"];
            B [label="Moteur de Règle\n(Probabilités)"];
            C [label="Score & Filtres\n(Ex: Pas de doublons)"];
            D [label="Recommandation\n(Produit C)"];
            
            A -> B [label="Input"];
            B -> C [label="P(C|A,B)"];
            C -> D [label="Top-K"];
        }
    """)

with c_right:
    st.subheader("Navigation")
    
    with st.expander("1. Dashboard", expanded=True):
        st.write("Vue macroscopique. Analysez où le modèle performe (Taille de panier, Segments).")
        st.page_link("pages/1_Dashboard.py", label="Ouvrir Dashboard")
        
    with st.expander("2. Inspector", expanded=True):
        st.write("Vue microscopique. Comprenez le 'Pourquoi' d'une recommandation pour un client donné.")
        st.page_link("pages/2_Inspector.py", label="Ouvrir Inspector")
    
    with st.expander("Documentation"):
        st.write("Détails techniques sur l'algorithme (Baseline vs XGBoost).")
        st.info("Voir `notes/comprendre.md`")

st.markdown("---")
st.caption("Developed for Zimnat Insurance Challenge | Baseline v0.1")
