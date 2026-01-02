import streamlit as st

st.set_page_config(page_title="Documentation - Zimnat IA", layout="wide")

# --- CSS ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Technical Documentation")
st.markdown("Details on Algorithms, Metrics, and System Logic.")

# --- TABS ---
tab1, tab2, tab3 = st.tabs(["1. Models & Hybridization", "2. Metrics Definition", "3. Tool Guide"])

# --- TAB 1: MODELS ---
with tab1:
    st.header("1. Dual-Brain Architecture")
    st.markdown("The system uses two distinct models to generate recommendations.")

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("A. Statistical Baseline")
        st.info("**Philosophy**: Collective Wisdom")
        st.markdown("""
        The Baseline model relies on **Conditional Probabilities**.
        It calculates the probability of buying product B given that a customer already owns product A.
        
        $$ P(B | A) = \\frac{Count(A \\cap B)}{Count(A)} $$
        
        *   **Pros**: Extremely reliable for common patterns (e.g. "Car Insurance -> Life Insurance").
        *   **Cons**: Blind to demographic nuances (Age, Sex, Income).
        """)
        
    with c2:
        st.subheader("B. CatBoost (Gradient Boosting)")
        st.info("**Philosophy**: Contextual Intelligence")
        st.markdown("""
        The AI model uses **Gradient Boosted Decision Trees (GBDT)**.
        It learns non-linear relationships between client features (Age, Join Year, Job) and product ownership.
        
        *   **Input Features**:
            *   `age`: Calculated from Birth Year.
            *   `sex`, `marital_status`: Categorical embeddings.
            *   `join_year`: Tenure proxy.
        *   **Pros**: Highly personalized (e.g. "Young single male needs X").
        *   **Cons**: Can overfit if data is sparse.
        """)
        
    st.divider()
    st.subheader("C. Hybrid Strategy (Alpha Blending)")
    st.markdown("""
    To get the best of both worlds, we multiply the scores using an **Alpha Parameter**.
    This forces the AI to respect statistical consensus while allowing it to refine the ranking.
    
    $$ Score_{Final} = P_{AI} \\times (P_{Stats})^{\\alpha} $$
    
    *   **Alpha = 0.0**: **Pure AI** (Riskier, more personalized).
    *   **Alpha = 1.0**: **Pure Stats** (Safest, crowd-based).
    *   **Alpha = 0.5**: **Balanced** (Recommended Production settings).
    """)

# --- TAB 2: METRICS ---
with tab2:
    st.header("2. Performance Metrics")
    
    st.subheader("Hit@k (Accuracy)")
    st.markdown("""
    Measures if the **True Next Product** is present in the Top-K recommendations.
    *   **Hit@1**: Is the #1 recommendation correct? (Strict Accuracy).
    *   **Hit@5**: Is the correct product somewhere in the Top 5? (Recall).
    """)
    
    st.subheader("MAP (Mean Average Precision)")
    st.markdown("""
    A stricter metric that cares about the **Order** of recommendations.
    If the correct product is at position #1, it scores 1.0. At position #2, it scores 0.5.
    
    $$ MAP = \\frac{1}{N} \\sum_{i=1}^{N} \\frac{1}{Rank_i} $$
    
    *High MAP means the model not only finds the right product but places it at the top.*
    """)

# --- TAB 3: GUIDE ---
with tab3:
    st.header("3. User Guide")
    
    st.markdown("### Model Inspector")
    st.write("Use this tool to audit specific clients.")
    st.code("Tip: Uncheck a product in the 'Portfolio' to see if the model can 'guess' it back (Hide & Seek test).")
    
    st.markdown("### Simulator")
    st.write("Use this tool to test hypotheses.")
    st.code("Tip: Use 'Real Client' to load distinct profiles, then tweak 'Age' to see how recommending policy changes.")
    
    st.markdown("### Model Monitoring")
    st.write("Monitoring view.")
    st.code("Tip: Check 'Hyperparam Impact' to see if complex trees (Depth 8+) are really needed (often Depth 6 is enough).")
