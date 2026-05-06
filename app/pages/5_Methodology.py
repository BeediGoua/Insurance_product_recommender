import sys
from pathlib import Path

# Fix Path
root_path = Path(__file__).resolve().parent.parent.parent
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import streamlit as st

st.set_page_config(layout="wide", page_title="Documentation - Zimnat IA")

# --- CSS PRO ---
# --- CSS LOADING ---
with open("app/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.title("Technical Documentation")
st.markdown("**Zimnat Expert System**: A Hybrid Recommendation Engine combining Statistical Wisdom and Machine Learning.")

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["1. The Dual Engine", "2. Evaluation Protocol", "3. Metrics & Checks", "4. User Guide"])

# --- TAB 1: THE DUAL ENGINE ---
with tab1:
    st.header("1. The Dual Engine Architecture")
    st.markdown("The system reconciles two opposing philosophies: **Collective Wisdom** (Statistics) and **Contextual Intelligence** (AI).")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("A. The Statistical Brain (Baseline)")
        st.markdown("""
        <div class="doc-box">
            <p><strong>Philosophy</strong>: <em>"What do people usually buy together?"</em></p>
            <p>Based on <strong>Conditional Probabilities</strong>, specifically weighted by support to avoid noise.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### The Formula")
        st.latex(r'''
        Score(B) = \sum_{k=1}^{m} w_k \cdot P(B|A_k)
        ''')
        st.caption("Where A_k are products already owned by the client, and w_k is a weight based on support.")
        
        st.markdown("### Logic Flow")
        st.graphviz_chart('''
        digraph {
            rankdir=TB;
            Input [label="Observed Basket\n{A, B, C}", shape=box, style=filled, fillcolor="#E3F2FD"]
            Matrix [label="Probability Matrix\nP(Target|Item)", shape=folder]
            Calc [label="Aggregated Score\nSum(Weights * Probs)", shape=ellipse]
            Filter [label="Filter Owned\n(Don't recommend A, B, C)", shape=octagon]
            Result [label="Ranked Recommendations", shape=note, style=filled, fillcolor="#C8E6C9"]
            
            Input -> Matrix
            Matrix -> Calc
            Calc -> Filter
            Filter -> Result
        }
        ''')

    with col2:
        st.subheader("B. The AI Brain (CatBoost)")
        st.markdown("""
        <div class="doc-box">
            <p><strong>Philosophy</strong>: <em>"Who is this person?"</em> (Contextual Intelligence)</p>
            <p>Unlike the Baseline which looks at <em>Products</em>, the AI looks at the <em>User</em>. It learns non-linear interactions between Demographics and Portfolio.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 1. Problem Formulation: Multiclass")
        st.markdown("We treat recommendation as a **Multiclass Classification** problem with 21 classes (one for each product).")
        st.markdown("> **Input**: User Profile + Current Portfolio (minus one hidden item).")
        st.markdown("> **Output**: Probability distribution over all 21 products.")

        st.markdown("### 2. Why CatBoost?")
        st.markdown("""
        *   **High Cardinality Handling**: We have features like `occupation_code` with hundreds of values. CatBoost handles them natively (Ordered Target Statistics) without exploding memory like One-Hot Encoding.
        *   **Robustness**: It uses 'Oblivious Trees' (symmetric trees) which are less prone to overfitting on small datasets.
        """)
        
        st.markdown("### 3. The 'Smart Masking' Strategy")
        st.markdown("To prevent the model from just learning 'Popularity' (e.g. always predict Funeral Insurance), we use **Inverse Frequency Masking** during training.")
        st.latex(r'''P(mask=c) \propto \frac{1}{freq(c)^\gamma}''')
        st.caption("We hide rare products more often to force the model to learn hard patterns.")

        st.markdown("### The Learning Logic")
        st.latex(r'''
        h_t(x) \approx - \frac{\partial \mathcal{L}(y, \hat{f}_{t-1}(x))}{\partial \hat{f}_{t-1}(x)}
        ''')
        st.caption("Gradient Boosting: Each tree corrects the mistakes of the previous one.")

    st.divider()
    st.subheader("C. Hybridization (Alpha Blending)")
    st.markdown("We fuse the two scores using a linear combination controlled by **Alpha**.")
    st.latex(r'''
    Score_{Final} = (1 - \alpha) \cdot P_{AI} + \alpha \cdot P_{Stats}
    ''')
    st.info("Alpha = 0.5 offers the best balance between Personalization and Reliability.")

# --- TAB 2: EVALUATION ---
with tab2:
    st.header("2. Evaluation Protocol: 'Hide and Seek'")
    
    st.markdown("### The Core Challenge")
    st.markdown("We don't have a 'Future' dataset. So how do we know if the model works? **We simulate the future.**")
    
    st.subheader("The 'Magic Trick' (Leave-One-Out)")
    st.markdown("""
    We use a strict scientific protocol on our historical data:
    1.  **Take a Real Client**: Who owns {Auto, Health, Home}.
    2.  **The Mask**: We artificially *hide* one product (e.g., 'Home').
    3.  **The Test**: We ask the model: *"What does this person need?"*
    4.  **The Verdict**: If the model suggests 'Home' in the Top-3, it's a **HIT**.
    """)
    st.info("This converts a static dataset into a dynamic prediction test.")

    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        Real [label="Real Portfolio\n{Auto, Health, Home}", shape=box, style=filled, fillcolor="#C8E6C9"]
        Mask [label="HIDE 'Home'", shape=circle, style=filled, fillcolor="#FFCDD2"]
        Input [label="Input to Model\n{Auto, Health}", shape=box]
        Model [label="AI Model", shape=ellipse]
        Pred [label="Top Recommendations", shape=note]
        Check [label="Was 'Home' found?", shape=diamond, style=filled, fillcolor="#FFF9C4"]
        
        Real -> Mask
        Mask -> Input
        Input -> Model
        Model -> Pred
        Pred -> Check
    }
    ''')

    st.divider()
    
    st.subheader("Advanced Validation: The 'Stress Test'")
    st.markdown("To prove the model isn't just reciting memory, we go further:")
    st.markdown("""
    *   **Cross-Validation**: We train on 80% of clients and test on the hidden 20%. The model *never* sees the test clients during training.
    *   **Stress Test**: Sometimes we hide **2 or 3 products** at once. If the model can reconstruct the missing pieces, it understands the *structure* of the portfolio, not just simple pairs.
    """)
    
    st.warning("**Constraint**: We only evaluate on clients with **Basket Size ≥ 2**. You cannot play 'complete the pattern' with a single dot.")

# --- TAB 3: METRICS ---
with tab3:
    st.header("3. Key Metrics & Sanity Checks")
    
    st.markdown("We use industry-standard ranking metrics to measure success, plus custom checks to ensure fairness.")
    
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("A. Success Metrics")
        
        st.markdown("#### 1. Hit@K (Discovery)")
        st.latex(r'''Hit@K = \frac{1}{M} \sum_{i=1}^{M} \mathbb{1}(y_{true} \in TopK_i)''')
        st.caption("Did the right answer appear in the shortlist?")
        st.markdown("""
        *   **Hit@1 (Precision)**: The model is absolutely sure. Essential for automated systems.
        *   **Hit@5 (Recall)**: The correct item is in the top 5 suggestions. Useful for human agents offering choices.
        """)
        
        st.markdown("#### 2. MRR (Ranking Quality)")
        st.latex(r'''MRR = \frac{1}{M} \sum_{i=1}^{M} \frac{1}{rank_i}''')
        st.caption("Mean Reciprocal Rank: Rewards being #1.")
        st.markdown("""
        *   If the answer is **#1**, score is **1.0**.
        *   If the answer is **#2**, score is **0.5**.
        *   If the answer is **#10**, score is **0.1**.
        *   *Why it matters*: Hit@5 doesn't care if the answer is 1st or 5th. **MRR demands the best answer first.**
        """)
        
    with c2:
        st.subheader("B. Sanity Checks (The 'Audit')")
        st.markdown("High scores can be misleading. We run these mandatory checks:")
        
        st.markdown("""
        #### 1. The 'Anti-Cheat' Check
        *   **What**: Ensure we **NEVER** recommend a product the client already owns.
        *   **Why**: It's easy to predict 'Auto Insurance' for someone who has it. It counts as a Hit but is useless business-wise.
        *   **Fix**: Hard-filter owned items ($Score = -\infty$).
        
        #### 2. The 'Popularity' Check
        *   **What**: Look at the distribution of the #1 Recommendation.
        *   **Why**: If 95% of recommendations are "Funeral Insurance", the model is lazy (Popularity Bias).
        *   **Fix**: Use **Inverse Frequency Masking** (see Tab 1) to force diversity.
        
        #### 3. The 'Leakage' Check
        *   **What**: Ensure the target variable `y` was not accidentally left in the features `X`.
        *   **Verify**: If Hit@1 is 100%, you have a leak. Real humans are unpredictable; 100% is impossible.
        """)

# --- TAB 4: USER GUIDE ---
with tab4:
    st.header("4. Standard Operating Procedures (SOP)")
    
    st.markdown("This guide details how to leverage the Expert System for daily operations and strategic planning.")

    st.subheader("Scenario 1: Individual Policy Audit (Agent Level)")
    st.markdown("**Tool**: Model Inspector")
    st.markdown("**Goal**: Understand why a specific client received a recommendation.")
    st.markdown("""
    1.  **Select a Client ID**: Pick a profile from the sidebar.
    2.  **Review the 'Why'**: Look at the top contributing features (e.g., "Age > 45" or "Has Car Insurance").
    3.  **Stress Test**: Uncheck a product in the 'Current Portfolio'. Does the model immediate recommend it back? 
        *   *If Yes*: The model clearly understands the need for this product.
        *   *If No*: The client might be an outlier or the connection is weak.
    """)

    st.divider()

    st.subheader("Scenario 2: Market Segmentation (Strategy Level)")
    st.markdown("**Tool**: Simulator")
    st.markdown("**Goal**: Define target demographics for a new campaign.")
    st.markdown("""
    1.  **Define a Persona**: Set Age=30, Sex=M, Occupation=Teacher.
    2.  **Observe Preferences**: Note the Top 3 products.
    3.  **Iterate**: Change Age to 50. Note how 'Education Plan' fades and 'Funeral Plan' emerges.
    4.  **Action**: Use these insights to craft age-specific marketing messages.
    """)

    st.divider()

    st.subheader("Scenario 3: Portfolio Optimization (Management Level)")
    st.markdown("**Tool**: Business Analytics")
    st.markdown("**Goal**: Identify under-equipped high-value clients.")
    st.markdown("""
    1.  **Check 'Sleeping Giants'**: This metric reveals loyal clients (>5 years) with only 1 product.
    2.  **Analyze Seasonality**: Check the 'Operational Rhythm' chart. If May is the peak month, prepare staffing in April.
    3.  **Branch Strategy**: Use the Quadrant Analysis to identify which branches need training (Low Quality) vs which need volume support (Low Volume).
    """)

st.markdown("---")
st.caption("Zimnat Insurance AI | **Goua Beedi** | [LinkedIn](https://www.linkedin.com/in/goua-beedi-henri-a152bb1b2/) | [GitHub](https://github.com/BeediGoua) | [Portfolio](https://beedigoua.github.io/)")

