import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from src.evaluation.evaluate_V2 import evaluate_masking

def test_alignment():
    print("=== Testing Evaluation Alignment ===")
    
    # 1. Fake Data
    N = 100
    P = 5
    product_cols = [f"prod_{i}" for i in range(P)]
    
    # Products (random 0/1)
    X_products = np.random.randint(0, 2, size=(N, P))
    # Ensure at least 2 products for some
    X_products[:50, :2] = 1 
    
    # Context (DataFrame)
    df_context = pd.DataFrame({
        "age": np.random.randint(20, 60, size=N),
        "ID": range(N)
    })
    
    # 2. Mock Legacy Baseline (No context)
    print("\n[1] Testing Legacy Baseline (ignores index)...")
    def score_baseline(x_row):
        # Dumb scorer: return random probabilities
        return np.random.random(P)
        
    try:
        # evaluate_masking expects X_full (products interaction)
        info, metrics = evaluate_masking(X_products, score_fn=score_baseline, hide_k=1, min_observed=1)
        print(f"✅ Success! Metrics: Hit@1={metrics['Hit@1']:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")
        
    # 3. Mock New Model (Context Aware)
    print("\n[2] Testing Context-Aware Model (uses index)...")
    def score_context_aware(x_row, idx):
        # Fetch age from context using idx
        age = df_context.loc[idx, "age"]
        # Dummy logic: if age > 30, prefer prod_0, else prod_1
        scores = np.zeros(P)
        if age > 30:
            scores[0] = 1.0
        else:
            scores[1] = 1.0
        return scores + np.random.random(P)*0.1 # Add noise
        
    try:
        info, metrics = evaluate_masking(X_products, score_fn=score_context_aware, hide_k=1, min_observed=1)
        print(f"✅ Success! Metrics: Hit@1={metrics['Hit@1']:.4f}")
    except Exception as e:
        print(f"❌ Failed: {e}")

if __name__ == "__main__":
    test_alignment()
