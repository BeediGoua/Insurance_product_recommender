
import json
import pandas as pd
import numpy as np
from pathlib import Path
import streamlit as st

from src.config import ARTIFACTS_DIR, BASELINE_VERSION
from src.pipelines.baseline_pipeline import BaselineArtifact, recommend_from_selection
from src.models.catboost.trainer import CatboostTrainer
from src.models.catboost.predictor import HybridPredictor

# Define the Categorical Features used in Training (Must match Experiment Notebook)
CAT_FEATURES = ["sex", "marital_status", "branch_code", "occupation_code", "occupation_category_code"]

# Cache loading to avoid reloading model on every interaction
@st.cache_resource
def load_baseline():
    """Charge uniquement la Baseline (Stats)"""
    artifact_path = ARTIFACTS_DIR / BASELINE_VERSION
    if not artifact_path.exists():
        return None
    return BaselineArtifact.load(artifact_path)

@st.cache_resource
def load_hybrid_model():
    """Charge le Modèle Hybride (CatBoost + Baseline)"""
    
    # 1. Paths
    cbm_path = ARTIFACTS_DIR / "catboost_champion_v1.cbm"
    config_path = ARTIFACTS_DIR / "best_config_v1.json"
    baseline_path = ARTIFACTS_DIR / BASELINE_VERSION

    # 2. Check existence
    if not cbm_path.exists() or not config_path.exists():
        st.error(f"Modèle CatBoost introuvable. Avez-vous lancé le notebook d'optimisation ? ({cbm_path})")
        return None

    # 3. Load Config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 4. Load Baseline (Required for Hybrid)
    baseline = BaselineArtifact.load(baseline_path)
    
    # 5. Load CatBoost
    # FIX: Use CAT_FEATURES (Input features), NOT product_cols (Targets)
    trainer = CatboostTrainer(
        cat_features=CAT_FEATURES, 
        iterations=100
    )
    trainer.load(cbm_path)
    
    # 6. Create Hybrid Predictor
    alpha = config.get("best_alpha", 0.5) # Default safety
    hybrid = HybridPredictor(trainer, baseline)
    
    return hybrid, alpha

def get_recommendations(model_type, context_data, topk=5, alpha_override=None):
    """
    Fonction unifiée pour prédire.
    
    Args:
        model_type (str): "Baseline" ou "CatBoost"
        context_data (dict): 
            - 'owned_products': list of strings
            - 'user_features': dict (age, sex, etc.) -> ONLY for CatBoost
        topk (int): Number of recs
        alpha_override (float): Force a specific alpha (0.0 to 1.0). If None, use optimized.
        
    Returns:
        pd.Series: Top-K items with scores
    """
    
    # --- A. BASELINE STRATEGY ---
    if model_type.startswith("Baseline"):
        artifact = load_baseline()
        if not artifact:
            return pd.Series()
            
        owned = context_data.get("owned_products", [])
        return recommend_from_selection(artifact, owned, topk=topk)

    # --- B. CATBOOST STRATEGY ---
    elif model_type.startswith("CatBoost"):
        res = load_hybrid_model()
        if not res:
            return pd.Series()
        
        hybrid_model, best_alpha = res
        
        # Override Alpha if requested
        final_alpha = alpha_override if alpha_override is not None else best_alpha
        
        # Prepare Input DataFrame for CatBoost
        # 1. Create a single row DataFrame
        user_features = context_data.get("user_features", {})
        
        # We assume user_features has keys matching columns (sex, age, etc.)
        # We also need the Product Columns (One-Hot encoded in context)
        owned = context_data.get("owned_products", [])
        
        # Create row
        row_dict = user_features.copy()
        
        # Add Product Columns (0 or 1)
        # We get the list of ALL features expected by the model
        model_feature_names = hybrid_model.trainer.model.feature_names_
        
        # Fill row dict 
        for col in model_feature_names:
            if col in row_dict:
                continue # Already set (e.g. age, sex)
            elif col in owned:
                row_dict[col] = 1 # Owns product
            elif col in hybrid_model.baseline.product_cols:
                row_dict[col] = 0 # Doesn't own product
            elif col == "join_year":
                # Special case: if 'join_year' is missing but needed
                row_dict[col] = 2017 # Default
            else:
                # Other missing feature? 
                # If it's a categorical feature (in CAT_FEATURES), provide '?' or similar if missing
                if col in CAT_FEATURES and col not in row_dict:
                    row_dict[col] = "unknown"
                pass
                
        # Convert to DataFrame
        ctx_df = pd.DataFrame([row_dict])
        
        # Predict
        try:
            probas = hybrid_model.predict_proba(ctx_df, alpha=final_alpha)[0] # Take first row
        except Exception as e:
            st.error(f"Erreur CatBoost: {e}")
            return pd.Series()
        
        # Map to Series
        product_cols = hybrid_model.baseline.product_cols
        
        # Safety check lengths
        if len(probas) != len(product_cols):
            # This can happen if alpha=0.0 (Pure CatBoost) or alpha=1.0 (Pure Baseline) 
            # and shapes differ? No, HybridPredictor handles that.
            st.warning(f"Mismatch scores/products: {len(probas)} vs {len(product_cols)}")
            return pd.Series()
            
        scores = pd.Series(probas, index=product_cols)
        
        # Filter out owned products
        scores = scores.drop(owned, errors='ignore')
        
        return scores.nlargest(topk)
        
    return pd.Series()
