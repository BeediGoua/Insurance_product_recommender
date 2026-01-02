
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

def prepare_input_for_catboost(user_features, owned_products, model_feature_names, product_cols):
    """
    Robust input preparation for CatBoost:
    1. Calculates 'age' from 'birth_year' if missing (Critical).
    2. Ensures types (int for years, string for categories).
    3. Prevents Data Leakage (removes product columns from user_features).
    4. Sets 1/0 for Owned/Not-Owned products in the context.
    """
    row_dict = {}
    
    # A. Clean User Features (Remove Leakage & Fix Types)
    for k, v in user_features.items():
        # Skip if it is a product column (Leakage protection)
        if k in product_cols:
            continue
            
        # Cast specific numeric fields
        if k in ['join_year', 'birth_year']:
            try:
                row_dict[k] = int(v)
            except:
                row_dict[k] = 2017 # Default
        else:
            row_dict[k] = v

    # B. Feature Engineering (Missing 'age')
    if 'age' not in row_dict:
        if 'birth_year' in row_dict:
            row_dict['age'] = 2020 - row_dict['birth_year']
        else:
            row_dict['age'] = 35 # Default fallback

    # C. Build Final Dict matching Model Expectations
    final_dict = {}
    for feature in model_feature_names:
        # 1. Product Context (One-Hot)
        if feature in product_cols:
            final_dict[feature] = 1 if feature in owned_products else 0
            
        # 2. Known User Feature
        elif feature in row_dict:
            final_dict[feature] = row_dict[feature]
            
        # 3. Missing/Unknown
        else:
            if feature == 'join_year':
                final_dict[feature] = 2017
            elif feature == 'age':
                final_dict[feature] = 40
            else:
                # SAFE FALLBACK
                # CatBoost crashes if numeric feature gets "unknown"
                # We assume features ending in 'code' or specific categories are strings
                if feature in ['sex', 'marital_status', 'branch_code', 'occupation_code', 'occupation_category_code']:
                    final_dict[feature] = "unknown"
                else:
                    # For anything else (likely numeric or product), use 0 or safe default
                    final_dict[feature] = 0

    return pd.DataFrame([final_dict])

def get_recommendations(model_type, context_data, topk=5, alpha_override=None):
    """
    Fonction unifiée pour prédire.
    """
    
    # --- A. BASELINE STRATEGY (Pure Stats) ---
    if model_type.startswith("Baseline"):
        artifact = load_baseline()
        if not artifact:
            return pd.Series()
            
        owned = context_data.get("owned_products", [])
        return recommend_from_selection(artifact, owned, topk=topk)

    # --- B. CATBOOST STRATEGY (Hybrid AI) ---
    elif model_type.startswith("CatBoost"):
        res = load_hybrid_model()
        if not res:
            return pd.Series()
        
        hybrid_model, best_alpha = res
        final_alpha = alpha_override if alpha_override is not None else best_alpha
        
        # Inputs
        user_features = context_data.get("user_features", {})
        owned = context_data.get("owned_products", [])
        
        # Prepare Robust Input
        try:
            # Attempt to access feature names safely
            if hasattr(hybrid_model.model, 'model'):
                # Valid Trainer -> CatBoost structure
                model_names = hybrid_model.model.model.feature_names_
            elif hasattr(hybrid_model.model, 'feature_names_'):
                 # Direct CatBoost model
                model_names = hybrid_model.model.feature_names_
            else:
                 # Fallback / Error
                 st.error(f"Structure Modèle Inconnue: {type(hybrid_model.model)}")
                 return pd.Series()
        except AttributeError as e:
            st.error(f"Erreur d'attribut Modèle: {e}. (Try Clearing Switch Cache)")
            return pd.Series()
            
        prod_cols = hybrid_model.baseline.product_cols
        
        ctx_df = prepare_input_for_catboost(user_features, owned, model_names, prod_cols)
        
        # Predict
        try:
            probas = hybrid_model.predict_proba(ctx_df, alpha=final_alpha)[0]
        except Exception as e:
            st.error(f"Erreur CatBoost: {e}")
            return pd.Series()
        
        # Map Scores
        scores = pd.Series(probas, index=prod_cols)
        scores = scores.drop(owned, errors='ignore')
        
        return scores.nlargest(topk)
        
    return pd.Series()
