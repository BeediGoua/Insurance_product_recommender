import numpy as np
import pandas as pd
from typing import List, Optional
from src.pipelines.baseline_pipeline import BaselineArtifact
from src.baselines.conditional import ConditionalBaseline

class HybridPredictor:
    """
    Combine les prédictions du modèle CatBoost avec les scores de la Baseline (Prior).
    Score = P(CatBoost) * (P(Baseline) ^ alpha)
    """
    def __init__(self, model_catboost, baseline_artifact: BaselineArtifact):
        self.model = model_catboost
        self.baseline = ConditionalBaseline.from_stats(
            baseline_artifact.product_cols,
            baseline_artifact.cond,
            baseline_artifact.support_A
        )
        self.product_cols = baseline_artifact.product_cols
        
    def predict_proba(self, X_eval: pd.DataFrame, alpha: float = 0.0) -> np.ndarray:
        """
        Retourne une matrice de probabilités (N, P).
        
        Args:
            X_eval: DataFrame contenant les features pour CatBoost. 
                    Doit aussi contenir les colonnes produits (0/1) pour la baseline.
            alpha: Poids du prior Baseline (0.0 = Catboost pur, 1.0 = Produit complet).
                   En pratique, alpha entre 0.1 et 0.5 lisse bien.
        """
        # 1. Scores CatBoost (Probabilités)
        # Attention : s'assurer que l'ordre des classes de CatBoost correspond à self.product_cols
        # CatBoost classe par ordre lexicographique ou d'apparition. 
        # C'est un point CRITIQUE. 
        # On va supposer que le DatasetBuilder a encodé les labels 0..20 dans l'ordre de self.product_cols.
        # Donc la classe 0 correspond à product_cols[0].
        
        preds_cb = self.model.predict_proba(X_eval) # (N, 21)
        
        if alpha <= 1e-9:
            return preds_cb
            
        # 2. Scores Baseline
        # On doit itérer pour calculer le score baseline de chaque ligne (vectorisation possible mais complexe avec le masque)
        # La baseline attend un vecteur binaire (produits possédés)
        
        # Astuce : On récupère la matrice numpy des produits depuis X_eval
        # On suppose que X_eval a les colonnes produits
        present_prod_cols = [c for c in self.product_cols if c in X_eval.columns]
        if len(present_prod_cols) != len(self.product_cols):
             # Si les colonnes produits ne sont pas dans X_eval (cas rare en prod pure ?), on ne peut pas calculer le prior
             # Fallback sur CatBoost pur
             return preds_cb
             
        X_products = X_eval[self.product_cols].values
        
        scores_bl_list = []
        for i in range(len(X_eval)):
            # score_one renvoie des scores non-normalisés (somme w * cond)
            # Et met -inf sur les produits possédés.
            # Pour le prior, on veut une probabilité sur les produits MANQUANTS.
            
            # Note : Dans le cas du masking pour entraînement/eval, un produit est masqué (mis à 0 dans X).
            # Donc la baseline va lui donner un score > 0.
            
            raw_scores = self.baseline.score_one(X_products[i])
            
            # Transformation en pseudo-proba (Softmax ou simple division ?)
            # La baseline additive n'est pas des probas strictes.
            # On va remplacer les -inf par min_val pour ne pas casser le calcul
            valid_mask = np.isfinite(raw_scores)
            
            if not np.any(valid_mask):
                # Cas dégénéré (aucun produit ou tout possédé ?)
                scores_bl_list.append(np.ones(len(self.product_cols)) / len(self.product_cols))
                continue
                
            min_val = raw_scores[valid_mask].min() if np.any(valid_mask) else -1.0
            safe_scores = raw_scores.copy()
            safe_scores[~valid_mask] = min_val - 1.0 # Très bas
            
            # Simple normalisation min-max ou sum sur les positifs ?
            # Pour un prior multiplicatif, on veut des valeurs dans [0, 1].
            # Softmax est robuste.
            exps = np.exp(safe_scores - safe_scores.max()) # stable softmax
            probs = exps / exps.sum()
            scores_bl_list.append(probs)
            
        preds_bl = np.array(scores_bl_list)
        
        # 3. Fusion
        # P_final = P_cb * (P_bl ^ alpha)
        # Puis re-normalisation
        mixed = preds_cb * np.power(preds_bl, alpha)
        
        # Renormaliser ligne par ligne
        row_sums = mixed.sum(axis=1, keepdims=True)
        # Éviter division par 0
        row_sums[row_sums == 0] = 1.0
        
        return mixed / row_sums
