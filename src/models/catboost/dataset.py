import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass

@dataclass
class CatboostDatasetBuilder:
    """
    Construit le dataset d'entraînement pour CatBoost Multiclass.
    Stratégie : Masking (on cache 1 produit possédé et on demande de le retrouver).
    """
    product_cols: List[str]
    cat_cols: List[str]
    random_state: int = 42

    def build_dataset(self, df: pd.DataFrame, strategy: str = 'smart', min_basket: int = 2) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Génère X (features) et y (target = index du produit masqué).
        
        Args:
            df: DataFrame nettoyé (1 ligne par client).
            strategy: 'uniform' (choix aléatoire) ou 'smart' (inverse frequency).
            min_basket: Minimum de produits possédés pour être éligible.
        
        Returns:
            X: DataFrame augmenté (N_samples, features)
            y: Series (N_samples,) avec l'index du produit masqué (0..20)
        """
        rng = np.random.default_rng(self.random_state)
        
        # 1. Calculer les statistiques pour le smart masking
        # Fréquence globale de chaque produit pour pondérer le choix
        if strategy == 'smart':
            freqs = df[self.product_cols].mean().values
            # Poids inverse (plus un produit est rare, plus on a envie de le masquer pour l'apprendre)
            # On ajoute un petit epsilon et on prend la puissance pour accentuer ou lisser
            weights = 1.0 / (freqs + 1e-4)
            weights = weights / weights.sum()
        
        # 2. Identifier les éligibles
        X_vals = df[self.product_cols].values
        basket_sizes = X_vals.sum(axis=1)
        eligible_indices = np.where(basket_sizes >= min_basket)[0]
        
        X_rows = []
        y_rows = []
        
        # 3. Génération des samples
        # On peut vectoriser ou boucler. Vu la taille (milliers de clients), boucler en Python est acceptable
        # si on pré-alloue, mais ici on va faire une liste de dicts pour la lisibilité
        
        # Optimisation : récupérer les numpy arrays pour la boucle
        df_reset = df.iloc[eligible_indices].reset_index(drop=True)
        products_arr = df_reset[self.product_cols].values
        profile_df = df_reset.drop(columns=self.product_cols) # Garde ID, cat_cols, etc.
        
        # Map product name -> index
        prod_map = {p: i for i, p in enumerate(self.product_cols)}
        
        for i in range(len(df_reset)):
            owned_mask = products_arr[i] == 1
            owned_indices = np.where(owned_mask)[0]
            
            if len(owned_indices) == 0: continue
                
            # Choisir le produit à masquer
            if strategy == 'smart':
                # Proba proportionnelle au poids inverse
                w_local = weights[owned_indices]
                w_local = w_local / w_local.sum()
                hidden_idx = rng.choice(owned_indices, p=w_local)
            else:
                hidden_idx = rng.choice(owned_indices)
            
            # Créer la ligne X
            # On copie le profil
            row = profile_df.iloc[i].to_dict()
            # On ajoute les produits (avec le masqué à 0)
            current_prods = products_arr[i].copy()
            current_prods[hidden_idx] = 0
            
            # Ajouter les features produits au dict
            for p_idx, p_name in enumerate(self.product_cols):
                row[p_name] = int(current_prods[p_idx])
            
            X_rows.append(row)
            y_rows.append(int(hidden_idx))
            
        X_out = pd.DataFrame(X_rows)

        # Retrait explicite des colonnes non-features (ID, date) qui font planter CatBoost
        # si elles ne sont pas déclarées dans cat_features
        cols_to_drop = ["ID", "join_date"]
        X_out.drop(columns=[c for c in cols_to_drop if c in X_out.columns], inplace=True, errors="ignore")

        y_out = pd.Series(y_rows, name='target')
        
        # S'assurer que les colonnes sont dans le bon ordre (profil + produits)
        # On veut surtout que les cat_cols soient présentes
        return X_out, y_out
