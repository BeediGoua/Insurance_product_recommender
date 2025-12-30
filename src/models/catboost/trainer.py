import pandas as pd
import numpy as np
from pathlib import Path
from catboost import CatBoostClassifier, Pool
from typing import List, Optional

class CatboostTrainer:
    """
    Wrapper pour l'entraînement et la gestion du modèle CatBoost.
    """
    def __init__(self, cat_features: List[str], iterations: int = 1000, learning_rate: float = 0.05, depth: int = 6):
        self.cat_features = cat_features
        self.params = {
            'iterations': iterations,
            'learning_rate': learning_rate,
            'depth': depth,
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy', # On peut aussi utiliser MultiClass
            'random_seed': 42,
            'allow_writing_files': False, # Évite de polluer le dossier avec catboost_info
            'verbose': 100
        }
        self.model = CatBoostClassifier(**self.params)
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, 
            X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None,
            class_weights: Optional[List[float]] = None):
        """
        Entraîne le modèle.
        """
        if class_weights is not None:
             self.model.set_params(class_weights=class_weights)

        # Identifier les indices des colonnes catégorielles car CatBoost préfère parfois les noms, parfois les indices
        # Si X_train est un DataFrame, passer les noms dans cat_features est OK pour le constructeur
        # mais on doit s'assurer qu'elles sont dans X_train
        present_cat_features = [c for c in self.cat_features if c in X_train.columns]
        
        train_pool = Pool(X_train, y_train, cat_features=present_cat_features)
        
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = Pool(X_val, y_val, cat_features=present_cat_features)
            
        self.model.fit(train_pool, eval_set=eval_pool, early_stopping_rounds=50)
        
    def save(self, path: Path):
        """Sauvegarde au format CBM (CatBoost Binary Model)"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(path))
        
    def load(self, path: Path):
        self.model.load_model(str(path))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)
