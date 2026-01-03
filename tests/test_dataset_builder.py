import pytest
import pandas as pd
import numpy as np
import sys
import os

# Ajouter le root au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.catboost.dataset import CatboostDatasetBuilder

def test_dataset_builder_shapes():
    # Données mock
    data = {
        'ID': [1, 2, 3],
        'join_date': ['2018', '2019', '2020'],
        'P1': [1, 0, 1], # Produit 1
        'P2': [1, 1, 1], # Produit 2
        'cat_feat': ['a', 'b', 'a']
    }
    df = pd.DataFrame(data)
    
    product_cols = ['P1', 'P2']
    cat_cols = ['cat_feat']
    
    builder = CatboostDatasetBuilder(product_cols=product_cols, cat_cols=cat_cols)
    
    # Avec min_basket=2, seul ID=1 et ID=3 (P1=1, P2=1 -> sum=2) sont éligibles
    # ID=2 a P1=0, P2=1 -> sum=1 -> exclu
    
    X, y = builder.build_dataset(df, min_basket=2, strategy='uniform')
    
    # On attend 2 samples
    assert len(X) == 2
    assert len(y) == 2
    
    # Vérifier que les colonnes produits sont bien dans X
    assert 'P1' in X.columns
    assert 'P2' in X.columns
    assert 'cat_feat' in X.columns
    
    # Vérifier le masking: Pour chaque ligne, le produit cible doit être 0 dans X
    for idx in range(len(X)):
        target_idx = y.iloc[idx]
        target_prod_name = product_cols[target_idx]
        assert X.iloc[idx][target_prod_name] == 0, f"Le produit cible {target_prod_name} devrait être masqué à 0"

def test_dataset_builder_smart_strategy():
    # Test simple pour voir si ça ne crash pas
    data = {
        'ID': range(10),
        'P1': [1]*10,
        'P2': [1]*10,
        'cat': ['a']*10
    }
    df = pd.DataFrame(data)
    builder = CatboostDatasetBuilder(product_cols=['P1', 'P2'], cat_cols=['cat'])
    X, y = builder.build_dataset(df, min_basket=2, strategy='smart')
    assert len(X) == 10
