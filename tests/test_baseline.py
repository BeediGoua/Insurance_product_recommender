import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.pipelines.baseline_pipeline import recommend_from_selection, BaselineArtifact

# Mock data for testing
@pytest.fixture
def mock_artifact():
    product_cols = ["ProdA", "ProdB", "ProdC"]
    # P(B|A) matrix
    # A=ProdA -> 0.8 ProdB, 0.1 ProdC
    # A=ProdB -> 0.2 ProdA, 0.5 ProdC
    cond = np.array([
        [0.0, 0.8, 0.1], 
        [0.2, 0.0, 0.5],
        [0.3, 0.4, 0.0]
    ])
    support_A = np.array([100, 100, 100])
    prevalence = np.array([0.3, 0.3, 0.3])
    N = 300
    
    return BaselineArtifact(
        product_cols=product_cols,
        cond=cond,
        support_A=support_A,
        prevalence=prevalence,
        N=N
    )

def test_recommend_empty_selection(mock_artifact):
    """Test that empty selection returns valid recommendations (based on nothing/priors or empty)."""
    recs = recommend_from_selection(mock_artifact, [], topk=3)
    assert len(recs) == 3 
    assert (recs == 0.0).all()

def test_recommend_valid_selection(mock_artifact):
    """Test recommendation with a valid owned product."""
    recs = recommend_from_selection(mock_artifact, ["ProdA"], topk=2)
    
    assert "ProdB" in recs.index
    assert recs["ProdB"] > recs.get("ProdC", 0.0)
    assert "ProdA" not in recs.index

def test_recommend_unknown_product(mock_artifact):
    """Test robustness against unknown product names."""
    recs = recommend_from_selection(mock_artifact, ["UnknownProd", "ProdA"], topk=2)
    
    assert "ProdB" in recs.index
    assert recs["ProdB"] > 0.5

def test_artifact_save_load(tmp_path, mock_artifact):
    """Test artifact persistence."""
    save_dir = tmp_path / "artifact_test"
    mock_artifact.save(save_dir)
    
    loaded = BaselineArtifact.load(save_dir)
    
    assert loaded.product_cols == mock_artifact.product_cols
    np.testing.assert_array_almost_equal(loaded.cond, mock_artifact.cond)
    assert loaded.N == mock_artifact.N
