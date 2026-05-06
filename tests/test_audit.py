import os
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import tempfile
from src.decisionflow.schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult
from src.decisionflow.audit import create_audit_record, save_audit_record, load_audit_records


def test_audit_save_and_load():
    profile = ClientProfile(client_id="C2")
    rec = RecommendationResult(
        client_id="C2",
        raw_scores={"Life": 0.8},
        filtered_scores={"Life": 0.8},
        top_k=["Life"],
    )
    policy = PolicyDecision(allowed=["Life"], blocked=[])
    expl = [ExplanationResult(product="Life", reasons=["dummy reason"], limitations=[])]
    risk = {"confidence": 0.9, "risk_level": "low", "manual_review_required": False}
    record = create_audit_record(profile, rec, policy, expl, risk)
    with tempfile.TemporaryDirectory() as tmp:
        log_path = pathlib.Path(tmp) / "audit.jsonl"
        save_audit_record(record, log_path)
        loaded = load_audit_records(log_path)
        assert len(loaded) == 1
        assert loaded[0].client_id == "C2"