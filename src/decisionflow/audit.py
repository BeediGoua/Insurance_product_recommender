"""
Audit utilities for the DecisionFlow system.

Audit logs provide traceability for recommendations made by the system.
Each record captures the inputs, outputs and context for a single
recommendation event.  The logs can be stored in a file or passed to
an external monitoring system.  The default implementation writes
records to a JSON Lines file in the ``artifacts`` directory.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from pathlib import Path

from .schemas import ClientProfile, RecommendationResult, PolicyDecision, ExplanationResult, AuditRecord

DEFAULT_AUDIT_LOG_PATH = Path("artifacts/audit_logs.jsonl")


def create_audit_record(
    profile: ClientProfile,
    rec: RecommendationResult,
    policy: PolicyDecision,
    explanations: List[ExplanationResult],
    risk: Dict[str, Any],
    model_version: str = "unknown",
    policy_version: str = "v1",
) -> AuditRecord:
    """Construct an :class:`AuditRecord` from the inputs and outputs.

    The timestamp is generated at call time in ISO8601 format.
    """
    ts = datetime.now().isoformat()
    return AuditRecord(
        timestamp=ts,
        client_id=profile.client_id,
        input_profile={
            "segment": profile.segment,
            "current_products": profile.current_products,
            "needs_signals": profile.needs_signals,
            "data_quality": profile.data_quality,
            "extra_info": profile.extra_info,
        },
        raw_model_scores=rec.raw_scores,
        rules_triggered=list(policy.reasons.values()),
        final_recommendations=policy.allowed,
        explanation="; ".join([" | ".join(e.reasons) for e in explanations]) if explanations else None,
        model_version=model_version,
        policy_version=policy_version,
    )


def save_audit_record(record: AuditRecord, log_path: Path = DEFAULT_AUDIT_LOG_PATH) -> None:
    """Append the audit record to a JSON Lines file.

    Parameters
    ----------
    record: AuditRecord
        The record to be persisted.
    log_path: Path
        Location of the audit log file.  Parent directories will be
        created as needed.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        json.dump(record.__dict__, f, default=str)
        f.write("\n")


def load_audit_records(log_path: Path = DEFAULT_AUDIT_LOG_PATH) -> List[AuditRecord]:
    """Load all audit records from the log file.

    Returns an empty list if the file does not exist.  Each line is
    parsed into an :class:`AuditRecord`.  Any errors in parsing are
    silently ignored to prevent the entire load failing on a single
    corrupt line.
    """
    records: List[AuditRecord] = []
    if not log_path.exists():
        return records
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                records.append(AuditRecord(**data))
            except Exception:
                continue
    return records
