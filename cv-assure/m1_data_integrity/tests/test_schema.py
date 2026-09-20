"""Tests for M1 Clause 2.2.1 schemas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV_ASSURE))
sys.path.insert(0, str(ROOT))

from schema import (  # noqa: E402
    CapabilityMatrix,
    ContributorRollup,
    FindingType,
    M1Result,
    SampleAnomaly,
)


def _anomaly(**overrides) -> SampleAnomaly:
    payload = {
        "asset_id": "img-1",
        "contributor_id": "c1",
        "finding_type": FindingType.DATA_MODEL_COLLUSION,
        "score": 0.82,
        "details": {"e_ref_sub_cosine": 0.11, "mahalanobis": 4.2},
    }
    payload.update(overrides)
    return SampleAnomaly.from_dict(payload)


def test_finding_types_cover_clause_221():
    names = {item.name for item in FindingType}
    assert names == {
        "TRIGGER_INJECTION",
        "LABEL_FLIP",
        "SYSTEMATIC_MISLABEL",
        "NEAR_DUPLICATE_FLOOD",
        "OOD_INSERTION",
        "ANNOTATION_ANOMALY",
        "DATA_MODEL_COLLUSION",
    }


def test_sample_anomaly_rejects_score_outside_unit_interval():
    with pytest.raises(Exception):
        _anomaly(score=1.2)


def test_contributor_rollup_json_roundtrip():
    rollup = ContributorRollup(
        contributor_id="c1",
        total_samples=100,
        flagged_samples=18,
        flag_rate=0.18,
        pipeline_baseline_rate=0.04,
        p_value=0.001,
        estimated_poison_rate=0.14,
        concentrated_classes=["stop_sign"],
        disposition="quarantine",
        sybil_cluster_id="sybil-7",
    )
    restored = ContributorRollup.from_json(rollup.to_json())
    assert restored.disposition == "quarantine"
    assert restored.sybil_cluster_id == "sybil-7"
    assert restored.to_dict()["flagged_samples"] == 18


def test_rollup_rejects_flagged_above_total():
    with pytest.raises(Exception):
        ContributorRollup(
            contributor_id="c1",
            total_samples=2,
            flagged_samples=3,
            flag_rate=1.0,
            pipeline_baseline_rate=0.1,
            p_value=0.5,
            estimated_poison_rate=0.5,
            concentrated_classes=[],
            disposition="review",
        )


def test_m1_result_roundtrip_and_matrix_coercion():
    class _M0Matrix:
        def __init__(self) -> None:
            self.access_tier = type("T", (), {"value": "T0"})()
            self.task_type = "DETECTION"
            self.reference_mode = "BOOTSTRAPPED"
            self.source = "local_file"
            self.metadata = {"k": 1}

        def to_dict(self) -> dict:
            return {
                "access_tier": self.access_tier.value,
                "task_type": self.task_type,
                "reference_mode": self.reference_mode,
                "source": self.source,
                "metadata": self.metadata,
            }

    result = M1Result(
        capability_matrix=_M0Matrix(),
        sample_anomalies=[_anomaly()],
        contributor_rollups=[],
        sybil_suspect_groups=[["c1", "c2"]],
    )
    assert result.capability_matrix.access_tier == "T0"
    assert result.sample_anomalies[0].finding_type is FindingType.DATA_MODEL_COLLUSION
    restored = M1Result.from_json(result.to_json())
    assert restored.sybil_suspect_groups == [["c1", "c2"]]
    assert restored.capability_matrix.source == "local_file"


def test_capability_matrix_from_plain_dict():
    matrix = CapabilityMatrix.from_m0(
        {"access_tier": "T2", "task_type": "CLASSIFICATION", "reference_mode": "ATTESTED"}
    )
    assert matrix.access_tier == "T2"
    assert matrix.model_digest is None
