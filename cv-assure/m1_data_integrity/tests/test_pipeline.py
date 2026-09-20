"""Tests for the M1 orchestrator."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV_ASSURE))
sys.path.insert(0, str(ROOT))

from pipeline import run_data_integrity  # noqa: E402
from schema import FindingType, M1Result  # noqa: E402


def _matrix(**overrides):
    payload = {
        "access_tier": "T0",
        "task_type": "DETECTION",
        "reference_mode": "BOOTSTRAPPED",
    }
    payload.update(overrides)
    return payload


def _record(asset_id, contributor_id, bbox, label="car", width=100, height=80):
    return SimpleNamespace(
        asset_id=asset_id,
        contributor_id=contributor_id,
        file_path=f"/tmp/{asset_id}.png",
        annotations=[{"bbox": bbox, "category_id": 0, "class_name": label}],
        image_metadata={"width": width, "height": height, "channels": 3},
        class_name=label,
    )


def test_pipeline_collects_annotation_findings_and_rollups():
    records = [
        _record("ok", "c1", [10, 10, 20, 20]),
        _record("wide", "c2", [0, 0, 99, 2]),
        _record("empty", "c2", [5, 5, 0, 8]),
        _record("off", "c2", [200, 0, 10, 10]),
    ]
    result = run_data_integrity(records, _matrix())
    assert isinstance(result, M1Result)
    ids = {item.asset_id for item in result.sample_anomalies}
    assert {"wide", "empty", "off"} <= ids
    by_contrib = {item.contributor_id: item for item in result.contributor_rollups}
    assert by_contrib["c2"].flagged_samples >= 3
    assert by_contrib["c2"].disposition in {"review", "quarantine"}
    restored = M1Result.from_json(result.to_json())
    assert restored.capability_matrix.access_tier == "T0"


def test_pipeline_skips_annotation_on_classification():
    records = [_record("wide", "c1", [0, 0, 99, 2])]
    result = run_data_integrity(records, _matrix(task_type="CLASSIFICATION"))
    assert all(item.finding_type is not FindingType.ANNOTATION_ANOMALY for item in result.sample_anomalies)
    json.loads(result.to_json())
