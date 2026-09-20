"""Tests for contributor rollups and Sybil clustering."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV_ASSURE))
sys.path.insert(0, str(ROOT))

from contributor_rollup import attach_sybil_ids, rollup_contributors  # noqa: E402
from schema import FindingType, SampleAnomaly  # noqa: E402
from sybil_clustering import cluster_sybils  # noqa: E402


def _record(asset_id: str, contributor_id: str, label: str = "car"):
    return SimpleNamespace(
        asset_id=asset_id,
        contributor_id=contributor_id,
        annotations=[{"bbox": [1, 1, 10, 10], "class_name": label}],
        class_name=label,
    )


def _anomaly(asset_id: str, contributor_id: str, finding=FindingType.LABEL_FLIP, **details):
    return SampleAnomaly(
        asset_id=asset_id,
        contributor_id=contributor_id,
        finding_type=finding,
        score=0.8,
        details=details or {"class_name": "car"},
    )


def test_clean_contributor_is_accepted():
    records = [_record(f"a{i}", "clean") for i in range(40)]
    anomalies = [_anomaly("a0", "clean")]
    rollups = {item.contributor_id: item for item in rollup_contributors(records, anomalies, 0.05)}
    item = rollups["clean"]
    assert item.flagged_samples == 1
    assert item.flag_rate == pytest.approx(1 / 40)
    assert item.disposition == "accept"
    assert item.p_value > 0.05


def test_poisoned_contributor_is_quarantined():
    records = [_record(f"p{i}", "bad", label="stop") for i in range(50)]
    anomalies = [_anomaly(f"p{i}", "bad", class_name="stop") for i in range(20)]
    rollups = {item.contributor_id: item for item in rollup_contributors(records, anomalies, 0.05)}
    item = rollups["bad"]
    assert item.flagged_samples == 20
    assert item.disposition == "quarantine"
    assert item.p_value < 0.01
    assert item.estimated_poison_rate > 0.2
    assert "stop" in item.concentrated_classes


def test_duplicate_asset_flags_count_once():
    records = [_record("x", "c1"), _record("y", "c1")]
    anomalies = [
        _anomaly("x", "c1", FindingType.LABEL_FLIP),
        _anomaly("x", "c1", FindingType.OOD_INSERTION),
    ]
    item = rollup_contributors(records, anomalies)[0]
    assert item.flagged_samples == 1
    assert item.total_samples == 2


def test_sybil_groups_from_cross_contributor_duplicates():
    records = [_record("a", "alice"), _record("b", "bob"), _record("c", "carol")]
    match_ab = {
        "matches": [
            {"other_asset_id": "b", "other_contributor_id": "bob", "cross_contributor": True},
            {"other_asset_id": "b2", "other_contributor_id": "bob", "cross_contributor": True},
        ]
    }
    match_ba = {
        "matches": [
            {"other_asset_id": "a", "other_contributor_id": "alice", "cross_contributor": True},
            {"other_asset_id": "a2", "other_contributor_id": "alice", "cross_contributor": True},
        ]
    }
    anomalies = [
        _anomaly("a", "alice", FindingType.NEAR_DUPLICATE_FLOOD, **match_ab),
        _anomaly("b", "bob", FindingType.NEAR_DUPLICATE_FLOOD, **match_ba),
        _anomaly("c", "carol", FindingType.LABEL_FLIP, class_name="car"),
    ]
    groups = cluster_sybils(records, anomalies, min_cross_duplicate_edges=2)
    assert ["alice", "bob"] in groups
    assert all("carol" not in group for group in groups)


def test_attach_sybil_ids_writes_cluster_field():
    records = [_record(f"a{i}", "alice") for i in range(10)] + [_record(f"b{i}", "bob") for i in range(10)]
    anomalies = [_anomaly(f"a{i}", "alice") for i in range(4)] + [_anomaly(f"b{i}", "bob") for i in range(4)]
    rollups = rollup_contributors(records, anomalies)
    updated = attach_sybil_ids(rollups, [["alice", "bob"]])
    by_id = {item.contributor_id: item for item in updated}
    assert by_id["alice"].sybil_cluster_id == "sybil-0"
    assert by_id["bob"].sybil_cluster_id == "sybil-0"
