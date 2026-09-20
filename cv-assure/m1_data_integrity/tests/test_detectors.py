"""Tests for M1 capability-gated integrity detectors."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CV_ASSURE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CV_ASSURE))
sys.path.insert(0, str(ROOT))

from detectors import (  # noqa: E402
    ActivationClusteringDetector,
    AnnotationAnomalyDetector,
    LabelFlipDetector,
    MislabelDetector,
    NearDuplicateDetector,
    OODInsertionDetector,
    SpectralSignatureDetector,
)
from schema import FindingType  # noqa: E402


def _matrix(**overrides):
    payload = {
        "access_tier": "T0",
        "task_type": "DETECTION",
        "reference_mode": "BOOTSTRAPPED",
    }
    payload.update(overrides)
    return payload


def _record(asset_id, contributor_id, path, bbox=None, label="car", width=100, height=80):
    anns = []
    if bbox is not None:
        anns = [{"bbox": bbox, "category_id": 0, "class_name": label}]
    elif label:
        anns = [{"bbox": [1, 1, 10, 10], "category_id": 0, "class_name": label}]
    return SimpleNamespace(
        asset_id=asset_id,
        contributor_id=contributor_id,
        file_path=path,
        annotations=anns,
        image_metadata={"width": width, "height": height, "channels": 3},
        class_name=label,
    )


def _png(path: Path, color=(40, 80, 120)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), color).save(path)
    return path


def test_annotation_detector_flags_impossible_boxes():
    records = [
        _record("ok", "c1", "/tmp/a.png", bbox=[10, 10, 20, 20]),
        _record("wide", "c1", "/tmp/b.png", bbox=[0, 0, 99, 2]),  # 49.5:1
        _record("empty", "c1", "/tmp/c.png", bbox=[5, 5, 0, 8]),
        _record("off", "c1", "/tmp/d.png", bbox=[200, 0, 10, 10]),
    ]
    result = AnnotationAnomalyDetector().run(records, _matrix())
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert ids == {"wide", "empty", "off"}
    assert all(item.finding_type is FindingType.ANNOTATION_ANOMALY for item in result["anomalies"])


def test_annotation_detector_unavailable_for_classification():
    result = AnnotationAnomalyDetector().run([], _matrix(task_type="CLASSIFICATION"))
    assert result["status"] == "UNAVAILABLE"
    assert "UNAVAILABLE_FOR_TASK_TYPE" in result["reason"]
    assert result["anomalies"] == []


def test_near_duplicate_detector_flags_identical_images(tmp_path: Path):
    a = _png(tmp_path / "a.png", (10, 20, 30))
    b = _png(tmp_path / "b.png", (10, 20, 30))
    c = _png(tmp_path / "c.png", (200, 10, 10))
    records = [
        _record("a", "c1", a),
        _record("b", "c2", b),
        _record("c", "c1", c),
    ]
    result = NearDuplicateDetector().run(records, _matrix())
    assert result["status"] == "OK"
    flagged = {item.asset_id for item in result["anomalies"]}
    assert "a" in flagged
    assert all(item.finding_type is FindingType.NEAR_DUPLICATE_FLOOD for item in result["anomalies"])


def test_label_flip_knn_disagreement():
    rng = np.random.default_rng(0)
    cluster_a = [rng.normal(0, 0.05, size=8) for _ in range(8)]
    cluster_b = [rng.normal(4, 0.05, size=8) for _ in range(8)]
    flipped = cluster_b[0]  # sits in B but we label as A
    records = []
    embeddings = {}
    for i, vec in enumerate(cluster_a):
        rid = f"a{i}"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label="cat"))
        embeddings[rid] = {"ref_emb": vec}
    for i, vec in enumerate(cluster_b):
        rid = f"b{i}"
        label = "cat" if i == 0 else "dog"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label=label))
        embeddings[rid] = {"ref_emb": vec}
    result = LabelFlipDetector(k=5, min_agreement=0.5).run(records, _matrix(), embeddings=embeddings)
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert "b0" in ids
    assert result["anomalies"][0].finding_type is FindingType.LABEL_FLIP
    _ = flipped


def test_ood_unavailable_without_reference():
    result = OODInsertionDetector().run([], _matrix(reference_mode="UNREFERENCED"))
    assert result["status"] == "UNAVAILABLE"
    assert "UNAVAILABLE_WITHOUT_REFERENCE" in result["reason"]


def test_ood_flags_far_sample():
    rng = np.random.default_rng(1)
    records = []
    embeddings = {}
    for i in range(20):
        rid = f"in{i}"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label="car"))
        embeddings[rid] = {"ref_emb": rng.normal(0, 0.1, size=6)}
    records.append(_record("ood", "c2", "/tmp/ood.png", label="car"))
    embeddings["ood"] = {"ref_emb": np.ones(6) * 8.0}
    result = OODInsertionDetector(percentile=90.0).run(
        records, _matrix(reference_mode="BOOTSTRAPPED"), embeddings=embeddings
    )
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert "ood" in ids
    assert all(item.finding_type is FindingType.OOD_INSERTION for item in result["anomalies"])


def test_spectral_flags_aligned_outlier():
    rng = np.random.default_rng(2)
    base = rng.normal(0, 0.2, size=(12, 8))
    spike = np.zeros(8)
    spike[0] = 12.0
    records = []
    embeddings = {}
    for i, vec in enumerate(base):
        rid = f"n{i}"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label="person"))
        embeddings[rid] = {"ref_emb": vec}
    records.append(_record("trig", "c1", "/tmp/trig.png", label="person"))
    embeddings["trig"] = {"ref_emb": spike}
    result = SpectralSignatureDetector(z_threshold=2.5).run(records, _matrix(), embeddings=embeddings)
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert "trig" in ids
    assert all(item.finding_type is FindingType.TRIGGER_INJECTION for item in result["anomalies"])


def test_activation_clustering_flags_minority_cluster():
    rng = np.random.default_rng(3)
    clean = rng.normal(0, 0.1, size=(12, 6))
    poison = rng.normal(6, 0.1, size=(4, 6))
    records = []
    embeddings = {}
    for i, vec in enumerate(clean):
        rid = f"c{i}"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label="truck"))
        embeddings[rid] = {"ref_emb": vec}
    for i, vec in enumerate(poison):
        rid = f"p{i}"
        records.append(_record(rid, "c1", f"/tmp/{rid}.png", label="truck"))
        embeddings[rid] = {"ref_emb": vec}
    result = ActivationClusteringDetector(max_minority_fraction=0.4, min_separation=2.0).run(
        records, _matrix(), embeddings=embeddings
    )
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert ids == {"p0", "p1", "p2", "p3"}
    assert all(item.finding_type is FindingType.TRIGGER_INJECTION for item in result["anomalies"])


def test_mislabel_detector_flags_class_swap_campaign():
    rng = np.random.default_rng(4)
    cats = [rng.normal(0, 0.1, size=6) for _ in range(8)]
    dogs = [rng.normal(5, 0.1, size=6) for _ in range(8)]
    records = []
    embeddings = {}
    for i, vec in enumerate(cats):
        rid = f"cat{i}"
        records.append(_record(rid, "honest", f"/tmp/{rid}.png", label="cat"))
        embeddings[rid] = {"ref_emb": vec}
    for i, vec in enumerate(dogs):
        rid = f"dog{i}"
        label = "cat" if i < 4 else "dog"
        contributor = "attacker" if i < 4 else "honest"
        records.append(_record(rid, contributor, f"/tmp/{rid}.png", label=label))
        embeddings[rid] = {"ref_emb": vec}
    result = MislabelDetector(margin=1.1, min_campaign=3).run(records, _matrix(), embeddings=embeddings)
    assert result["status"] == "OK"
    ids = {item.asset_id for item in result["anomalies"]}
    assert {"dog0", "dog1", "dog2", "dog3"} <= ids
    assert all(item.finding_type is FindingType.SYSTEMATIC_MISLABEL for item in result["anomalies"])
    assert all(item.contributor_id == "attacker" for item in result["anomalies"])
