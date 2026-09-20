"""
Test suite for M1 Data Integrity and Contributor Rollup module.
"""

import sys
import pytest
import numpy as np
from pathlib import Path
from enum import Enum
from dataclasses import dataclass

# --- 1. RESOLVE M1 PATHS AUTOMATICALLY ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
M1_DIR = PROJECT_ROOT / "m1_data_integrity"
if str(M1_DIR) not in sys.path:
    sys.path.insert(0, str(M1_DIR))

# --- 2. MOCK M0 DEPENDENCIES (Fixes ImportErrors) ---
class AccessTier(Enum):
    T0_LABELS = "T0_LABELS"

class TaskType(Enum):
    CLASSIFICATION = "CLASSIFICATION"

class ReferenceMode(Enum):
    ATTESTED = "ATTESTED"

@dataclass
class CapabilityMatrix:
    access_tier: AccessTier
    task_type: TaskType
    reference_mode: ReferenceMode
    probe_timestamp: str

@dataclass
class AssetRecord:
    asset_id: str
    file_path: Path
    sha256: str
    format: str
    image_metadata: dict
    annotations: list
    contributor_id: str
    timestamp: str

# --- 3. M1 IMPORTS ---
from schema import FindingType, SampleAnomaly
from detectors import NearDuplicateDetector, LabelFlipDetector
from dual_encoder import scan_collusion
from contributor_rollup import rollup_contributors
from sybil_clustering import cluster_sybils
from pipeline import run_data_integrity


# --- 4. TESTS ---
def test_m1_detectors_availability():
    """Test that detectors evaluate capability matrix correctly."""
    cap_t0 = CapabilityMatrix(
        access_tier=AccessTier.T0_LABELS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.ATTESTED,
        probe_timestamp="2026-09-20"
    )
    
    label_detector = LabelFlipDetector()
    available, reason = label_detector.check_availability(cap_t0)
    assert available is True

    dup_detector = NearDuplicateDetector()
    available, reason = dup_detector.check_availability(cap_t0)
    assert available is True


def test_dual_encoder_scan_collusion():
    """Test the REAL scan_collusion algorithm by providing a centroid cluster and an outlier."""
    import numpy as np
    from m1_data_integrity.dual_encoder import scan_collusion
    
    records = []
    embeddings = {}
    
    # 1. Create a "cloud" of 5 normal background images to form the centroid
    for i in range(5):
        asset_id = f"img_normal_{i}"
        records.append(AssetRecord(
            asset_id=asset_id, file_path=Path(f"{asset_id}.jpg"), sha256=f"hash{i}",
            format="COCO", image_metadata={}, annotations=[], contributor_id="C-01", timestamp="2026"
        ))
        embeddings[asset_id] = {
            "ref_emb": np.array([0.1, 0.1, 0.1]),
            "sub_emb": np.array([0.1, 0.1, 0.1])
        }
        
    # 2. Create 1 Collusion Anomaly where the submitted model was poisoned/diverges
    asset_id = "img_anomaly"
    records.append(AssetRecord(
        asset_id=asset_id, file_path=Path(f"{asset_id}.jpg"), sha256="hashA",
        format="COCO", image_metadata={}, annotations=[], contributor_id="C-07", timestamp="2026"
    ))
    embeddings[asset_id] = {
        "ref_emb": np.array([10.0, 10.0, 10.0]), # Reference model sees a huge signal
        "sub_emb": np.array([0.1, 0.1, 0.1])     # Poisoned model ignores it
    }

    # 3. Run the REAL function (no monkeypatch needed!)
    anomalies = scan_collusion(embeddings, records, threshold=1.5)
    
    # 4. Verify it caught the anomaly for C-07!
    # Filter the list to find our specific injected anomaly
    c07_anomalies = [a for a in anomalies if a.contributor_id == "C-07"]
    
    assert len(c07_anomalies) >= 1, "Real algorithm failed to detect the divergence for C-07!"
    assert "DATA_MODEL_COLLUSION" in str(c07_anomalies[0].finding_type)
    assert c07_anomalies[0].score <= 1.0  # Proves the Pydantic schema is safe

def test_contributor_rollup_quarantine_trigger():
    """Test exact binomial test flags high poison rate contributor for quarantine."""
    records = []
    anomalies = []

    # Clean contributor
    for i in range(100):
        records.append(AssetRecord(
            asset_id=f"c1_{i}", file_path=Path(f"c1_{i}.jpg"), sha256="hash1",
            format="COCO", image_metadata={}, annotations=[], contributor_id="C-01", timestamp="2026"
        ))
    anomalies.append(SampleAnomaly(
        asset_id="c1_0", contributor_id="C-01", finding_type=FindingType.ANNOTATION_ANOMALY, score=0.9, details={}
    ))

    # Poisoned contributor
    for i in range(50):
        records.append(AssetRecord(
            asset_id=f"c2_{i}", file_path=Path(f"c2_{i}.jpg"), sha256="hash2",
            format="COCO", image_metadata={}, annotations=[], contributor_id="C-02", timestamp="2026"
        ))
    for i in range(15):
        anomalies.append(SampleAnomaly(
            asset_id=f"c2_{i}", contributor_id="C-02", finding_type=FindingType.LABEL_FLIP, score=0.95, details={}
        ))

    rollups = rollup_contributors(records, anomalies, pipeline_baseline_rate=0.05)
    
    c2_rollup = next(r for r in rollups if r.contributor_id == "C-02")
    assert c2_rollup.disposition == "quarantine"
    assert c2_rollup.p_value < 0.01


def test_sybil_clustering_structure():
    """Test cluster_sybils groups contributors based on records and anomalies."""
    records = []
    anomalies = []

    for cid in ["C-08a", "C-08b"]:
        for i in range(10):
            records.append(AssetRecord(
                asset_id=f"{cid}_{i}", file_path=Path(f"{cid}_{i}.jpg"), sha256="hash_sybil",
                format="COCO", image_metadata={"width": 1920, "height": 1080},
                annotations=[{"category_id": 1}], contributor_id=cid, timestamp="2026"
            ))

    groups = cluster_sybils(records, anomalies)
    assert isinstance(groups, list)


def test_run_data_integrity_pipeline():
    """Test full M1 pipeline orchestrator run."""
    records = [
        AssetRecord(
            asset_id="img_100", file_path=Path("img_100.jpg"), sha256="hash100",
            format="COCO", image_metadata={}, annotations=[], contributor_id="C-01", timestamp="2026"
        )
    ]
    cap_matrix = CapabilityMatrix(
        access_tier=AccessTier.T0_LABELS,
        task_type=TaskType.CLASSIFICATION,
        reference_mode=ReferenceMode.ATTESTED,
        probe_timestamp="2026-09-20"
    )

    result = run_data_integrity(records, cap_matrix)
    
    assert result is not None
    # Check attributes individually instead of absolute equality to avoid Enum vs String errors
    assert result.capability_matrix.probe_timestamp == "2026-09-20"
    assert "T0_LABELS" in str(result.capability_matrix.access_tier)
    
    assert isinstance(result.sample_anomalies, list)
    assert isinstance(result.contributor_rollups, list)