"""Module M1 orchestrator: capability-gated detectors, rollup, Sybil groups."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
_M0_SRC = _CV_ASSURE / "m0_ingest_capability" / "src"
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT), str(_M0_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from contributor_rollup import attach_sybil_ids, rollup_contributors  # noqa: E402
from detectors import (  # noqa: E402
    ActivationClusteringDetector,
    AnnotationAnomalyDetector,
    LabelFlipDetector,
    MislabelDetector,
    NearDuplicateDetector,
    OODInsertionDetector,
    SpectralSignatureDetector,
)
from dual_encoder import extract_dual_embeddings, scan_collusion  # noqa: E402
from schema import M1Result, SampleAnomaly  # noqa: E402
from sybil_clustering import cluster_sybils  # noqa: E402


def default_detectors() -> List[Any]:
    """Sample-level detectors specified by Clause 2.2.1."""
    return [
        AnnotationAnomalyDetector(),
        NearDuplicateDetector(),
        LabelFlipDetector(),
        OODInsertionDetector(),
        SpectralSignatureDetector(),
        ActivationClusteringDetector(),
        MislabelDetector(),
    ]


def run_data_integrity(
    records: Sequence[Any],
    capability_matrix: Any,
    *,
    embeddings: Optional[Mapping[str, Any]] = None,
    encoder_binding: Any = None,
    reference_profile: Optional[Mapping[str, Any]] = None,
    pipeline_baseline_rate: float = 0.05,
    collusion_threshold: float = 2.5,
    detectors: Optional[Sequence[Any]] = None,
) -> M1Result:
    """Run gated M1 detectors and emit the fusion-ready ``M1Result``.

    Detectors that the capability matrix cannot support return
    ``UNAVAILABLE`` and contribute no findings. Dual-encoder collusion
    runs only when both E_ref and E_sub embeddings are present.
    """
    records_list = list(records)
    resolved_embeddings = _resolve_embeddings(records_list, embeddings, encoder_binding)
    detector_kwargs = {
        "embeddings": resolved_embeddings,
        "reference_profile": reference_profile or {},
    }

    anomalies: List[SampleAnomaly] = []
    for detector in detectors or default_detectors():
        payload = detector.run(records_list, capability_matrix, **detector_kwargs)
        anomalies.extend(list(payload.get("anomalies") or []))

    if _has_subject_embeddings(resolved_embeddings):
        anomalies.extend(
            scan_collusion(resolved_embeddings, records_list, threshold=collusion_threshold)
        )

    sybil_groups = cluster_sybils(records_list, anomalies)
    sybil_map = {
        contributor_id: f"sybil-{index}"
        for index, group in enumerate(sybil_groups)
        for contributor_id in group
        if len(group) >= 2
    }
    rollups = rollup_contributors(
        records_list,
        anomalies,
        pipeline_baseline_rate=pipeline_baseline_rate,
        sybil_cluster_by_contributor=sybil_map,
    )
    rollups = attach_sybil_ids(rollups, sybil_groups)

    return M1Result(
        capability_matrix=capability_matrix,
        sample_anomalies=anomalies,
        contributor_rollups=rollups,
        sybil_suspect_groups=sybil_groups,
    )


def _resolve_embeddings(
    records: Sequence[Any],
    embeddings: Optional[Mapping[str, Any]],
    encoder_binding: Any,
) -> Dict[str, Any]:
    if embeddings:
        return dict(embeddings)
    if encoder_binding is None:
        return {}
    return extract_dual_embeddings(list(records), encoder_binding)


def _has_subject_embeddings(embeddings: Mapping[str, Any]) -> bool:
    for row in embeddings.values():
        if isinstance(row, Mapping) and row.get("sub_emb") is not None:
            return True
    return False


__all__ = ["default_detectors", "run_data_integrity"]
