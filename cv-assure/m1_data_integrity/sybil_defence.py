"""Sybil-identity clustering for Module M1 (Clause 2.2.1)."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import ContributorRollup, FindingType, SampleAnomaly  # noqa: E402

AssetRecord = Any

SIMILARITY_THRESHOLD = 0.85
EPS = 1e-12
FINDING_TYPES = tuple(item.value for item in FindingType)
METADATA_KEYS = (
    "camera",
    "make",
    "model",
    "device",
    "device_id",
    "sensor",
    "acquisition_id",
    "source",
)


def detect_sybil_contributors(
    records: list[AssetRecord],
    anomalies: list[SampleAnomaly],
    rollups: list[ContributorRollup],
) -> list[list[str]]:
    """Cluster contributors whose class, flag, and acquisition signatures match.

    Pairwise similarity is the mean of:
    - cosine similarity of class-distribution vectors
    - cosine similarity of finding-type flag signatures
    - Jaccard similarity of resolution / camera metadata tokens
    - cosine similarity of mean (width, height) acquisition vectors

    Contributors with similarity ``> 0.85`` are grouped into connected components.
    Matching ``ContributorRollup`` objects are updated in place with a shared
    ``sybil_cluster_id``. If any member is already ``quarantine``, every member
    of the cluster is elevated to ``quarantine``.
    """
    signatures = _build_signatures(records, anomalies, rollups)
    contributor_ids = sorted(signatures)
    if len(contributor_ids) < 2:
        return []

    graph: Dict[str, Set[str]] = {cid: set() for cid in contributor_ids}
    for i, left in enumerate(contributor_ids):
        for right in contributor_ids[i + 1 :]:
            if _pairwise_similarity(signatures[left], signatures[right]) > SIMILARITY_THRESHOLD:
                graph[left].add(right)
                graph[right].add(left)

    clusters = _connected_components(graph)
    _apply_cluster_updates(rollups, clusters)
    return clusters


def _build_signatures(
    records: Sequence[Any],
    anomalies: Sequence[SampleAnomaly],
    rollups: Sequence[ContributorRollup],
) -> Dict[str, Dict[str, Any]]:
    records_by_contributor: Dict[str, List[Any]] = defaultdict(list)
    for record in records:
        records_by_contributor[_contributor_id(record)].append(record)

    flags_by_contributor: Dict[str, Counter] = defaultdict(Counter)
    for anomaly in anomalies:
        flags_by_contributor[anomaly.contributor_id][anomaly.finding_type.value] += 1

    contributor_ids = (
        set(records_by_contributor)
        | set(flags_by_contributor)
        | {rollup.contributor_id for rollup in rollups}
    )
    class_names = sorted(
        {
            _primary_label(record)
            for group in records_by_contributor.values()
            for record in group
        }
    )

    signatures: Dict[str, Dict[str, Any]] = {}
    for contributor_id in contributor_ids:
        group = records_by_contributor.get(contributor_id, [])
        class_counts = Counter(_primary_label(record) for record in group)
        signatures[contributor_id] = {
            "class_vec": _normalized_vector(class_names, class_counts),
            "flag_vec": _normalized_vector(FINDING_TYPES, flags_by_contributor.get(contributor_id, Counter())),
            "meta_tokens": _metadata_tokens(group),
            "size_vec": _mean_size_vector(group),
        }
    return signatures


def _pairwise_similarity(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    parts = [
        _cosine(left["class_vec"], right["class_vec"]),
        _cosine(left["flag_vec"], right["flag_vec"]),
        _jaccard(left["meta_tokens"], right["meta_tokens"]),
        _cosine(left["size_vec"], right["size_vec"]),
    ]
    usable = [value for value in parts if value is not None]
    if not usable:
        return 0.0
    return float(sum(usable) / len(usable))


def _apply_cluster_updates(rollups: Sequence[ContributorRollup], clusters: Sequence[Sequence[str]]) -> None:
    cluster_of: Dict[str, str] = {}
    members_of: Dict[str, List[str]] = {}
    for index, group in enumerate(clusters):
        cluster_id = f"sybil-{index}"
        members_of[cluster_id] = list(group)
        for contributor_id in group:
            cluster_of[contributor_id] = cluster_id

    by_id = {rollup.contributor_id: rollup for rollup in rollups}
    quarantined_clusters: Set[str] = set()
    for cluster_id, members in members_of.items():
        if any(by_id[cid].disposition == "quarantine" for cid in members if cid in by_id):
            quarantined_clusters.add(cluster_id)

    for rollup in rollups:
        cluster_id = cluster_of.get(rollup.contributor_id)
        if not cluster_id:
            continue
        rollup.sybil_cluster_id = cluster_id
        if cluster_id in quarantined_clusters and rollup.disposition != "quarantine":
            rollup.disposition = "quarantine"


def _connected_components(graph: Mapping[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    clusters: List[List[str]] = []
    for node in sorted(graph):
        if node in seen:
            continue
        stack = [node]
        component: List[str] = []
        seen.add(node)
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbour in graph[current]:
                if neighbour in seen:
                    continue
                seen.add(neighbour)
                stack.append(neighbour)
        if len(component) >= 2:
            clusters.append(sorted(component))
    return clusters


def _normalized_vector(keys: Sequence[str], counts: Mapping[str, int]) -> np.ndarray:
    raw = np.array([float(counts.get(key, 0)) for key in keys], dtype=np.float64)
    norm = float(np.linalg.norm(raw))
    if norm < EPS:
        return raw
    return raw / norm


def _metadata_tokens(records: Sequence[Any]) -> Set[str]:
    tokens: Set[str] = set()
    for record in records:
        meta = _image_metadata(record)
        width, height = _resolution(meta)
        if width is not None and height is not None:
            tokens.add(f"res:{int(width)}x{int(height)}")
        for key in METADATA_KEYS:
            value = meta.get(key)
            if value:
                tokens.add(f"{key}:{str(value).strip().lower()}")
    return tokens


def _mean_size_vector(records: Sequence[Any]) -> np.ndarray:
    widths: List[float] = []
    heights: List[float] = []
    for record in records:
        width, height = _resolution(_image_metadata(record))
        if width is None or height is None:
            continue
        widths.append(width)
        heights.append(height)
    if not widths:
        return np.zeros(2, dtype=np.float64)
    return np.array([float(np.mean(widths)), float(np.mean(heights))], dtype=np.float64)


def _image_metadata(record: Any) -> Dict[str, Any]:
    meta = _record_get(record, "image_metadata") or _record_get(record, "metadata") or {}
    if isinstance(meta, Mapping):
        return dict(meta)
    return {
        "width": getattr(meta, "width", None),
        "height": getattr(meta, "height", None),
        "camera": getattr(meta, "camera", None),
        "make": getattr(meta, "make", None),
        "model": getattr(meta, "model", None),
        "device": getattr(meta, "device", None),
        "device_id": getattr(meta, "device_id", None),
        "sensor": getattr(meta, "sensor", None),
        "acquisition_id": getattr(meta, "acquisition_id", None),
        "source": getattr(meta, "source", None),
    }


def _resolution(meta: Mapping[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    try:
        width = meta.get("width")
        height = meta.get("height")
        return (
            float(width) if width is not None else None,
            float(height) if height is not None else None,
        )
    except (TypeError, ValueError):
        return None, None


def _cosine(left: np.ndarray, right: np.ndarray) -> Optional[float]:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom < EPS:
        return None
    return float(np.clip(np.dot(left, right) / denom, 0.0, 1.0))


def _jaccard(left: Set[str], right: Set[str]) -> Optional[float]:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return float(len(left & right) / len(union))


def _record_get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _contributor_id(record: Any) -> str:
    return str(_record_get(record, "contributor_id") or "unknown")


def _primary_label(record: Any) -> str:
    anns = _record_get(record, "annotations") or []
    if anns:
        first = anns[0]
        if isinstance(first, Mapping):
            name = first.get("class_name") or first.get("label")
        else:
            name = getattr(first, "class_name", None) or getattr(first, "label", None)
        if name:
            return str(name)
    return str(_record_get(record, "class_name") or _record_get(record, "label") or "unknown")


__all__ = ["SIMILARITY_THRESHOLD", "detect_sybil_contributors"]
