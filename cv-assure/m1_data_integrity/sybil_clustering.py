"""Fake-identity (Sybil) clustering over contributors (Clause 2.2.1)."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Set

import numpy as np

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import FindingType, SampleAnomaly  # noqa: E402

EPS = 1e-8
DEFAULT_DUP_EDGES = 2
DEFAULT_HIST_COSINE = 0.92
DEFAULT_MIN_FLAGS = 3


def cluster_sybils(
    records: Sequence[Any],
    anomalies: Sequence[SampleAnomaly],
    *,
    min_cross_duplicate_edges: int = DEFAULT_DUP_EDGES,
    histogram_cosine: float = DEFAULT_HIST_COSINE,
    min_flags_for_histogram: int = DEFAULT_MIN_FLAGS,
) -> List[List[str]]:
    """Group contributor ids that look like split identities.

    Edges come from two independent signals:
    1. Cross-contributor near-duplicate floods (shared or copied images).
    2. Highly similar finding-type histograms among contributors with enough flags.
    """
    contributors = sorted({_contributor_id(record) for record in records} | {item.contributor_id for item in anomalies})
    if len(contributors) < 2:
        return []

    graph: Dict[str, Set[str]] = {cid: set() for cid in contributors}
    _add_duplicate_edges(graph, anomalies, min_cross_duplicate_edges)
    _add_histogram_edges(graph, anomalies, histogram_cosine, min_flags_for_histogram)

    groups: List[List[str]] = []
    seen: Set[str] = set()
    for node in contributors:
        if node in seen:
            continue
        component = _bfs(graph, node)
        seen.update(component)
        if len(component) >= 2:
            groups.append(sorted(component))
    return groups


def _add_duplicate_edges(
    graph: Dict[str, Set[str]],
    anomalies: Sequence[SampleAnomaly],
    min_edges: int,
) -> None:
    pair_counts: Counter = Counter()
    for anomaly in anomalies:
        if anomaly.finding_type is not FindingType.NEAR_DUPLICATE_FLOOD:
            continue
        matches = anomaly.details.get("matches") or []
        for match in matches:
            if not isinstance(match, Mapping):
                continue
            other = str(match.get("other_contributor_id") or "")
            if not other or other == anomaly.contributor_id:
                continue
            if match.get("cross_contributor") is False:
                continue
            pair = tuple(sorted((anomaly.contributor_id, other)))
            pair_counts[pair] += 1
    for (a, b), count in pair_counts.items():
        if count < min_edges:
            continue
        if a in graph and b in graph:
            graph[a].add(b)
            graph[b].add(a)


def _add_histogram_edges(
    graph: Dict[str, Set[str]],
    anomalies: Sequence[SampleAnomaly],
    cosine_min: float,
    min_flags: int,
) -> None:
    histograms: Dict[str, Counter] = defaultdict(Counter)
    for anomaly in anomalies:
        histograms[anomaly.contributor_id][anomaly.finding_type.value] += 1
    eligible = [cid for cid, hist in histograms.items() if sum(hist.values()) >= min_flags]
    types = sorted({ftype.value for ftype in FindingType})
    vectors = {cid: np.array([histograms[cid][t] for t in types], dtype=np.float64) for cid in eligible}
    for i, a in enumerate(eligible):
        for b in eligible[i + 1 :]:
            if _cosine(vectors[a], vectors[b]) >= cosine_min:
                graph[a].add(b)
                graph[b].add(a)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def _bfs(graph: Mapping[str, Set[str]], start: str) -> List[str]:
    queued = [start]
    seen = {start}
    for node in queued:
        for neighbour in graph.get(node, ()):
            if neighbour in seen:
                continue
            seen.add(neighbour)
            queued.append(neighbour)
    return queued


def _contributor_id(record: Any) -> str:
    if isinstance(record, Mapping):
        return str(record.get("contributor_id") or "unknown")
    return str(getattr(record, "contributor_id", None) or "unknown")


__all__ = ["cluster_sybils"]
