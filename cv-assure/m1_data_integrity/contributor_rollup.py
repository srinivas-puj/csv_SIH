"""Contributor-level binomial rollup of sample findings (Clause 2.2.1)."""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from scipy.stats import binomtest

_MODULE_ROOT = Path(__file__).resolve().parent
_CV_ASSURE = Path(__file__).resolve().parent.parent
for _path in (str(_CV_ASSURE), str(_MODULE_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from schema import ContributorRollup, Disposition, SampleAnomaly  # noqa: E402

EPS = 1e-8
DEFAULT_BASELINE = 0.05
QUARANTINE_ALPHA = 0.01
REVIEW_ALPHA = 0.05
MIN_QUARANTINE_RATE = 0.10
MIN_REVIEW_RATE = 0.05


def rollup_contributors(
    records: Sequence[Any],
    anomalies: Sequence[SampleAnomaly],
    pipeline_baseline_rate: float = DEFAULT_BASELINE,
    *,
    quarantine_alpha: float = QUARANTINE_ALPHA,
    review_alpha: float = REVIEW_ALPHA,
    min_quarantine_rate: float = MIN_QUARANTINE_RATE,
    min_review_rate: float = MIN_REVIEW_RATE,
    sybil_cluster_by_contributor: Optional[Mapping[str, str]] = None,
) -> List[ContributorRollup]:
    """Aggregate sample findings into per-contributor binomial tests.

    A sample is flagged if it appears in ``anomalies`` at least once. The
    one-sided exact binomial test uses ``pipeline_baseline_rate`` as H0
    success probability (alternative: greater).
    """
    baseline = min(1.0, max(EPS, float(pipeline_baseline_rate)))
    totals, class_counts = _contributor_totals(records)
    flagged_ids, flag_classes = _flag_index(anomalies)
    sybil_map = dict(sybil_cluster_by_contributor or {})

    contributor_ids = sorted(set(totals) | set(flagged_ids) | set(sybil_map))
    rollups: List[ContributorRollup] = []
    for contributor_id in contributor_ids:
        total = int(totals.get(contributor_id, 0))
        flagged_set = flagged_ids.get(contributor_id, set())
        flagged = len(flagged_set)
        flag_rate = 0.0 if total == 0 else flagged / total
        p_value = _binomial_p(flagged, total, baseline)
        poison = 0.0 if total == 0 else max(0.0, (flag_rate - baseline) / max(EPS, 1.0 - baseline))
        poison = min(1.0, poison)
        concentrated = _concentrated_classes(
            flag_classes.get(contributor_id, Counter()),
            class_counts.get(contributor_id, Counter()),
        )
        disposition = _disposition(
            total,
            flag_rate,
            p_value,
            quarantine_alpha=quarantine_alpha,
            review_alpha=review_alpha,
            min_quarantine_rate=min_quarantine_rate,
            min_review_rate=min_review_rate,
        )
        rollups.append(
            ContributorRollup(
                contributor_id=contributor_id,
                total_samples=total,
                flagged_samples=flagged,
                flag_rate=flag_rate,
                pipeline_baseline_rate=baseline,
                p_value=p_value,
                estimated_poison_rate=poison,
                concentrated_classes=concentrated,
                disposition=disposition,
                sybil_cluster_id=sybil_map.get(contributor_id),
            )
        )
    return rollups


def attach_sybil_ids(
    rollups: Sequence[ContributorRollup],
    sybil_groups: Sequence[Sequence[str]],
) -> List[ContributorRollup]:
    """Return copies of ``rollups`` with ``sybil_cluster_id`` filled from groups."""
    mapping: Dict[str, str] = {}
    for index, group in enumerate(sybil_groups):
        if len(group) < 2:
            continue
        cluster_id = f"sybil-{index}"
        for contributor_id in group:
            mapping[str(contributor_id)] = cluster_id
    updated: List[ContributorRollup] = []
    for rollup in rollups:
        payload = rollup.to_dict()
        payload["sybil_cluster_id"] = mapping.get(rollup.contributor_id, rollup.sybil_cluster_id)
        updated.append(ContributorRollup.from_dict(payload))
    return updated


def _contributor_totals(records: Sequence[Any]) -> tuple[Dict[str, int], Dict[str, Counter]]:
    totals: Dict[str, int] = defaultdict(int)
    classes: Dict[str, Counter] = defaultdict(Counter)
    for record in records:
        contributor_id = _contributor_id(record)
        totals[contributor_id] += 1
        classes[contributor_id][_primary_label(record)] += 1
    return dict(totals), dict(classes)


def _flag_index(anomalies: Sequence[SampleAnomaly]) -> tuple[Dict[str, set], Dict[str, Counter]]:
    flagged: Dict[str, set] = defaultdict(set)
    classes: Dict[str, Counter] = defaultdict(Counter)
    for anomaly in anomalies:
        flagged[anomaly.contributor_id].add(anomaly.asset_id)
        class_name = _class_from_details(anomaly.details)
        if class_name:
            classes[anomaly.contributor_id][class_name] += 1
    return dict(flagged), dict(classes)


def _class_from_details(details: Mapping[str, Any]) -> Optional[str]:
    for key in ("class_name", "assigned_label", "label"):
        value = details.get(key)
        if value:
            return str(value)
    return None


def _binomial_p(flagged: int, total: int, baseline: float) -> float:
    if total <= 0:
        return 1.0
    k = min(flagged, total)
    return float(binomtest(k, total, p=baseline, alternative="greater").pvalue)


def _concentrated_classes(flag_classes: Counter, all_classes: Counter) -> List[str]:
    total_flags = sum(flag_classes.values())
    total_samples = sum(all_classes.values())
    if total_flags == 0:
        return []
    names: List[str] = []
    for class_name, flag_n in flag_classes.most_common():
        flag_share = flag_n / total_flags
        sample_share = (all_classes.get(class_name, 0) / total_samples) if total_samples else 0.0
        if flag_share >= 0.5 or flag_share >= max(0.25, 2.0 * sample_share):
            names.append(class_name)
    return names


def _disposition(
    total: int,
    flag_rate: float,
    p_value: float,
    *,
    quarantine_alpha: float,
    review_alpha: float,
    min_quarantine_rate: float,
    min_review_rate: float,
) -> Disposition:
    if total <= 0:
        return "accept"
    if p_value <= quarantine_alpha and flag_rate >= min_quarantine_rate:
        return "quarantine"
    if p_value <= review_alpha or flag_rate >= min_review_rate:
        return "review"
    return "accept"


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


__all__ = ["attach_sybil_ids", "rollup_contributors"]
