"""Contributor-level binomial rollup (CV-ASSURE v2, Clause 2.2.1)."""

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

# M0 AssetRecord is duck-typed (asset_id, contributor_id, annotations / class_name).
AssetRecord = Any

EPS = 1e-12


def perform_contributor_rollup(
    records: list[AssetRecord],
    anomalies: list[SampleAnomaly],
) -> list[ContributorRollup]:
    """Group sample findings by contributor and test against the pipeline baseline.

    Baseline rate is the pipeline flag rate: unique flagged assets / total assets.
    Each contributor is tested with a one-sided exact binomial test
    (``alternative='greater'``). Poison rate is the contributor flag rate with a
    95% Clopper-Pearson interval, reported in the Clause 2.2.1 reason string.
    """
    records_by_contributor: Dict[str, List[Any]] = defaultdict(list)
    record_by_asset: Dict[str, Any] = {}
    for record in records:
        contributor_id = _contributor_id(record)
        records_by_contributor[contributor_id].append(record)
        record_by_asset[_asset_id(record)] = record

    flagged_by_contributor: Dict[str, set[str]] = defaultdict(set)
    classes_by_contributor: Dict[str, Counter] = defaultdict(Counter)
    for anomaly in anomalies:
        flagged_by_contributor[anomaly.contributor_id].add(anomaly.asset_id)
        class_name = _class_from_anomaly(anomaly, record_by_asset.get(anomaly.asset_id))
        if class_name:
            classes_by_contributor[anomaly.contributor_id][class_name] += 1

    total_samples_pipeline = len(records)
    total_flagged_pipeline = len({anomaly.asset_id for anomaly in anomalies})
    baseline_rate = (
        0.0
        if total_samples_pipeline == 0
        else total_flagged_pipeline / total_samples_pipeline
    )

    contributor_ids = sorted(set(records_by_contributor) | set(flagged_by_contributor))
    rollups: List[ContributorRollup] = []
    for contributor_id in contributor_ids:
        contributor_records = records_by_contributor.get(contributor_id, [])
        flagged_ids = flagged_by_contributor.get(contributor_id, set())
        known_ids = {_asset_id(record) for record in contributor_records}
        flagged_in_records = flagged_ids & known_ids if known_ids else flagged_ids
        total_samples = len(contributor_records)
        flagged_samples = len(flagged_in_records)
        if total_samples == 0:
            total_samples = flagged_samples
        if flagged_samples > total_samples:
            flagged_samples = total_samples

        flag_rate = 0.0 if total_samples == 0 else flagged_samples / total_samples
        p_value = _binomial_p(flagged_samples, total_samples, baseline_rate)
        poison_rate, poison_half_width = _poison_rate_with_ci(flagged_samples, total_samples)
        concentrated = _concentrated_classes(classes_by_contributor.get(contributor_id, Counter()))
        disposition = _assign_disposition(p_value, flag_rate, baseline_rate)
        reason = _format_reason(
            contributor_id=contributor_id,
            flag_rate=flag_rate,
            baseline_rate=baseline_rate,
            p_value=p_value,
            concentrated_classes=concentrated,
            poison_rate=poison_rate,
            poison_half_width=poison_half_width,
            disposition=disposition,
        )
        rollups.append(
            ContributorRollup(
                contributor_id=contributor_id,
                total_samples=total_samples,
                flagged_samples=flagged_samples,
                flag_rate=flag_rate,
                pipeline_baseline_rate=min(1.0, max(0.0, baseline_rate)),
                p_value=p_value,
                estimated_poison_rate=poison_rate,
                concentrated_classes=concentrated,
                disposition=disposition,
                reason=reason,
            )
        )
    return rollups


def _binomial_p(flagged_samples: int, total_samples: int, baseline_rate: float) -> float:
    if total_samples <= 0:
        return 1.0
    p = min(1.0, max(0.0, baseline_rate))
    k = min(flagged_samples, total_samples)
    return float(binomtest(k, total_samples, p=p, alternative="greater").pvalue)


def _poison_rate_with_ci(flagged_samples: int, total_samples: int) -> tuple[float, float]:
    """Point estimate flagged/total and half-width of a 95% Clopper-Pearson interval."""
    if total_samples <= 0:
        return 0.0, 0.0
    k = min(flagged_samples, total_samples)
    point = k / total_samples
    interval = binomtest(k, total_samples).proportion_ci(confidence_level=0.95)
    half_width = 0.5 * (float(interval.high) - float(interval.low))
    return point, half_width


def _assign_disposition(p_value: float, flag_rate: float, baseline_rate: float) -> Disposition:
    if p_value < 0.01 and flag_rate > 3.0 * baseline_rate:
        return "quarantine"
    if p_value < 0.05:
        return "review"
    return "accept"


def _format_reason(
    *,
    contributor_id: str,
    flag_rate: float,
    baseline_rate: float,
    p_value: float,
    concentrated_classes: Sequence[str],
    poison_rate: float,
    poison_half_width: float,
    disposition: Disposition,
) -> str:
    class_text = concentrated_classes[0] if concentrated_classes else "unknown"
    return (
        f"Contributor {contributor_id}: {flag_rate * 100:.1f}% flag rate against "
        f"{baseline_rate * 100:.1f}% baseline ({_format_p(p_value)}), "
        f"concentrated in class {class_text}, "
        f"estimated poison rate {_format_percent(poison_rate)}±{_format_percent(poison_half_width)}%. "
        f"Disposition: {disposition}."
    )


def _format_p(p_value: float) -> str:
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def _format_percent(rate: float) -> str:
    return f"{rate * 100:.0f}"


def _concentrated_classes(flag_classes: Counter) -> List[str]:
    if not flag_classes:
        return []
    top_count = flag_classes.most_common(1)[0][1]
    return [name for name, count in flag_classes.most_common() if count == top_count]


def _class_from_anomaly(anomaly: SampleAnomaly, record: Any) -> Optional[str]:
    details = anomaly.details or {}
    for key in ("class_name", "assigned_label", "label"):
        value = details.get(key)
        if value:
            return str(value)
    if record is not None:
        return _primary_label(record)
    return None


def _record_get(record: Any, name: str, default: Any = None) -> Any:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _contributor_id(record: Any) -> str:
    return str(_record_get(record, "contributor_id") or "unknown")


def _asset_id(record: Any) -> str:
    return str(_record_get(record, "asset_id") or "unknown")


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


__all__ = ["AssetRecord", "perform_contributor_rollup"]
