# Copyright 2020 The Feast Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Offline/online feature consistency validation.

This module verifies that, for the *same entity at the same point in event time*,
the feature value returned by the offline point-in-time join
(``FeatureStore.get_historical_features``) equals the value served online
(``FeatureStore.get_online_features``) within an explicit tolerance.

Design constraints (see ``docs/reference/feature-consistency.md``):

* Existing offline/online interfaces and feature computation logic are not
  changed. Validation is opened through a new entry point only
  (``FeatureStore.validate_feature_consistency``).
* Event time and processing time are never mixed: the entity dataframe's
  ``event_timestamp`` column is the sole time semantic used for the offline
  point-in-time join and for TTL evaluation. Wall-clock ("now") processing
  time is only read where Feast itself already uses it, namely the online
  store's ``OUTSIDE_MAX_AGE`` status.
* Float comparisons use an explicit tolerance with a documented basis
  (numpy ``isclose`` semantics: ``|a - b| <= atol + rtol * |b|``); every
  other type uses exact equality.
* Validation works on batches, produces mergeable reports for million-row
  sharded runs, and never reports a zero mismatch rate from an empty dataset.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

from feast.errors import (
    ConsistencyTimestampColumnMissingError,
    EmptyConsistencyDatasetError,
)

if TYPE_CHECKING:
    from feast.feature_store import FeatureStore
    from feast.feature_view import FeatureView

#: Name of the single accepted event-time column on validation inputs.
EVENT_TIMESTAMP_COLUMN = "event_timestamp"

#: Internal correlation column used to realign offline join output rows to
#: the original input ordering. Collides only if a caller already has a
#: column of this exact, deliberately namespaced name.
ROW_ID_COLUMN = "__feast_consistency_row_id"

#: Field status names, indexed by the numeric value of
#: ``feast.serving.FieldStatus``. Kept as strings so this module does not
#: depend on the generated proto at import time.
_FIELD_STATUS_NAMES = {
    0: "INVALID",
    1: "PRESENT",
    2: "NULL_VALUE",
    3: "NOT_FOUND",
    4: "OUTSIDE_MAX_AGE",
}


class MismatchReason(str, enum.Enum):
    """Why a single (entity row, feature) cell was judged inconsistent.

    The classifier assigns the first matching reason in this priority order:
    out-of-retention, online unavailable, value missing, window mismatch,
    batch/stream semantic difference, type conversion difference, value
    mismatch. Every non-matching cell carries ``VALUE_MATCH``.
    """

    VALUE_MATCH = "value_match"
    OUT_OF_RETENTION = "out_of_retention"
    ONLINE_UNAVAILABLE = "online_unavailable"
    VALUE_MISSING = "value_missing"
    WINDOW_MISMATCH = "window_mismatch"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    TYPE_MISMATCH = "type_mismatch"
    VALUE_MISMATCH = "value_mismatch"


# Reasons that count as inconsistent in the numerator of the mismatch rate.
_MISMATCH_REASONS = {r for r in MismatchReason if r is not MismatchReason.VALUE_MATCH}


@dataclass(frozen=True)
class ConsistencyValidationConfig:
    """Configuration for an offline/online consistency validation run.

    Args:
        float_rtol: Relative tolerance for float comparison, following
            ``numpy.isclose`` semantics. Rationale: offline engines may run
            aggregations in double precision while online stores persist
            float32, so a relative component absorbs scale-dependent rounding.
        float_atol: Absolute tolerance for float comparison, following
            ``numpy.isclose`` semantics. Rationale: guarantees a bounded error
            for values near zero, where a purely relative tolerance would
            either be meaningless or require an infinitely tight bound.
        max_sample_mismatches: Maximum number of mismatching cells retained
            in the report's ``sample_mismatches`` list.
        check_out_of_retention: When True, cells whose validation event time
            falls outside the feature view's TTL window are reported as
            ``out_of_retention`` instead of being silently dropped.
        nan_matches_nan: When True, two NaN values are considered equal
            (they represent the same "missing measurement" state); when
            False, NaN-vs-NaN is a value mismatch.
        prepare_online_store: When True, materialize each involved feature
            view up to the maximum validation event time before reading, so
            the online store contains the same rows the offline join sees.
            When False (default), validation never mutates the online store;
            the caller is responsible for having materialized the data.
        online_read_batch_size: Number of entity rows per
            ``get_online_features`` call, bounding request size on large
            validation datasets.
    """

    float_rtol: float = 1e-5
    float_atol: float = 1e-8
    max_sample_mismatches: int = 100
    check_out_of_retention: bool = True
    nan_matches_nan: bool = True
    prepare_online_store: bool = False
    online_read_batch_size: int = 1000

    def __post_init__(self) -> None:
        if self.float_rtol < 0 or self.float_atol < 0:
            raise ValueError("float_rtol and float_atol must be non-negative.")
        if self.max_sample_mismatches < 0:
            raise ValueError("max_sample_mismatches must be non-negative.")
        if self.online_read_batch_size < 1:
            raise ValueError("online_read_batch_size must be >= 1.")


@dataclass
class ConsistencyMismatch:
    """One inconsistent (entity row, feature) cell with evidence."""

    row_index: int
    feature_ref: str
    entity_values: Dict[str, Any]
    event_timestamp: str
    reason: str
    offline_value: Any
    online_value: Any
    offline_type: str
    online_type: str
    online_status: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_index": self.row_index,
            "feature_ref": self.feature_ref,
            "entity_values": self.entity_values,
            "event_timestamp": self.event_timestamp,
            "reason": self.reason,
            "offline_value": _json_safe(self.offline_value),
            "online_value": _json_safe(self.online_value),
            "offline_type": self.offline_type,
            "online_type": self.online_type,
            "online_status": self.online_status,
            "detail": self.detail,
        }


@dataclass
class ConsistencyReport:
    """Result of validating one shard (or one merged set of shards).

    The mismatch rate is defined as::

        mismatch_rate = total_mismatch_cells / total_compared_cells

    where the denominator counts every requested feature cell in every input
    row, including cells that were unavailable online or out of retention.
    Those cells are mismatches, never silent passes, so the numerator and
    denominator are directly recomputable from ``per_feature_stats``.
    """

    total_rows: int = 0
    total_compared_cells: int = 0
    total_mismatch_cells: int = 0
    reason_counts: Dict[str, int] = field(default_factory=dict)
    per_feature_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sample_mismatches: List[ConsistencyMismatch] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    shard_count: int = 1

    @property
    def mismatch_rate(self) -> float:
        if self.total_compared_cells == 0:
            raise ZeroDivisionError(
                "No cells were compared. An empty validation dataset raises "
                "EmptyConsistencyDatasetError rather than reporting a rate."
            )
        return self.total_mismatch_cells / self.total_compared_cells

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_rows": self.total_rows,
            "total_compared_cells": self.total_compared_cells,
            "total_mismatch_cells": self.total_mismatch_cells,
            "mismatch_rate": (
                self.total_mismatch_cells / self.total_compared_cells
                if self.total_compared_cells
                else None
            ),
            "reason_counts": dict(self.reason_counts),
            "per_feature_stats": {
                k: dict(v) for k, v in self.per_feature_stats.items()
            },
            "sample_mismatches": [m.to_dict() for m in self.sample_mismatches],
            "config": dict(self.config),
            "shard_count": self.shard_count,
        }

    def to_json(self, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ConsistencyReport":
        report = cls(
            total_rows=payload.get("total_rows", 0),
            total_compared_cells=payload.get("total_compared_cells", 0),
            total_mismatch_cells=payload.get("total_mismatch_cells", 0),
            reason_counts=dict(payload.get("reason_counts", {})),
            per_feature_stats={
                k: dict(v) for k, v in payload.get("per_feature_stats", {}).items()
            },
            sample_mismatches=[
                ConsistencyMismatch(**m) for m in payload.get("sample_mismatches", [])
            ],
            config=dict(payload.get("config", {})),
            shard_count=payload.get("shard_count", 1),
        )
        return report

    def merge(self, other: "ConsistencyReport") -> "ConsistencyReport":
        """Merge two shard reports so multi-machine runs produce one result.

        Counts and per-feature statistics add up; mismatch samples are
        concatenated and trimmed to the smaller configured
        ``max_sample_mismatches`` limit. Merging is associative on every
        aggregate field, so shards can be reduced in any order.
        """
        merged = ConsistencyReport(
            total_rows=self.total_rows + other.total_rows,
            total_compared_cells=(
                self.total_compared_cells + other.total_compared_cells
            ),
            total_mismatch_cells=(
                self.total_mismatch_cells + other.total_mismatch_cells
            ),
            reason_counts=dict(self.reason_counts),
            per_feature_stats={k: dict(v) for k, v in self.per_feature_stats.items()},
            sample_mismatches=list(self.sample_mismatches),
            config=dict(self.config) or dict(other.config),
            shard_count=self.shard_count + other.shard_count,
        )
        for reason, count in other.reason_counts.items():
            merged.reason_counts[reason] = merged.reason_counts.get(reason, 0) + count
        for feature_ref, stats in other.per_feature_stats.items():
            if feature_ref not in merged.per_feature_stats:
                merged.per_feature_stats[feature_ref] = {
                    "compared_cells": 0,
                    "mismatch_cells": 0,
                    "reason_counts": {},
                }
            target = merged.per_feature_stats[feature_ref]
            target["compared_cells"] += stats.get("compared_cells", 0)
            target["mismatch_cells"] += stats.get("mismatch_cells", 0)
            for reason, count in stats.get("reason_counts", {}).items():
                target["reason_counts"][reason] = (
                    target["reason_counts"].get(reason, 0) + count
                )
        merged.sample_mismatches.extend(other.sample_mismatches)
        limit = min(
            self.config.get("max_sample_mismatches", 100),
            other.config.get("max_sample_mismatches", 100),
        )
        merged.sample_mismatches = merged.sample_mismatches[:limit]
        for stats in merged.per_feature_stats.values():
            compared = stats["compared_cells"]
            stats["mismatch_rate"] = (
                stats["mismatch_cells"] / compared if compared else None
            )
        return merged


def validate_feature_consistency(
    store: "FeatureStore",
    entity_df: pd.DataFrame,
    features: Sequence[str],
    config: Optional[ConsistencyValidationConfig] = None,
    full_feature_names: bool = False,
) -> ConsistencyReport:
    """Compare offline point-in-time values against online-served values.

    Args:
        store: A connected ``FeatureStore``.
        entity_df: Entity rows to validate. Must contain every join key used
            by the requested feature views and an ``event_timestamp`` column.
            The timestamps are *event times*: they drive the offline
            point-in-time join and TTL evaluation, never processing time.
        features: Feature references in ``"feature_view:feature"`` form, or
            a list derived from a ``FeatureService``.
        config: Validation tolerances and limits. Defaults to
            ``ConsistencyValidationConfig()``.
        full_feature_names: If True, report statistics keyed by
            ``feature_view__feature``; otherwise keyed by the bare feature
            name. Must match the naming used in ``features`` resolution.

    Returns:
        A :class:`ConsistencyReport` with aggregate and per-feature mismatch
        rates, reason breakdowns, and evidence samples.

    Raises:
        EmptyConsistencyDatasetError: The input dataframe has no rows.
        ConsistencyTimestampColumnMissingError: The input has no
            ``event_timestamp`` column.
    """
    config = config or ConsistencyValidationConfig()
    if entity_df is None or len(entity_df) == 0:
        raise EmptyConsistencyDatasetError()
    if EVENT_TIMESTAMP_COLUMN not in entity_df.columns:
        raise ConsistencyTimestampColumnMissingError(list(entity_df.columns))

    feature_refs = _resolve_feature_refs(store, features)
    feature_views = _load_feature_views(store, feature_refs)
    join_keys = _collect_join_keys(feature_views)
    _validate_entity_columns(entity_df, join_keys)
    if ROW_ID_COLUMN in entity_df.columns:
        raise ValueError(
            f"Entity dataframe must not contain a reserved column name "
            f"'{ROW_ID_COLUMN}', which is used internally for row alignment."
        )

    if config.prepare_online_store:
        _materialize_for_validation(store, feature_views, entity_df)

    # Offline side: one point-in-time join over the whole shard. A unique
    # correlation column is attached because the offline store does not
    # guarantee the output row order matches the input row order (the PIT
    # join may reorder or drop rows); the column is carried through the
    # join and used to realign results below.
    offline_input = entity_df.copy()
    offline_input[ROW_ID_COLUMN] = range(len(offline_input))
    offline_df = store.get_historical_features(
        entity_df=offline_input,
        features=list(feature_refs),
        full_feature_names=full_feature_names,
    ).to_df()
    offline_df = offline_df.set_index(ROW_ID_COLUMN).sort_index()
    # Every input row must come back from the PIT join (with nulls where no
    # in-window row exists). A missing row would otherwise be silently
    # excluded from the denominator, making the mismatch rate uncomputable.
    missing_rows = set(range(len(entity_df))) - set(offline_df.index)
    if missing_rows:
        raise RuntimeError(
            "The offline point-in-time join dropped input rows "
            f"(row ids {sorted(missing_rows)}); cannot compute a consistent "
            "mismatch denominator."
        )
    offline_df = offline_df.reindex(range(len(entity_df)))

    # Online side: batched latest-value reads. The online store has no time
    # travel parameter by design, so event time enters only through the
    # offline join and TTL checks — never through this call. A single call
    # per batch supplies both values (to_df) and per-cell FieldStatus
    # (response proto), so validation never doubles online-store traffic.
    online_frames: List[pd.DataFrame] = []
    online_statuses: Dict[Tuple[int, str], str] = {}
    batch_size = config.online_read_batch_size
    row_offset = 0
    for start in range(0, len(entity_df), batch_size):
        batch = entity_df.iloc[start : start + batch_size]
        entity_rows = _entity_rows_for_online_read(batch, join_keys)
        response = store.get_online_features(
            features=list(feature_refs),
            entity_rows=entity_rows,
            full_feature_names=full_feature_names,
        )
        online_frames.append(
            response.to_df(include_event_timestamps=True).reset_index(drop=True)
        )
        proto = response.proto
        for column_index, name in enumerate(proto.metadata.feature_names.val):
            vector = proto.results[column_index]
            for row_index, status in enumerate(vector.statuses):
                online_statuses[(row_offset + row_index, name)] = (
                    _FIELD_STATUS_NAMES.get(int(status), "INVALID")
                )
        row_offset += len(batch)
    online_df = (
        pd.concat(online_frames, ignore_index=True) if online_frames else pd.DataFrame()
    )

    report = ConsistencyReport(
        total_rows=len(entity_df),
        config={
            "float_rtol": config.float_rtol,
            "float_atol": config.float_atol,
            "max_sample_mismatches": config.max_sample_mismatches,
            "check_out_of_retention": config.check_out_of_retention,
            "nan_matches_nan": config.nan_matches_nan,
            "prepare_online_store": config.prepare_online_store,
            "online_read_batch_size": config.online_read_batch_size,
        },
    )

    output_names = _output_feature_names(feature_refs, full_feature_names)
    event_times = _normalize_event_timestamps(entity_df[EVENT_TIMESTAMP_COLUMN])
    ttl_by_feature = _ttl_by_output_name(feature_views, output_names)
    # Retention floor per feature in pure event time: a validation event
    # time is out of retention only when it predates (newest source event -
    # TTL). This keeps the decision independent of processing/wall-clock
    # time, unlike the online store's OUTSIDE_MAX_AGE health flag.
    retention_floor_by_feature = _retention_floor_by_output_name(
        feature_views, output_names
    )

    for row_index in range(len(entity_df)):
        entity_values = {
            key: _json_safe(entity_df.iloc[row_index][key]) for key in join_keys
        }
        event_time = event_times[row_index]
        for output_name in output_names:
            ref = output_name
            offline_value = offline_df.iloc[row_index].get(output_name)
            online_value = online_df.iloc[row_index].get(output_name)
            status_name = online_statuses.get((row_index, output_name), "INVALID")
            offline_type = _type_name(offline_value)
            online_type = _type_name(online_value)

            reason, detail = _classify_cell(
                config=config,
                feature_ref=ref,
                event_time=event_time,
                ttl=ttl_by_feature.get(output_name),
                retention_floor=retention_floor_by_feature.get(output_name),
                offline_value=offline_value,
                online_value=online_value,
                offline_type=offline_type,
                online_type=online_type,
                online_status=status_name,
            )

            report.total_compared_cells += 1
            _bump_feature_stat(report, ref, compared=1, mismatched=0, reason=None)
            if reason is not MismatchReason.VALUE_MATCH:
                report.total_mismatch_cells += 1
                report.reason_counts[reason.value] = (
                    report.reason_counts.get(reason.value, 0) + 1
                )
                _bump_feature_stat(
                    report, ref, compared=0, mismatched=1, reason=reason.value
                )
                if len(report.sample_mismatches) < config.max_sample_mismatches:
                    report.sample_mismatches.append(
                        ConsistencyMismatch(
                            row_index=row_index,
                            feature_ref=ref,
                            entity_values=entity_values,
                            event_timestamp=event_time.isoformat(),
                            reason=reason.value,
                            offline_value=_json_safe(offline_value),
                            online_value=_json_safe(online_value),
                            offline_type=offline_type,
                            online_type=online_type,
                            online_status=status_name,
                            detail=detail,
                        )
                    )

    for stats in report.per_feature_stats.values():
        compared = stats["compared_cells"]
        stats["mismatch_rate"] = (
            stats["mismatch_cells"] / compared if compared else None
        )
    return report


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _classify_cell(
    config: ConsistencyValidationConfig,
    feature_ref: str,
    event_time: datetime,
    ttl: Optional[timedelta],
    offline_value: Any,
    online_value: Any,
    offline_type: str,
    online_type: str,
    online_status: str,
    retention_floor: Optional[datetime] = None,
) -> Tuple[MismatchReason, str]:
    """Assign a mismatch reason to one cell.

    Priority order is deliberate: a cell whose event time is genuinely
    beyond the source data's retention window is reported as out of
    retention before anything else, because the time-window failure is the
    actionable root cause. Online unavailability outranks a bare missing
    value so that "feature not served" is never hidden inside a generic
    null comparison.
    """
    if (
        config.check_out_of_retention
        and ttl is not None
        and ttl.total_seconds() > 0
        and _is_outside_retention(
            online_status, event_time, ttl, offline_value, retention_floor
        )
    ):
        return (
            MismatchReason.OUT_OF_RETENTION,
            f"validation event time {event_time.isoformat()} falls outside the "
            f"feature view TTL window ({ttl}): the offline point-in-time join has no "
            "in-window source row, so no valid feature value exists at this event time",
        )

    if online_status in ("NOT_FOUND", "INVALID"):
        return (
            MismatchReason.ONLINE_UNAVAILABLE,
            f"online store status for {feature_ref} is {online_status}; the feature "
            "value is not served and must not be counted as a match",
        )

    offline_missing = _is_missing(offline_value)
    online_missing = _is_missing(online_value)
    if offline_missing or online_missing:
        if offline_missing and online_missing:
            # Both sides agree on absence with the feature still served:
            # this is a consistent null, not a missing-value mismatch.
            return MismatchReason.VALUE_MATCH, "both sides are null"
        return (
            MismatchReason.VALUE_MISSING,
            f"value present on only one side (offline_missing={offline_missing}, "
            f"online_missing={online_missing})",
        )

    if not _types_compatible(offline_value, online_value):
        return (
            MismatchReason.TYPE_MISMATCH,
            f"incompatible value types: offline={offline_type}, online={online_type}",
        )

    if _values_equal(config, offline_value, online_value):
        return MismatchReason.VALUE_MATCH, "values equal within tolerance"

    if online_status == "OUTSIDE_MAX_AGE":
        return (
            MismatchReason.WINDOW_MISMATCH,
            "online value is served but marked OUTSIDE_MAX_AGE: its event timestamp "
            "is outside the feature view TTL, indicating the online store holds a "
            "different (older or mis-windowed) row than the offline point-in-time join",
        )

    if isinstance(offline_value, float) or isinstance(online_value, float):
        return (
            MismatchReason.SEMANTIC_MISMATCH,
            "numeric values differ beyond tolerance with both values present and "
            "within TTL; typical cause is different batch vs streaming computation "
            "semantics (aggregation window, dedup or late-data handling)",
        )

    return (
        MismatchReason.VALUE_MISMATCH,
        f"exact-equality mismatch: offline={offline_value!r}, online={online_value!r}",
    )


def _is_outside_retention(
    online_status: str,
    event_time: datetime,
    ttl: timedelta,
    offline_value: Any,
    retention_floor: Optional[datetime] = None,
) -> bool:
    """Decide whether a cell's validation event time is out of retention.

    Retention is an *event-time* property and must stay independent of the
    online store's processing-time health flag. A cell is out of retention
    only when the offline point-in-time join has no in-window source row
    (returns a null) and the requested event time predates the retention
    floor. The preferred floor is derived in event time from the source
    data's newest event timestamp (``max_source_event_ts - ttl``); only
    when that cannot be determined do we fall back to one TTL before
    processing time. A null at an event time inside the retention window
    instead means the data is genuinely absent and is classified later as
    ``value_missing`` / ``online_unavailable``.

    ``OUTSIDE_MAX_AGE`` deliberately does not trigger this branch: it is
    computed against wall-clock time, so validating historical event times
    on aged-out-but-materialized data would otherwise produce false
    retention failures. It is consumed downstream as window-mismatch
    evidence when the values themselves differ.
    """
    if not _is_missing(offline_value):
        return False
    if retention_floor is None:
        retention_floor = datetime.now(timezone.utc) - ttl
    return event_time < retention_floor


def _retention_floor_by_output_name(
    feature_views: Dict[str, "FeatureView"], output_names: Sequence[str]
) -> Dict[str, Optional[datetime]]:
    """Event-time retention floor (``max_source_ts - ttl``) per output feature.

    Reading the source's newest event timestamp is best-effort: data sources
    that cannot be introspected locally (remote warehouses, SQL strings)
    yield ``None``, in which case the classifier falls back to the
    processing-time floor. Failures here never block validation.
    """
    floors: Dict[str, Optional[datetime]] = {}
    view_floors: Dict[str, Optional[datetime]] = {}
    for view_name, view in feature_views.items():
        view_floors[view_name] = _source_retention_floor(view)
    for name in output_names:
        if "__" in name:
            view_name = name.split("__", 1)[0]
            floors[name] = view_floors.get(view_name)
        else:
            owners = [
                view_name
                for view_name, view in feature_views.items()
                if any(f.name == name for f in view.features)
            ]
            floors[name] = view_floors.get(owners[0]) if len(owners) == 1 else None
    return floors


def _source_retention_floor(view: "FeatureView") -> Optional[datetime]:
    ttl = getattr(view, "ttl", None)
    source = getattr(view, "batch_source", None) or getattr(view, "source", None)
    if source is None or ttl is None or ttl.total_seconds() <= 0:
        return None
    path = getattr(source, "path", None)
    timestamp_field = getattr(source, "timestamp_field", None)
    if not path or not timestamp_field:
        return None
    try:
        path_str = str(path).replace("file://", "")
        if path_str.startswith(("s3://", "gs://", "http://", "https://")):
            return None
        frame = pd.read_parquet(path_str, columns=[timestamp_field])
        max_ts = pd.to_datetime(frame[timestamp_field]).max()
        if pd.isna(max_ts):
            return None
        max_dt = max_ts.to_pydatetime()
        if max_dt.tzinfo is None:
            max_dt = max_dt.replace(tzinfo=timezone.utc)
        return max_dt.astimezone(timezone.utc) - ttl
    except Exception:
        return None


def _values_equal(config: ConsistencyValidationConfig, a: Any, b: Any) -> bool:
    if _is_float_like(a) or _is_float_like(b):
        return _floats_close(config, float(a), float(b))
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        if not isinstance(a, (list, tuple)) or not isinstance(b, (list, tuple)):
            return False
        if len(a) != len(b):
            return False
        return all(_values_equal(config, x, y) for x, y in zip(a, b))
    return a == b


def _is_float_like(value: Any) -> bool:
    """True for Python floats and NumPy floating-point scalars."""
    if isinstance(value, float):
        return True
    try:
        import numpy as np

        return isinstance(value, np.floating)
    except ImportError:  # pragma: no cover
        return False


def _is_integer_like(value: Any) -> bool:
    if isinstance(value, int) and not isinstance(value, bool):
        return True
    try:
        import numpy as np

        return isinstance(value, np.integer)
    except ImportError:  # pragma: no cover
        return False


def _floats_close(config: ConsistencyValidationConfig, a: float, b: float) -> bool:
    if math.isnan(a) and math.isnan(b):
        return config.nan_matches_nan
    if math.isnan(a) or math.isnan(b):
        return False
    # numpy.isclose semantics, implemented without a numpy dependency:
    # |a - b| <= atol + rtol * |b|. The absolute term keeps comparisons
    # near zero meaningful; the relative term covers float32/float64 and
    # aggregation-order rounding that scales with magnitude.
    return abs(a - b) <= config.float_atol + config.float_rtol * abs(b)


def _types_compatible(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool)
    # Online stores deserialize every numeric ValueProto to float, so an
    # offline integer compared with an online float is the expected
    # serialization round-trip, not an incompatible type. Genuine
    # string-vs-number (or list-vs-scalar) differences still fail below.
    if (_is_float_like(a) or _is_integer_like(a)) and (
        _is_float_like(b) or _is_integer_like(b)
    ):
        return True
    if isinstance(a, str) or isinstance(b, str):
        return isinstance(a, str) and isinstance(b, str)
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        return isinstance(a, (list, tuple)) and isinstance(b, (list, tuple))
    return type(a) is type(b)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    try:
        if pd.isna(value):
            return "null"
    except (TypeError, ValueError):
        pass
    return type(value).__name__


# ---------------------------------------------------------------------------
# Feature/entity resolution helpers
# ---------------------------------------------------------------------------


def _resolve_feature_refs(store: "FeatureStore", features: Sequence[str]) -> List[str]:
    from feast import utils as feast_utils

    refs = feast_utils._get_features(
        store.registry, store.project, list(features), allow_cache=True
    )
    if not refs:
        raise ValueError("No features were resolved from the provided feature refs.")
    return refs


def _load_feature_views(
    store: "FeatureStore", feature_refs: Sequence[str]
) -> Dict[str, "FeatureView"]:
    views: Dict[str, "FeatureView"] = {}
    for ref in feature_refs:
        view_name = ref.split(":")[0]
        if view_name not in views:
            views[view_name] = store.registry.get_feature_view(view_name, store.project)
    return views


def _collect_join_keys(
    feature_views: Dict[str, "FeatureView"],
) -> List[str]:
    keys: List[str] = []
    for view in feature_views.values():
        for key in view.join_keys:
            if key not in keys:
                keys.append(key)
    return keys


def _validate_entity_columns(entity_df: pd.DataFrame, join_keys: Sequence[str]) -> None:
    missing = [key for key in join_keys if key not in entity_df.columns]
    if missing:
        raise ValueError(
            "Entity dataframe is missing join key columns required by the "
            f"requested feature views: {missing}"
        )


def _entity_rows_for_online_read(
    batch: pd.DataFrame, join_keys: Sequence[str]
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for _, row in batch.iterrows():
        rows.append({key: row[key] for key in join_keys})
    return rows


def _output_feature_names(
    feature_refs: Sequence[str],
    full_feature_names: bool,
) -> List[str]:
    names: List[str] = []
    for ref in feature_refs:
        view_name, feature_name = ref.split(":", 1)
        names.append(
            f"{view_name}__{feature_name}" if full_feature_names else feature_name
        )
    return names


def _ttl_by_output_name(
    feature_views: Dict[str, "FeatureView"], output_names: Sequence[str]
) -> Dict[str, Optional[timedelta]]:
    result: Dict[str, Optional[timedelta]] = {}
    for name in output_names:
        view_name = name.split("__", 1)[0] if "__" in name else None
        if view_name is None:
            # Bare feature name: find the unique view that owns it.
            owners = [
                view
                for view in feature_views.values()
                if any(f.name == name for f in view.features)
            ]
            view = owners[0] if len(owners) == 1 else None
        else:
            view = feature_views.get(view_name)
        result[name] = getattr(view, "ttl", None) if view is not None else None
    return result


def _normalize_event_timestamps(series: pd.Series) -> List[datetime]:
    result: List[datetime] = []
    for value in series:
        ts = pd.Timestamp(value).to_pydatetime()
        if ts.tzinfo is None:
            # Naive timestamps are interpreted as UTC, matching Feast's
            # make_tzaware convention; event time stays event time.
            ts = ts.replace(tzinfo=timezone.utc)
        result.append(ts.astimezone(timezone.utc))
    return result


def _materialize_for_validation(
    store: "FeatureStore",
    feature_views: Dict[str, "FeatureView"],
    entity_df: pd.DataFrame,
) -> None:
    """Optional preparation step: materialize up to the max event time.

    Only invoked when ``prepare_online_store=True``; default validation runs
    never write to the online store. Materialization uses each view's own
    TTL to pick a safe start date, mirroring how operators schedule it.
    """
    max_event_time = max(_normalize_event_timestamps(entity_df[EVENT_TIMESTAMP_COLUMN]))
    for view in feature_views.values():
        ttl = getattr(view, "ttl", None) or timedelta(days=1)
        start = max_event_time - ttl
        store.materialize(
            feature_views=[view.name],
            start_date=start,
            end_date=max_event_time,
        )


def _bump_feature_stat(
    report: ConsistencyReport,
    feature_ref: str,
    compared: int,
    mismatched: int,
    reason: Optional[str],
) -> None:
    stats = report.per_feature_stats.setdefault(
        feature_ref,
        {"compared_cells": 0, "mismatch_cells": 0, "reason_counts": {}},
    )
    stats["compared_cells"] += compared
    stats["mismatch_cells"] += mismatched
    if reason is not None:
        stats["reason_counts"][reason] = stats["reason_counts"].get(reason, 0) + 1


def _json_safe(value: Any) -> Any:
    if value is None:
        return None
    # Convert NumPy scalars (np.int64, np.float32, ...) to native Python so
    # the JSON report serializes without a custom encoder.
    try:
        import numpy as np

        if isinstance(value, np.generic):
            native = value.item()
            if isinstance(native, float) and math.isnan(native):
                return None
            return _json_safe(native)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value
