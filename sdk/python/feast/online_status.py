# Copyright 2026 The Feast Authors
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

"""Per-row, per-feature online value status tracking (opt-in).

This module implements the status side-car for online feature retrieval.
When ``online_status.enabled`` is set in ``feature_store.yaml``, every
``get_online_features`` call additionally produces an
:class:`OnlineStatusReport` that explains, for every row and every
feature, *why* the returned value looks the way it does — instead of
silently surfacing a bare ``None``.

Design invariants:

* The classification is **mutually exclusive and exhaustive**: every
  (row, feature) pair is assigned exactly one :class:`ValueStatus`.
  There is deliberately no "other"/"unknown" bucket.
* Statuses are computed from the **same snapshot** of registry metadata
  (the ``FeatureView`` objects already resolved for the request) and the
  **same** online-store rows returned by the single ``online_read``
  call. No extra registry lookup and no extra online-store round-trip
  is performed to derive them.
* The existing response structure is untouched; the report is attached
  to :class:`feast.online_response.OnlineResponse` as a side-car.
* Nothing in this module logs feature values or entity identifiers —
  only feature-view names, batch indices, counts and error *types*.
"""

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from feast.protos.feast.types.Value_pb2 import (
    BoolList,
    BytesList,
    DoubleList,
    FloatList,
    Int32List,
    Int64List,
    StringList,
)
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.value_type import ValueType

logger = logging.getLogger(__name__)


class ValueStatus(str, enum.Enum):
    """The mutually exclusive, exhaustive per-(row, feature) retrieval outcome.

    The string values double as Prometheus ``status`` label values for
    the existing ``feast_online_features_status_total`` counter.
    """

    #: Value present in the online store, type-checks against the
    #: declared feature dtype, and is inside the feature view's TTL.
    PRESENT = "present"
    #: The entity key does not exist in the online store at all.
    ENTITY_NOT_FOUND = "entity_not_found"
    #: The entity exists (other features were returned for it) but this
    #: feature was never materialized for it.
    FEATURE_NOT_MATERIALIZED = "feature_not_materialized"
    #: The feature was materialized but the stored value is null.
    NULL_VALUE = "null_value"
    #: The stored value's type violates the declared feature dtype
    #: (e.g. a string read back for a numeric feature).
    UNSUPPORTED_TYPE = "unsupported_type"
    #: The requested feature view is not registered in the registry.
    FEATURE_VIEW_NOT_REGISTERED = "feature_view_not_registered"
    #: Present but older than the feature view's TTL.
    OUTSIDE_MAX_AGE = "outside_max_age"
    #: The online store could not be read (after bounded retries); the
    #: value is *unavailable*, not *missing*.
    STORE_UNAVAILABLE = "store_unavailable"


#: Statuses whose response value is null and that are eligible for
#: explicit default filling. ``OUTSIDE_MAX_AGE`` is intentionally
#: excluded: a stale-but-real value is never overwritten by a default.
FILLABLE_STATUSES = frozenset(
    {
        ValueStatus.ENTITY_NOT_FOUND,
        ValueStatus.FEATURE_NOT_MATERIALIZED,
        ValueStatus.NULL_VALUE,
        ValueStatus.UNSUPPORTED_TYPE,
        ValueStatus.FEATURE_VIEW_NOT_REGISTERED,
        ValueStatus.STORE_UNAVAILABLE,
    }
)


class StoreReadFailure:
    """Sentinel placed in ``read_rows`` when a batch read failed.

    Carries no entity identifiers and no feature values — only the
    failure metadata needed for the report and logs.
    """

    __slots__ = ("attempts", "error_type")

    def __init__(self, attempts: int, error_type: str):
        self.attempts = attempts
        self.error_type = error_type

    def __repr__(self) -> str:
        return f"StoreReadFailure(attempts={self.attempts}, error_type={self.error_type!r})"


@dataclass
class FailedBatch:
    """One read batch that exhausted its retries.

    Identifies the failed slice by feature view, batch index and row
    count only — never by entity key.
    """

    feature_view: str
    batch_index: int
    row_count: int
    attempts: int
    error_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_view": self.feature_view,
            "batch_index": self.batch_index,
            "row_count": self.row_count,
            "attempts": self.attempts,
            "error_type": self.error_type,
        }


@dataclass
class FeatureViewStatusBlock:
    """Status matrix for the features of one feature view."""

    feature_view: str
    #: Registry version of the feature view in the snapshot used for the
    #: read (``None`` when the view is not registered at all).
    version: Optional[int]
    #: Result column names, in response order (respecting full_feature_names).
    features: List[str]
    #: column name -> per-row status.
    statuses: Dict[str, List[ValueStatus]]
    #: Per-row event timestamp of the stored row (shared by all features
    #: of this view); ``None`` when the entity was not found.
    event_timestamps: List[Optional[datetime]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_view": self.feature_view,
            "version": self.version,
            "event_timestamps": [
                ts.isoformat() if ts is not None else None
                for ts in self.event_timestamps
            ],
            "features": {
                name: [s.value for s in col] for name, col in self.statuses.items()
            },
        }


@dataclass
class OnlineStatusReport:
    """Side-car attached to ``OnlineResponse`` when status tracking is on."""

    blocks: List[FeatureViewStatusBlock] = field(default_factory=list)
    failed_batches: List[FailedBatch] = field(default_factory=list)
    #: True when the registry snapshot changed underneath the request
    #: (registry metadata and online data may come from different
    #: versions); individual statuses should then be treated as suspect.
    registry_version_mismatch: bool = False
    #: True when default filling was applied to the returned values.
    defaults_filled: bool = False

    def status_of(self, feature: str) -> List[ValueStatus]:
        """Return the per-row statuses for a result column."""
        for block in self.blocks:
            if feature in block.statuses:
                return block.statuses[feature]
        raise KeyError(f"Feature {feature!r} is not covered by the status report")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "registry_version_mismatch": self.registry_version_mismatch,
            "defaults_filled": self.defaults_filled,
            "failed_batches": [b.to_dict() for b in self.failed_batches],
            "feature_views": [b.to_dict() for b in self.blocks],
        }


# ---------------------------------------------------------------------------
# Type contract validation
# ---------------------------------------------------------------------------

# Declared ValueType -> acceptable ValueProto oneof "val" field names.
# Numeric types accept their narrower/wider sibling because stores do not
# always preserve the int32/int64 or float/double distinction; a *string*
# read back for a numeric feature is exactly the contract violation this
# check exists to catch.
_VALUE_TYPE_TO_PROTO_ONEOF: Dict[ValueType, frozenset] = {
    ValueType.INT32: frozenset({"int32_val", "int64_val"}),
    ValueType.INT64: frozenset({"int64_val", "int32_val"}),
    ValueType.FLOAT: frozenset({"float_val", "double_val"}),
    ValueType.DOUBLE: frozenset({"double_val", "float_val"}),
    ValueType.STRING: frozenset({"string_val"}),
    ValueType.BYTES: frozenset({"bytes_val"}),
    ValueType.BOOL: frozenset({"bool_val"}),
    ValueType.UNIX_TIMESTAMP: frozenset({"unix_timestamp_val", "int64_val"}),
    ValueType.INT32_LIST: frozenset({"int32_list_val", "int64_list_val"}),
    ValueType.INT64_LIST: frozenset({"int64_list_val", "int32_list_val"}),
    ValueType.FLOAT_LIST: frozenset({"float_list_val", "double_list_val"}),
    ValueType.DOUBLE_LIST: frozenset({"double_list_val", "float_list_val"}),
    ValueType.STRING_LIST: frozenset({"string_list_val"}),
    ValueType.BYTES_LIST: frozenset({"bytes_list_val"}),
    ValueType.BOOL_LIST: frozenset({"bool_list_val"}),
    ValueType.UNIX_TIMESTAMP_LIST: frozenset(
        {"unix_timestamp_list_val", "int64_list_val"}
    ),
}


def _is_type_compatible(value: ValueProto, value_type: Optional[ValueType]) -> bool:
    """Check a stored proto value against the declared feature type.

    Types without a fixed proto representation (maps, structs, nested
    collections) are not validated and therefore never flagged — the
    check only fires on a positive mismatch.
    """
    if value_type is None:
        return True
    expected = _VALUE_TYPE_TO_PROTO_ONEOF.get(value_type)
    if expected is None:
        return True
    return value.WhichOneof("val") in expected


# ---------------------------------------------------------------------------
# Status computation
# ---------------------------------------------------------------------------


def compute_feature_view_statuses(
    table: Any,
    requested_features: Sequence[str],
    read_rows: Sequence[Tuple[Optional[datetime], Optional[Dict[str, ValueProto]]]],
    indexes: Sequence[Sequence[int]],
    output_len: int,
    full_feature_names: bool,
) -> FeatureViewStatusBlock:
    """Classify every (row, feature) of one feature view.

    Uses the exact same inputs as
    ``utils._populate_response_from_feature_data`` — the ``FeatureView``
    object resolved from the registry snapshot and the rows returned by
    the single ``online_read`` call — so the classification can never
    diverge from the data actually returned.
    """
    table_name = table.projection.name_to_use()
    columns = [
        f"{table_name}__{fn}" if full_feature_names else fn for fn in requested_features
    ]
    field_types: Dict[str, ValueType] = {}
    for f in getattr(table, "features", []):
        try:
            field_types[f.name] = f.dtype.to_value_type()
        except Exception:
            continue

    ttl = getattr(table, "ttl", None)
    ttl_secs = int(ttl.total_seconds()) if ttl else 0
    now_secs = int(datetime.now(timezone.utc).timestamp()) if ttl_secs else 0

    statuses: Dict[str, List[ValueStatus]] = {
        col: [ValueStatus.ENTITY_NOT_FOUND] * output_len for col in columns
    }
    row_timestamps: List[Optional[datetime]] = [None] * output_len

    for row_idx, destinations in enumerate(indexes):
        row_ts, feature_data = read_rows[row_idx]
        for out_idx in destinations:
            row_timestamps[out_idx] = row_ts

        if isinstance(feature_data, StoreReadFailure):
            for col in columns:
                for out_idx in destinations:
                    statuses[col][out_idx] = ValueStatus.STORE_UNAVAILABLE
            continue
        if feature_data is None:
            # Entity key absent from the store: statuses stay
            # ENTITY_NOT_FOUND (the default above).
            continue

        expired = (
            ttl_secs > 0
            and row_ts is not None
            and now_secs - int(row_ts.timestamp()) > ttl_secs
        )
        for fn, col in zip(requested_features, columns):
            if fn not in feature_data:
                status = ValueStatus.FEATURE_NOT_MATERIALIZED
            else:
                value = feature_data[fn]
                if value.WhichOneof("val") is None:
                    status = ValueStatus.NULL_VALUE
                elif not _is_type_compatible(value, field_types.get(fn)):
                    status = ValueStatus.UNSUPPORTED_TYPE
                elif expired:
                    status = ValueStatus.OUTSIDE_MAX_AGE
                else:
                    status = ValueStatus.PRESENT
            for out_idx in destinations:
                statuses[col][out_idx] = status

    version = getattr(table, "current_version_number", 0) or 0
    return FeatureViewStatusBlock(
        feature_view=table.name,
        version=version,
        features=columns,
        statuses=statuses,
        event_timestamps=row_timestamps,
    )


def unregistered_feature_block(
    view_name: str,
    columns: Sequence[str],
    output_len: int,
) -> FeatureViewStatusBlock:
    """Build the status block for a feature view missing from the registry."""
    return FeatureViewStatusBlock(
        feature_view=view_name,
        version=None,
        features=list(columns),
        statuses={
            col: [ValueStatus.FEATURE_VIEW_NOT_REGISTERED] * output_len
            for col in columns
        },
        event_timestamps=[None] * output_len,
    )


def count_statuses(block: FeatureViewStatusBlock) -> Dict[str, int]:
    """Aggregate a block's statuses into ``{status_label: count}``."""
    counts: Dict[str, int] = {}
    for col_statuses in block.statuses.values():
        for status in col_statuses:
            counts[status.value] = counts.get(status.value, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Default filling (explicit opt-in only)
# ---------------------------------------------------------------------------

_UNSET = object()

_DEFAULT_PROTO_FACTORIES = {
    ValueType.INT32: lambda: ValueProto(int32_val=0),
    ValueType.INT64: lambda: ValueProto(int64_val=0),
    ValueType.FLOAT: lambda: ValueProto(float_val=0.0),
    ValueType.DOUBLE: lambda: ValueProto(double_val=0.0),
    ValueType.STRING: lambda: ValueProto(string_val=""),
    ValueType.BYTES: lambda: ValueProto(bytes_val=b""),
    ValueType.BOOL: lambda: ValueProto(bool_val=False),
    ValueType.UNIX_TIMESTAMP: lambda: ValueProto(unix_timestamp_val=0),
    ValueType.INT32_LIST: lambda: ValueProto(int32_list_val=Int32List()),
    ValueType.INT64_LIST: lambda: ValueProto(int64_list_val=Int64List()),
    ValueType.FLOAT_LIST: lambda: ValueProto(float_list_val=FloatList()),
    ValueType.DOUBLE_LIST: lambda: ValueProto(double_list_val=DoubleList()),
    ValueType.STRING_LIST: lambda: ValueProto(string_list_val=StringList()),
    ValueType.BYTES_LIST: lambda: ValueProto(bytes_list_val=BytesList()),
    ValueType.BOOL_LIST: lambda: ValueProto(bool_list_val=BoolList()),
    ValueType.UNIX_TIMESTAMP_LIST: lambda: ValueProto(int64_list_val=Int64List()),
}


def _default_value_proto(value_type: Optional[ValueType], override: Any) -> ValueProto:
    """Build the fill value for a feature.

    A configured per-feature override wins; otherwise a type-appropriate
    zero value is used. Types without a zero value stay null.
    """
    if override is not _UNSET:
        from feast.type_map import python_values_to_proto_values

        return python_values_to_proto_values(
            [override], value_type or ValueType.UNKNOWN
        )[0]
    factory = _DEFAULT_PROTO_FACTORIES.get(value_type) if value_type else None
    if factory is None:
        return ValueProto()
    return factory()


def apply_default_fills(
    online_features_response: Any,
    report: OnlineStatusReport,
    feature_types: Mapping[str, ValueType],
    configured_defaults: Mapping[str, Any],
) -> None:
    """Replace fillable null values in the response proto with defaults.

    Only runs when ``online_status.fill_defaults`` is explicitly enabled.
    The report keeps the *original* statuses, so downstream consumers can
    always tell a filled default apart from a real value.
    """
    names = list(online_features_response.metadata.feature_names.val)
    for block in report.blocks:
        for col, col_statuses in block.statuses.items():
            if col not in names:
                continue
            vector = online_features_response.results[names.index(col)]
            value_type = feature_types.get(col)
            override = configured_defaults.get(col, _UNSET)
            for row_idx, status in enumerate(col_statuses):
                if status in FILLABLE_STATUSES and row_idx < len(vector.values):
                    vector.values[row_idx].CopyFrom(
                        _default_value_proto(value_type, override)
                    )
