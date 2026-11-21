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
Unit tests for offline/online feature consistency validation.

The four acceptance scenarios are covered end to end on the local stack
(file offline store + SQLite online store, no external infrastructure):

1. consistent values after materialization;
2. inconsistency caused by a wrong aggregation window (online holds a
   different row than the offline point-in-time join);
3. feature missing online (``NOT_FOUND`` must count as a mismatch);
4. validation event time outside the feature view's retention window.

Additional tests pin down the mismatch-rate numerator/denominator, empty
dataset handling, batching/merging, float tolerance, and type differences.

Processing time is frozen with ``freezegun`` throughout. Feast's online
``OUTSIDE_MAX_AGE`` status is computed against wall-clock time, while
offline point-in-time joins use event time only; freezing keeps the two
time semantics independently controllable inside the tests.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
from freezegun import freeze_time

from feast import (
    ConsistencyReport,
    ConsistencyValidationConfig,
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    RepoConfig,
)
from feast.consistency import MismatchReason
from feast.errors import (
    ConsistencyTimestampColumnMissingError,
    EmptyConsistencyDatasetError,
)
from feast.types import Float32, Int64

_FV_NAME = "driver_stats"
_FEATURE_VALUE = "conv_rate"
_FEATURE_TRIPS = "avg_daily_trips"
_JOIN_KEY = "driver_id"

# A fixed processing-time anchor. All source rows sit well inside the TTL
# window relative to this instant, so materialized rows are PRESENT online.
_NOW = datetime(2026, 1, 10, 12, 0, 0, tzinfo=timezone.utc)
_TTL = timedelta(days=2)


def _build_source_dataframe() -> pd.DataFrame:
    """Two entities, each with two hourly rows (older and latest)."""
    rows = []
    for driver_id in (1, 2):
        for hours_ago, value, trips in (
            (2, 0.10, 10),
            (1, 0.20, 20),
        ):
            rows.append(
                {
                    _JOIN_KEY: driver_id,
                    _FEATURE_VALUE: value + driver_id,
                    _FEATURE_TRIPS: trips + driver_id,
                    "event_timestamp": _NOW - timedelta(hours=hours_ago),
                    "created_timestamp": _NOW - timedelta(hours=hours_ago),
                }
            )
    return pd.DataFrame(rows)


def _make_store(tmp_path: Path) -> tuple[FeatureStore, FeatureView]:
    source_path = tmp_path / "driver_stats.parquet"
    _build_source_dataframe().to_parquet(source_path)

    source = FileSource(
        path=str(source_path),
        timestamp_field="event_timestamp",
        created_timestamp_column="created_timestamp",
    )
    entity = Entity(name=_JOIN_KEY, join_keys=[_JOIN_KEY])
    view = FeatureView(
        name=_FV_NAME,
        entities=[entity],
        schema=[
            Field(name=_FEATURE_VALUE, dtype=Float32),
            Field(name=_FEATURE_TRIPS, dtype=Int64),
        ],
        ttl=_TTL,
        source=source,
    )
    store = FeatureStore(
        config=RepoConfig(
            project="consistency_test",
            registry=str(tmp_path / "registry.db"),
            provider="local",
            online_store={"type": "sqlite", "path": str(tmp_path / "online.db")},
            offline_store={"type": "file"},
            entity_key_serialization_version=3,
        )
    )
    store.apply([entity, view])
    return store, view


def _entity_dataframe() -> pd.DataFrame:
    # Event time after the latest source rows so the PIT join selects the
    # 1-hour-old row for each entity.
    event_time = _NOW - timedelta(minutes=30)
    return pd.DataFrame(
        {
            _JOIN_KEY: [1, 2],
            "event_timestamp": [event_time, event_time],
        }
    )


@pytest.fixture
def prepared_store(tmp_path: Path):
    store, view = _make_store(tmp_path)
    try:
        with freeze_time(_NOW):
            store.materialize(
                feature_views=[_FV_NAME],
                start_date=_NOW - timedelta(days=1),
                end_date=_NOW,
            )
            yield store, _entity_dataframe(), view
    finally:
        store.teardown()


def _overwrite_online_row(
    store: FeatureStore, view: FeatureView, driver_id: int, value: float, trips: int
) -> None:
    """Write a fresh, PRESENT online row with an arbitrary value."""
    fresh_ts = _NOW - timedelta(minutes=30)
    store.write_to_online_store(
        view.name,
        df=pd.DataFrame(
            [
                {
                    _JOIN_KEY: driver_id,
                    _FEATURE_VALUE: value,
                    _FEATURE_TRIPS: trips,
                    "event_timestamp": fresh_ts,
                    "created_timestamp": fresh_ts,
                }
            ]
        ),
    )


# ---------------------------------------------------------------------------
# Scenario 1: consistent values
# ---------------------------------------------------------------------------


def test_consistent_values_after_materialization(prepared_store):
    store, entity_df, _view = prepared_store
    with freeze_time(_NOW):
        report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=[
                f"{_FV_NAME}:{_FEATURE_VALUE}",
                f"{_FV_NAME}:{_FEATURE_TRIPS}",
            ],
        )

    assert report.total_rows == 2
    # 2 rows x 2 features = 4 compared cells; denominator definition under test.
    assert report.total_compared_cells == 4
    assert report.total_mismatch_cells == 0
    assert report.mismatch_rate == 0.0
    assert report.reason_counts == {}
    assert report.sample_mismatches == []
    for stats in report.per_feature_stats.values():
        assert stats["compared_cells"] == 2
        assert stats["mismatch_cells"] == 0
        assert stats["mismatch_rate"] == 0.0


# ---------------------------------------------------------------------------
# Scenario 2: wrong aggregation window -> mismatch with attribution
# ---------------------------------------------------------------------------


def test_wrong_window_online_value_is_flagged(prepared_store):
    store, entity_df, view = prepared_store
    with freeze_time(_NOW):
        # Overwrite driver 1's online value with a different number while
        # keeping a fresh event timestamp (PRESENT). The offline PIT join
        # still returns the materialized 1-hour-old value: the two
        # computation paths disagree for the same entity/event time.
        _overwrite_online_row(store, view, driver_id=1, value=9.99, trips=21)
        report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )

    assert report.total_compared_cells == 2
    assert report.total_mismatch_cells == 1
    # Explicit, recomputable numerator/denominator assertion.
    assert report.mismatch_rate == (
        report.total_mismatch_cells / report.total_compared_cells
    )
    mismatch = report.sample_mismatches[0]
    assert mismatch.row_index == 0
    assert mismatch.reason == MismatchReason.SEMANTIC_MISMATCH.value
    assert mismatch.online_status == "PRESENT"
    assert mismatch.offline_value != mismatch.online_value


def test_stale_online_row_is_classified_as_window_mismatch(prepared_store):
    """An aged-out online row that disagrees -> window mismatch."""
    store, entity_df, view = prepared_store
    with freeze_time(_NOW):
        # Write the older 2-hour-old value with an old event timestamp; the
        # frozen "now" makes it older than the TTL -> OUTSIDE_MAX_AGE.
        stale_ts = _NOW - _TTL - timedelta(hours=1)
        store.write_to_online_store(
            view.name,
            df=pd.DataFrame(
                [
                    {
                        _JOIN_KEY: 1,
                        _FEATURE_VALUE: 0.10 + 1,
                        _FEATURE_TRIPS: 11,
                        "event_timestamp": stale_ts,
                        "created_timestamp": stale_ts,
                    }
                ]
            ),
        )
        report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )

    mismatch = report.sample_mismatches[0]
    assert report.total_mismatch_cells == 1
    assert mismatch.reason == MismatchReason.WINDOW_MISMATCH.value
    assert mismatch.online_status == "OUTSIDE_MAX_AGE"


# ---------------------------------------------------------------------------
# Scenario 3: feature missing online
# ---------------------------------------------------------------------------


def test_missing_online_feature_counts_as_mismatch(tmp_path):
    store, _view = _make_store(tmp_path)
    try:
        with freeze_time(_NOW):
            # Deliberately do NOT materialize: every online read is NOT_FOUND.
            report = store.validate_feature_consistency(
                entity_df=_entity_dataframe(),
                features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
            )

        assert report.total_rows == 2
        assert report.total_compared_cells == 2
        assert report.total_mismatch_cells == 2
        assert report.mismatch_rate == 1.0
        assert report.reason_counts[MismatchReason.ONLINE_UNAVAILABLE.value] == 2
        for mismatch in report.sample_mismatches:
            assert mismatch.reason == MismatchReason.ONLINE_UNAVAILABLE.value
            assert mismatch.online_status == "NOT_FOUND"
            # The offline side is unaffected: the PIT join still returns data.
            assert mismatch.offline_value is not None
    finally:
        store.teardown()


def test_online_unavailable_is_never_counted_as_match(tmp_path):
    """Acceptance assertion: an unavailable online feature is not a pass."""
    store, _view = _make_store(tmp_path)
    try:
        with freeze_time(_NOW):
            report = store.validate_feature_consistency(
                entity_df=_entity_dataframe(),
                features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
            )
        matched = report.total_compared_cells - report.total_mismatch_cells
        assert matched == 0
        assert MismatchReason.VALUE_MATCH.value not in report.reason_counts
        # full_feature_names=False keys per-feature stats by bare feature name.
        assert (
            report.per_feature_stats[_FEATURE_VALUE]["reason_counts"][
                MismatchReason.ONLINE_UNAVAILABLE.value
            ]
            == 2
        )
    finally:
        store.teardown()


# ---------------------------------------------------------------------------
# Scenario 4: validation event time outside retention
# ---------------------------------------------------------------------------


def test_event_time_outside_retention_is_reported(tmp_path):
    store, _view = _make_store(tmp_path)
    try:
        with freeze_time(_NOW):
            store.materialize(
                feature_views=[_FV_NAME],
                start_date=_NOW - timedelta(days=10),
                end_date=_NOW,
            )
            # Event time beyond the TTL window: no in-window source row exists.
            stale_entity_df = pd.DataFrame(
                {
                    _JOIN_KEY: [1],
                    "event_timestamp": [_NOW - _TTL - timedelta(days=1)],
                }
            )
            report = store.validate_feature_consistency(
                entity_df=stale_entity_df,
                features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
            )
        assert report.total_compared_cells == 1
        assert report.total_mismatch_cells == 1
        mismatch = report.sample_mismatches[0]
        assert mismatch.reason == MismatchReason.OUT_OF_RETENTION.value
        assert "TTL" in mismatch.detail
    finally:
        store.teardown()


# ---------------------------------------------------------------------------
# Error handling: empty datasets and missing columns
# ---------------------------------------------------------------------------


def test_validation_aligns_reordered_offline_rows(tmp_path):
    """Offline joins may reorder output rows; validation must realign them.

    Regression test: comparing by positional index would attribute the
    value of one entity to another row. The internal correlation column
    must keep every value attached to its own input row, even when the two
    entities have different feature values and only one is materialized.
    """
    store, view = _make_store(tmp_path)
    try:
        with freeze_time(_NOW):
            # Put only driver 1 into the online store; driver 2 stays
            # NOT_FOUND. The two entities have distinct offline values, so
            # positional misalignment would cross-attribute driver 1's value
            # to driver 2 instead of reporting it unavailable.
            _overwrite_online_row(store, view, driver_id=1, value=1.20, trips=21)
            report = store.validate_feature_consistency(
                entity_df=_entity_dataframe(),
                features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
            )
        assert report.total_compared_cells == 2
        # Driver 1 matches; driver 2 is unavailable — never cross-attributed.
        assert report.total_mismatch_cells == 1
        mismatch = report.sample_mismatches[0]
        assert mismatch.entity_values[_JOIN_KEY] == 2
        assert mismatch.reason == MismatchReason.ONLINE_UNAVAILABLE.value
        assert mismatch.online_status == "NOT_FOUND"
    finally:
        store.teardown()


def test_empty_dataset_raises_instead_of_false_pass(prepared_store):
    store, _entity_df, _view = prepared_store
    with pytest.raises(EmptyConsistencyDatasetError):
        store.validate_feature_consistency(
            entity_df=pd.DataFrame(
                {_JOIN_KEY: pd.Series(dtype="int64"), "event_timestamp": []}
            ),
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )


def test_missing_event_timestamp_column_raises(prepared_store):
    store, _entity_df, _view = prepared_store
    with pytest.raises(ConsistencyTimestampColumnMissingError):
        store.validate_feature_consistency(
            entity_df=pd.DataFrame({_JOIN_KEY: [1]}),
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )


# ---------------------------------------------------------------------------
# Batching, merging, JSON round-trip, and tolerance semantics
# ---------------------------------------------------------------------------


def test_batching_produces_same_result_and_mergeable_reports(prepared_store):
    store, entity_df, _view = prepared_store
    features = [f"{_FV_NAME}:{_FEATURE_VALUE}"]
    with freeze_time(_NOW):
        full_report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=features,
            config=ConsistencyValidationConfig(online_read_batch_size=100),
        )
        shard_reports = [
            store.validate_feature_consistency(
                entity_df=entity_df.iloc[[i]],
                features=features,
                config=ConsistencyValidationConfig(online_read_batch_size=1),
            )
            for i in range(len(entity_df))
        ]
    merged = shard_reports[0]
    for shard in shard_reports[1:]:
        merged = merged.merge(shard)

    assert merged.total_rows == full_report.total_rows
    assert merged.total_compared_cells == full_report.total_compared_cells
    assert merged.total_mismatch_cells == full_report.total_mismatch_cells
    assert merged.mismatch_rate == full_report.mismatch_rate
    assert merged.shard_count == len(shard_reports)


def test_report_round_trips_through_json(prepared_store):
    store, entity_df, _view = prepared_store
    with freeze_time(_NOW):
        report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )
    restored = ConsistencyReport.from_dict(report.to_dict())
    assert restored.total_compared_cells == report.total_compared_cells
    assert restored.mismatch_rate == report.mismatch_rate


def test_float_tolerance_basis_is_explicit(prepared_store):
    store, entity_df, view = prepared_store
    with freeze_time(_NOW):
        # Offline value is 1.20; differ by 1e-7 which is inside the default
        # relative tolerance budget (1e-5 * 1.20).
        _overwrite_online_row(store, view, driver_id=1, value=1.20 + 1e-7, trips=21)
        within = store.validate_feature_consistency(
            entity_df=entity_df.iloc[[0]],
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
        )
        assert within.total_mismatch_cells == 0

        # A zero tolerance makes the same numeric difference a mismatch.
        outside = store.validate_feature_consistency(
            entity_df=entity_df.iloc[[0]],
            features=[f"{_FV_NAME}:{_FEATURE_VALUE}"],
            config=ConsistencyValidationConfig(float_rtol=0.0, float_atol=0.0),
        )
    assert outside.total_mismatch_cells == 1
    assert outside.sample_mismatches[0].reason == MismatchReason.SEMANTIC_MISMATCH.value


def test_type_mismatch_is_classified_by_classifier():
    # The classifier is exercised directly for incompatible Python types
    # (string vs number), which the local online store cannot produce on its
    # own but remote/serialized integrations can.
    from feast.consistency import _classify_cell

    config = ConsistencyValidationConfig()
    reason, _detail = _classify_cell(
        config=config,
        feature_ref="fv:string_vs_number",
        event_time=_NOW,
        ttl=_TTL,
        offline_value=1.0,
        online_value="1.0",
        offline_type="float",
        online_type="str",
        online_status="PRESENT",
    )
    assert reason == MismatchReason.TYPE_MISMATCH


def test_feature_service_refs_are_supported(prepared_store):
    from feast import FeatureService

    store, entity_df, view = prepared_store
    service = FeatureService(name="driver_fs", features=[view[["conv_rate"]]])
    store.apply([service])
    with freeze_time(_NOW):
        report = store.validate_feature_consistency(
            entity_df=entity_df,
            features=store.get_feature_service("driver_fs"),
        )
    assert report.total_compared_cells == 2
    assert report.total_mismatch_cells == 0
