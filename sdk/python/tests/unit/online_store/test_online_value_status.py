"""Tests for per-row, per-feature online value status tracking.

Covers the opt-in ``online_status`` capability of ``get_online_features``:

* the five missing-value classifications (entity not found, feature never
  materialized, materialized-but-null, unsupported type, unregistered view),
* partial batch failure (other rows still returned, failed batch listed),
* entity deduplication (one store read, results fanned out to all positions),
* per-reason metric aggregation,
* explicit default filling with the original status preserved,
* store-unavailable and registry-version-mismatch reporting,
* and the disabled path staying byte-identical to the old behavior.
"""

import pytest

from feast import FeatureStore
from feast.infra.online_stores.sqlite import SqliteOnlineStore
from feast.online_status import ValueStatus
from feast.protos.feast.types.EntityKey_pb2 import EntityKey as EntityKeyProto
from feast.protos.feast.types.Value_pb2 import Value as ValueProto
from feast.repo_config import OnlineStatusConfig
from feast.utils import _utc_now
from tests.utils.cli_repo_creator import CliRunner, get_example_repo


def _enable_status(store: FeatureStore, **kwargs) -> None:
    store.config.online_status_config = OnlineStatusConfig(enabled=True, **kwargs)


def _write_driver_row(store, driver_id: int, features: dict) -> None:
    provider = store._get_provider()
    driver_locations_fv = store.get_feature_view(name="driver_locations")
    provider.online_write_batch(
        config=store.config,
        table=driver_locations_fv,
        data=[
            (
                EntityKeyProto(
                    join_keys=["driver_id"],
                    entity_values=[ValueProto(int64_val=driver_id)],
                ),
                features,
                _utc_now(),
                _utc_now(),
            )
        ],
        progress=None,
    )


@pytest.fixture
def store():
    runner = CliRunner()
    with runner.local_repo(
        get_example_repo("example_feature_repo_1.py"),
        "file",
    ) as store:
        yield store


def test_five_missing_scenarios(store) -> None:
    """Each of the five missing-value shapes gets its own status."""
    # driver 1: only lat materialized -> lon never materialized
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=0.5)})
    # driver 2: lat materialized as null
    _write_driver_row(
        store, 2, {"lat": ValueProto(), "lon": ValueProto(string_val="2.0")}
    )
    # driver 3: lat stored with a string for a numeric feature
    _write_driver_row(
        store,
        3,
        {"lat": ValueProto(string_val="oops"), "lon": ValueProto(string_val="3.0")},
    )
    # driver 4: nothing written -> entity not in store

    _enable_status(store)
    response = store.get_online_features(
        features=[
            "driver_locations:lat",
            "driver_locations:lon",
            "ghost_view:foo",
        ],
        entity_rows=[{"driver_id": i} for i in (1, 2, 3, 4)],
    )
    report = response.status_report
    assert report is not None

    lat = report.status_of("lat")
    lon = report.status_of("lon")
    foo = report.status_of("foo")

    # row 0 (driver 1): present + never materialized
    assert lat[0] == ValueStatus.PRESENT
    assert lon[0] == ValueStatus.FEATURE_NOT_MATERIALIZED
    # row 1 (driver 2): materialized-but-null + present
    assert lat[1] == ValueStatus.NULL_VALUE
    assert lon[1] == ValueStatus.PRESENT
    # row 2 (driver 3): type contract violation
    assert lat[2] == ValueStatus.UNSUPPORTED_TYPE
    assert lon[2] == ValueStatus.PRESENT
    # row 3 (driver 4): entity absent from the store
    assert lat[3] == ValueStatus.ENTITY_NOT_FOUND
    assert lon[3] == ValueStatus.ENTITY_NOT_FOUND
    # unregistered view: flagged on every row, did not abort the request
    assert all(s == ValueStatus.FEATURE_VIEW_NOT_REGISTERED for s in foo)

    # The existing response structure is unchanged: same columns, same values.
    result = response.to_dict()
    assert result["driver_id"] == [1, 2, 3, 4]
    assert result["lat"][0] == 0.5
    assert result["lat"][1] is None
    assert result["lat"][2] == "oops"  # value passed through untouched
    assert result["lat"][3] is None
    assert result["foo"] == [None, None, None, None]

    # Every result carries a value timestamp and a source version.
    block = next(b for b in report.blocks if b.feature_view == "driver_locations")
    assert block.version is not None
    assert block.event_timestamps[0] is not None  # driver 1 has a stored row
    assert block.event_timestamps[3] is None  # driver 4 was not found
    ghost = next(b for b in report.blocks if b.feature_view == "ghost_view")
    assert ghost.version is None


def test_all_views_unregistered(store) -> None:
    """A request whose views are all unregistered still returns rows."""
    _enable_status(store)
    response = store.get_online_features(
        features=["ghost_view:foo", "ghost_view:bar"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 2}],
    )
    report = response.status_report
    assert report is not None
    result = response.to_dict()
    assert result["driver_id"] == [1, 2]
    assert result["foo"] == [None, None]
    assert result["bar"] == [None, None]
    assert report.status_of("foo") == [ValueStatus.FEATURE_VIEW_NOT_REGISTERED] * 2


def test_partial_batch_failure(store, monkeypatch) -> None:
    """One failing read batch does not take down the rest of the request."""
    for i in (1, 2, 3):
        _write_driver_row(store, i, {"lat": ValueProto(float_val=float(i))})

    _enable_status(store, read_batch_size=1, max_batch_retries=1)

    original = SqliteOnlineStore.online_read

    def flaky(self, config, table, entity_keys, requested_features=None):
        for ek in entity_keys:
            for v in ek.entity_values:
                if v.int64_val == 2:
                    raise RuntimeError("simulated store failure")
        return original(self, config, table, entity_keys, requested_features)

    monkeypatch.setattr(SqliteOnlineStore, "online_read", flaky)

    response = store.get_online_features(
        features=["driver_locations:lat"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 2}, {"driver_id": 3}],
    )
    report = response.status_report
    assert report is not None

    lat = report.status_of("lat")
    by_driver = dict(zip(response.to_dict()["driver_id"], lat))
    assert by_driver[1] == ValueStatus.PRESENT
    assert by_driver[3] == ValueStatus.PRESENT
    # The failed batch is reported as unavailable, not as missing.
    assert by_driver[2] == ValueStatus.STORE_UNAVAILABLE

    # The failed batch is listed explicitly (no entity identifiers).
    assert len(report.failed_batches) == 1
    failed = report.failed_batches[0]
    assert failed.feature_view == "driver_locations"
    assert failed.row_count == 1
    assert failed.attempts == 2  # 1 initial + 1 retry
    assert failed.error_type == "RuntimeError"

    # The other rows still carry their values.
    result = response.to_dict()
    values = dict(zip(result["driver_id"], result["lat"]))
    assert values[1] == 1.0
    assert values[3] == 3.0
    assert values[2] is None


def test_store_unavailable_does_not_mark_missing(store, monkeypatch) -> None:
    """A fully unavailable store yields store_unavailable, never not_found."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=1.0)})
    _enable_status(store, max_batch_retries=2)

    def down(self, config, table, entity_keys, requested_features=None):
        raise ConnectionError("store is down")

    monkeypatch.setattr(SqliteOnlineStore, "online_read", down)

    response = store.get_online_features(
        features=["driver_locations:lat"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 2}],
    )
    report = response.status_report
    assert report is not None
    lat = report.status_of("lat")
    assert lat == [ValueStatus.STORE_UNAVAILABLE] * 2
    assert ValueStatus.ENTITY_NOT_FOUND not in lat
    assert len(report.failed_batches) == 1
    assert report.failed_batches[0].attempts == 3  # 1 initial + 2 retries
    assert response.to_dict()["lat"] == [None, None]


def test_failed_batch_retried_then_succeeds(store, monkeypatch) -> None:
    """A transient failure is retried within the cap and succeeds."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=1.0)})
    _enable_status(store, max_batch_retries=2)

    original = SqliteOnlineStore.online_read
    calls = {"n": 0}

    def flaky_once(self, config, table, entity_keys, requested_features=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("transient")
        return original(self, config, table, entity_keys, requested_features)

    monkeypatch.setattr(SqliteOnlineStore, "online_read", flaky_once)

    response = store.get_online_features(
        features=["driver_locations:lat"], entity_rows=[{"driver_id": 1}]
    )
    report = response.status_report
    assert report is not None
    assert report.status_of("lat") == [ValueStatus.PRESENT]
    assert report.failed_batches == []
    assert calls["n"] == 2
    assert response.to_dict()["lat"] == [1.0]


def test_duplicate_entities_read_once(store, monkeypatch) -> None:
    """Repeated entities in one request are fetched once and fanned out."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=1.5)})
    _write_driver_row(store, 2, {"lat": ValueProto(float_val=2.5)})
    _enable_status(store)

    original = SqliteOnlineStore.online_read
    calls = []

    def spy(self, config, table, entity_keys, requested_features=None):
        calls.append(len(entity_keys))
        return original(self, config, table, entity_keys, requested_features)

    monkeypatch.setattr(SqliteOnlineStore, "online_read", spy)

    response = store.get_online_features(
        features=["driver_locations:lat"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 1}, {"driver_id": 2}],
    )
    # One read for the feature view, over the 2 unique entities only.
    assert calls == [2]

    result = response.to_dict()
    assert result["lat"] == [1.5, 1.5, 2.5]
    lat_status = response.status_report.status_of("lat")
    assert lat_status == [ValueStatus.PRESENT] * 3
    # Both positions of the duplicated entity got identical results.
    assert result["lat"][0] == result["lat"][1]


def test_metrics_aggregated_by_reason(store, monkeypatch) -> None:
    """Missing-reason counts land in the existing status metric, by reason."""
    import feast.metrics as metrics

    original_flags = metrics._config
    metrics._config = metrics._MetricsFlags(
        enabled=True,
        resource=True,
        request=True,
        online_features=True,
        push=True,
        materialization=True,
        freshness=True,
        offline_features=True,
        audit_logging=False,
    )
    try:
        _write_driver_row(store, 1, {"lat": ValueProto(float_val=0.5)})
        _enable_status(store)

        def _count(status: str) -> float:
            return metrics.online_features_status_total.labels(
                feature_view="driver_locations", status=status
            )._value.get()

        before = {
            s: _count(s)
            for s in (
                "present",
                "entity_not_found",
                "feature_not_materialized",
            )
        }

        store.get_online_features(
            features=["driver_locations:lat", "driver_locations:lon"],
            entity_rows=[{"driver_id": 1}, {"driver_id": 999}],
        )

        # driver 1: lat present, lon never materialized.
        # driver 999: both entity_not_found.
        assert _count("present") == before["present"] + 1
        assert (
            _count("feature_not_materialized") == before["feature_not_materialized"] + 1
        )
        assert _count("entity_not_found") == before["entity_not_found"] + 2
    finally:
        metrics._config = original_flags


def test_fill_defaults_explicit_and_status_preserved(store) -> None:
    """Defaults are filled only when explicitly enabled; status is kept."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=0.5)})
    _enable_status(store, fill_defaults=True, defaults={"lat": -1.5})

    response = store.get_online_features(
        features=["driver_locations:lat", "driver_locations:lon"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 999}],
    )
    report = response.status_report
    assert report is not None
    assert report.defaults_filled is True

    result = response.to_dict()
    # Configured override for lat; type-zero ("") for lon. Both missing
    # shapes (never materialized, entity not found) are fillable.
    assert result["lat"] == [0.5, -1.5]
    assert result["lon"] == ["", ""]
    # The original status survives the fill.
    assert report.status_of("lat")[1] == ValueStatus.ENTITY_NOT_FOUND
    assert report.status_of("lon") == [  # row 0: never materialized
        ValueStatus.FEATURE_NOT_MATERIALIZED,
        ValueStatus.ENTITY_NOT_FOUND,
    ]


def test_no_fill_by_default(store) -> None:
    """Without fill_defaults, missing values stay None."""
    _enable_status(store)
    response = store.get_online_features(
        features=["driver_locations:lat"],
        entity_rows=[{"driver_id": 999}],
    )
    assert response.to_dict()["lat"] == [None]
    assert response.status_report.defaults_filled is False
    assert response.status_report.status_of("lat") == [ValueStatus.ENTITY_NOT_FOUND]


def test_registry_version_mismatch_reported(store, monkeypatch) -> None:
    """A registry snapshot change mid-request is reported, not hidden."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=1.0)})
    _enable_status(store)

    original = SqliteOnlineStore.online_read

    def mutating(self, config, table, entity_keys, requested_features=None):
        # Simulate the registry being refreshed underneath the request.
        proto = store.registry.cached_registry_proto
        proto.last_updated.seconds += 1
        return original(self, config, table, entity_keys, requested_features)

    monkeypatch.setattr(SqliteOnlineStore, "online_read", mutating)

    response = store.get_online_features(
        features=["driver_locations:lat"], entity_rows=[{"driver_id": 1}]
    )
    report = response.status_report
    assert report is not None
    assert report.registry_version_mismatch is True


def test_disabled_returns_legacy_response(store) -> None:
    """With the capability off, the response is exactly as before."""
    _write_driver_row(store, 1, {"lat": ValueProto(float_val=0.5)})

    response = store.get_online_features(
        features=["driver_locations:lat", "driver_locations:lon"],
        entity_rows=[{"driver_id": 1}, {"driver_id": 2}],
    )
    assert response.status_report is None
    result = response.to_dict()
    assert result == {
        "driver_id": [1, 2],
        "lat": [0.5, None],
        "lon": [None, None],
    }
