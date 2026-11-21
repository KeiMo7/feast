"""Tests for opt-in feature-definition tracking.

Covers the acceptance scenarios end to end against a real local store backed by
a file registry and SQLite online store:

* change records are fingerprint based (no manual version numbers),
* a logic change marks prior partitions as "must recompute",
* a metadata-only change marks them as "can skip",
* missing metadata is reported as undetermined, never guessed,
* an interrupted recompute resumes from where it stopped,
* version-qualified reads return the pinned definition while the newer one
  stays active,
* tracking off leaves behavior and outputs byte-identical.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.definition_tracking import fingerprint as fp
from feast.definition_tracking.tracker import (
    DefinitionTracker,
    FingerprintUnavailable,
)
from feast.types import Float32, Int64

PROJECT = "tracking_project"


def _write_config(tmp_path, *, tracking: bool, versioning: bool = False, strict=False):
    tracking_block = ""
    if tracking:
        strict_line = (
            "    require_explicit_partition_provenance: true\n" if strict else ""
        )
        tracking_block = f"  definition_tracking:\n    enabled: true\n{strict_line}"
    versioning_line = (
        "  enable_online_feature_view_versioning: true\n" if versioning else ""
    )
    config_path = tmp_path / "feature_store.yaml"
    config_path.write_text(
        "project: {project}\n"
        "provider: local\n"
        "registry:\n"
        "  path: data/registry.db\n"
        "{tracking}{versioning}"
        "online_store:\n"
        "  type: sqlite\n"
        "  path: data/online.db\n"
        "entity_key_serialization_version: 3\n".format(
            project=PROJECT,
            tracking=tracking_block,
            versioning=versioning_line,
        )
    )
    return config_path


def _write_data(tmp_path, columns=("conv_rate",)):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    frame = {
        "driver_id": [1001, 1002],
        "event_timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)] * 2,
    }
    # Always include conv_rate, plus any extra columns the test declares.
    for index, column in enumerate(("conv_rate", *columns)):
        frame[column] = [0.1 + index, 0.9 + index]
    path = data_dir / "driver_stats.parquet"
    pd.DataFrame(frame).to_parquet(path)
    return path


def _entity():
    return Entity(name="driver_id", join_keys=["driver_id"])


def _feature_view(
    path,
    *,
    description: str = "",
    extra_fields=(),
    version: str = "latest",
    ttl_days: int = 2,
):
    return FeatureView(
        name="driver_stats",
        entities=[_entity()],
        ttl=timedelta(days=ttl_days),
        description=description,
        version=version,
        schema=[
            Field(name="driver_id", dtype=Int64),
            Field(name="conv_rate", dtype=Float32),
            *extra_fields,
        ],
        source=FileSource(
            name="driver_stats_source",
            path=str(path),
            timestamp_field="event_timestamp",
        ),
    )


def _store(tmp_path, repo_config_path):
    return FeatureStore(repo_path=str(tmp_path), fs_yaml_file=repo_config_path)


def _materialize(store, start_day, end_day):
    store.materialize(
        start_date=datetime(2024, 1, start_day, tzinfo=timezone.utc),
        end_date=datetime(2024, 1, end_day, tzinfo=timezone.utc),
    )


@pytest.fixture
def tracked_store(tmp_path):
    config_path = _write_config(tmp_path, tracking=True)
    # acc_rate is present in the data although not declared initially, so tests
    # can add it to the view's schema and still materialize the new definition.
    path = _write_data(tmp_path, columns=("acc_rate",))
    store = _store(tmp_path, config_path)
    store.apply([_entity(), _feature_view(path)])
    return store, path, config_path


# ----------------------------------------------------------------- fingerprints


class TestFingerprints:
    def _fv(self, **kwargs):
        path = kwargs.pop("path", "data/driver.parquet")
        return _feature_view(path, **kwargs).to_proto()

    def test_same_content_same_fingerprint(self):
        assert fp.definition_fingerprint(self._fv()) == fp.definition_fingerprint(
            self._fv()
        )

    def test_description_changes_definition_but_not_logic(self):
        base = self._fv()
        renamed = self._fv(description="documented")
        assert fp.definition_fingerprint(base) != fp.definition_fingerprint(renamed)
        assert fp.logic_fingerprint(base) == fp.logic_fingerprint(renamed)

    def test_source_change_changes_logic(self):
        base = self._fv()
        moved = self._fv(path="data/other.parquet")
        assert fp.logic_fingerprint(base) != fp.logic_fingerprint(moved)

    def test_ttl_change_changes_logic(self):
        # TTL bounds the point-in-time join window, so it changes values.
        assert fp.logic_fingerprint(self._fv()) != fp.logic_fingerprint(
            self._fv(ttl_days=5)
        )

    def test_feature_diff(self):
        base = self._fv()
        extended = self._fv(extra_fields=(Field(name="acc_rate", dtype=Float32),))
        changes = fp.diff_feature_sets(base, extended, logic_changed=True)
        assert ("acc_rate", 1, "", "6") in changes  # FEATURE_ADDED
        assert all(change[0] != "driver_id" for change in changes)


# ------------------------------------------------------------ change recording


class TestChangeRecording:
    def test_first_apply_records_baseline(self, tracked_store):
        store, _, _ = tracked_store
        changes = store.list_definition_changes("driver_stats")
        assert len(changes) == 1
        assert changes[0]["baseline_missing"] is True
        assert changes[0]["fingerprint"]

    def test_unchanged_reapply_records_nothing(self, tracked_store):
        store, path, _ = tracked_store
        store.apply([_entity(), _feature_view(path)])
        assert len(store.list_definition_changes("driver_stats")) == 1

    def test_metadata_change_is_recorded_without_logic_change(self, tracked_store):
        store, path, _ = tracked_store
        store.apply([_entity(), _feature_view(path, description="documented")])
        changes = store.list_definition_changes("driver_stats")
        assert len(changes) == 2
        assert changes[-1]["logic_changed"] is False

    def test_fingerprint_does_not_come_from_version_string(self, tracked_store):
        """Two applies of identical content at different version numbers share
        a fingerprint; the version string is never an input."""
        store, path, _ = tracked_store
        first = store.list_definition_changes("driver_stats")[0]["fingerprint"]
        store.apply([_entity(), _feature_view(path, description="x")])
        store.apply([_entity(), _feature_view(path)])
        changes = store.list_definition_changes("driver_stats")
        # Back to identical content as the first apply: same fingerprint reused.
        assert changes[-1]["fingerprint"] == first


# -------------------------------------------------------------- impact analysis


class TestImpactAnalysis:
    def test_new_feature_marks_prior_partitions_for_recompute(self, tracked_store):
        store, path, _ = tracked_store
        _materialize(store, 1, 2)
        _materialize(store, 2, 3)

        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )
        impact = store.analyze_definition_change("driver_stats")

        assert impact.logic_changed is True
        # Two distinct partitions -> two verdicts, all must recompute.
        assert len(impact.must_recompute) == 2
        assert all(
            "computation logic differs" in verdict.reason
            for verdict in impact.must_recompute
        )
        # The impact range starts at the earliest materialized partition.
        assert impact.affected_start is not None
        assert impact.affected_start.date().isoformat() == "2024-01-01"

    def test_impact_comes_from_dependency_and_partition_metadata(self, tracked_store):
        """The recompute verdict is derived from materialization-interval
        metadata + recorded provenance, not from any manual annotation."""
        store, path, _ = tracked_store
        _materialize(store, 1, 2)
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )
        impact = store.analyze_definition_change("driver_stats")

        # The single must-recompute partition matches the registry's recorded
        # materialization interval exactly.
        fv = store.get_feature_view("driver_stats")
        intervals = {(start, end) for start, end in fv.materialization_intervals}
        verdicts = {(item.start, item.end) for item in impact.must_recompute}
        assert verdicts.issubset(intervals)
        assert len(verdicts) >= 1

        # And the planner's reasoning cites the recorded provenance, not a tag.
        assert impact.must_recompute[0].fingerprint is not None

    def test_metadata_only_change_can_skip(self, tracked_store):
        store, path, _ = tracked_store
        _materialize(store, 1, 2)
        store.apply([_entity(), _feature_view(path, description="documented")])
        impact = store.analyze_definition_change("driver_stats")

        assert impact.must_recompute == []
        assert len(impact.can_skip) == 1
        assert "only metadata changed" in impact.can_skip[0].reason

    def test_recomputing_clears_the_partition(self, tracked_store):
        store, path, _ = tracked_store
        _materialize(store, 1, 2)
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )
        plan = store.create_recompute_plan("driver_stats")
        store.execute_recompute_plan(plan.plan_id)

        # After recomputing, a fresh analysis of the same change finds nothing
        # left to do for that partition.
        impact = store.analyze_definition_change("driver_stats")
        assert impact.must_recompute == []
        assert len(impact.can_skip) == 1

    def test_downstream_view_is_flagged(self, tmp_path):
        """A derived view that consumes the changed view is flagged too."""
        from feast.on_demand_feature_view import on_demand_feature_view

        config_path = _write_config(tmp_path, tracking=True)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)

        base = _feature_view(path)

        @on_demand_feature_view(
            sources=[base],
            schema=[Field(name="scaled", dtype=Float32)],
            mode="pandas",
        )
        def odfv(inputs):
            return pd.DataFrame({"scaled": inputs["conv_rate"].astype("float32") * 2})

        store.apply([_entity(), base, odfv])
        _materialize(store, 1, 2)

        # Add a feature to the base view -> its logic changes.
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
                odfv,
            ]
        )
        impact = store.analyze_definition_change("driver_stats")
        assert "odfv" in impact.downstream_views


# ------------------------------------------------------- missing metadata path


class TestMissingMetadata:
    def test_partitions_without_provenance_are_undetermined(self, tmp_path):
        path = _write_data(tmp_path)

        # Phase 1: tracking OFF, materialize a partition.
        off_config = _write_config(tmp_path, tracking=False)
        off_store = _store(tmp_path, off_config)
        off_store.apply([_entity(), _feature_view(path)])
        _materialize(off_store, 1, 2)
        # Clear any tracking state written by other phases of the same repo.
        off_store.registry.cached_registry_proto.definition_tracking.Clear()
        off_store.registry.commit()

        # Phase 2: tracking ON, change the definition.
        on_config = _write_config(tmp_path, tracking=True)
        store = _store(tmp_path, on_config)
        store.apply([_entity(), _feature_view(path, description="changed")])
        impact = store.analyze_definition_change("driver_stats")

        assert impact.impact_known is False
        assert impact.must_recompute == []
        assert impact.can_skip == []
        assert len(impact.undetermined) == 1
        assert "provenance is missing" in impact.undetermined[0].reason

    def test_strict_mode_raises_instead_of_reporting(self, tmp_path):
        path = _write_data(tmp_path)

        off_config = _write_config(tmp_path, tracking=False)
        off_store = _store(tmp_path, off_config)
        off_store.apply([_entity(), _feature_view(path)])
        _materialize(off_store, 1, 2)
        off_store.registry.cached_registry_proto.definition_tracking.Clear()
        off_store.registry.commit()

        strict_config = _write_config(tmp_path, tracking=True, strict=True)
        store = _store(tmp_path, strict_config)
        store.apply([_entity(), _feature_view(path, description="changed")])
        with pytest.raises(FingerprintUnavailable):
            store.analyze_definition_change("driver_stats")

    def test_first_tracked_apply_has_no_previous_baseline(self, tracked_store):
        store, _, _ = tracked_store
        changes = store.list_definition_changes("driver_stats")
        assert changes[0]["baseline_missing"] is True
        # With no materialized partitions, the first apply affects no data.
        impact = store.analyze_definition_change("driver_stats")
        assert impact.partitions == []
        assert impact.impact_known is True


# --------------------------------------------------------------- recompute plan


class TestRecomputePlan:
    def _prepare(self, store, path):
        _materialize(store, 1, 2)
        _materialize(store, 2, 3)
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )
        return store.create_recompute_plan("driver_stats")

    def test_plan_is_idempotent(self, tracked_store):
        store, path, _ = tracked_store
        plan = self._prepare(store, path)
        again = store.create_recompute_plan("driver_stats")
        assert plan.plan_id == again.plan_id

    def test_interrupted_recompute_resumes(self, tracked_store, monkeypatch):
        store, path, _ = tracked_store
        plan = self._prepare(store, path)

        original = store.materialize
        calls = {"count": 0}

        def flaky(*args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 2:
                raise RuntimeError("simulated interruption")
            return original(*args, **kwargs)

        monkeypatch.setattr(store, "materialize", flaky)
        first = store.execute_recompute_plan(plan.plan_id)
        assert first["completed"] == 1
        assert first["failed"] == 1

        progress = store.get_recompute_plan(plan.plan_id)["progress"]
        assert progress.get("recompute_completed") == 1
        assert progress.get("recompute_failed") == 1

        # Resume: only the failed partition is retried.
        monkeypatch.setattr(store, "materialize", original)
        calls["count"] = 0
        second = store.execute_recompute_plan(plan.plan_id)
        assert second["completed"] == 1
        assert second["failed"] == 0
        assert store.get_recompute_plan(plan.plan_id)["next_pending_partition"] is None

        # A third run is a no-op.
        third = store.execute_recompute_plan(plan.plan_id)
        assert third == {"processed": 0, "completed": 0, "failed": 0, "errors": []}


# ------------------------------------------------------------ two versions live


class TestVersionedCoexistence:
    def test_pinned_read_sees_old_definition(self, tmp_path):
        config_path = _write_config(tmp_path, tracking=True, versioning=True)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)

        store.apply([_entity(), _feature_view(path)])
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )

        versions = [
            item["version"] for item in store.list_feature_view_versions("driver_stats")
        ]
        assert versions == ["v0", "v1"]

        v0 = store.registry.get_feature_view_by_version("driver_stats", PROJECT, 0)
        active = store.get_feature_view("driver_stats")
        assert not any(feature.name == "acc_rate" for feature in v0.features)
        assert any(feature.name == "acc_rate" for feature in active.features)

        entity_df = pd.DataFrame(
            {
                "driver_id": [1001],
                "event_timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            }
        )
        pinned = store.get_historical_features(
            entity_df=entity_df, features=["driver_stats@v0:conv_rate"]
        ).to_df()
        assert "acc_rate" not in pinned.columns
        assert "conv_rate" in pinned.columns

    def test_pinning_with_modified_definition_raises(self, tmp_path):
        from feast.errors import FeatureViewPinConflict

        config_path = _write_config(tmp_path, tracking=True, versioning=True)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)
        store.apply([_entity(), _feature_view(path)])
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )

        conflicting = _feature_view(
            path,
            extra_fields=(
                Field(name="acc_rate", dtype=Float32),
                Field(name="brand_new", dtype=Float32),
            ),
            version="v0",
        )
        with pytest.raises(FeatureViewPinConflict):
            store.apply([_entity(), conflicting])


# ------------------------------------------------------- tracking-off invariance


class TestTrackingOff:
    def _outputs(self, tmp_path, tracking: bool):
        tmp_path.mkdir(parents=True, exist_ok=True)
        config_path = _write_config(tmp_path, tracking=tracking)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)
        store.apply([_entity(), _feature_view(path)])
        _materialize(store, 1, 3)

        online = store.get_online_features(
            features=["driver_stats:conv_rate"], entity_rows=[{"driver_id": 1001}]
        ).to_dict()
        entity_df = pd.DataFrame(
            {
                "driver_id": [1001],
                "event_timestamp": [datetime(2024, 1, 1, tzinfo=timezone.utc)],
            }
        )
        historical = (
            store.get_historical_features(
                entity_df=entity_df, features=["driver_stats:conv_rate"]
            )
            .to_df()
            .to_dict("records")
        )
        intervals = store.get_feature_view("driver_stats").materialization_intervals
        return online, historical, intervals

    def test_outputs_identical_with_tracking_on_and_off(self, tmp_path):
        off = self._outputs(tmp_path / "off", tracking=False)
        on = self._outputs(tmp_path / "on", tracking=True)
        assert on == off

    def test_no_tracking_state_written_when_disabled(self, tmp_path):
        config_path = _write_config(tmp_path, tracking=False)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)
        store.apply([_entity(), _feature_view(path)])
        _materialize(store, 1, 2)

        state = store.registry.get_definition_tracking_state()
        assert len(state.changes) == 0
        assert len(state.partitions) == 0
        assert store.definition_tracking_enabled is False

    def test_api_requires_opt_in(self, tmp_path):
        config_path = _write_config(tmp_path, tracking=False)
        path = _write_data(tmp_path)
        store = _store(tmp_path, config_path)
        store.apply([_entity(), _feature_view(path)])
        with pytest.raises(ValueError, match="definition tracking"):
            store.list_definition_changes("driver_stats")


# --------------------------------------------------------------- sql registry


class TestSqlRegistryTracking:
    def test_state_persists_per_project(self, tmp_path):
        from feast.infra.registry.sql import SqlRegistry, SqlRegistryConfig

        config = SqlRegistryConfig(
            path=f"sqlite:///{tmp_path}/registry.db",
            definition_tracking={"enabled": True},
        )
        registry = SqlRegistry(config, PROJECT, None)
        path = _write_data(tmp_path)
        registry.apply_feature_view(_feature_view(path), PROJECT)

        state = registry.get_definition_tracking_state()
        assert len(state.changes) == 1
        assert state.changes[0].feature_view_name == "driver_stats"

        # A fresh registry instance (new process) reads the same state.
        fresh = SqlRegistry(config, PROJECT, None)
        reloaded = fresh.get_definition_tracking_state(PROJECT)
        assert len(reloaded.changes) == 1
        assert reloaded.changes[0].fingerprint == state.changes[0].fingerprint


# ------------------------------------------------------------ tracker unit tests


class TestTrackerUnit:
    def _proto(self, **kwargs):
        return _feature_view("data/x.parquet", **kwargs).to_proto()

    def test_change_record_contains_per_feature_diff(self):
        from feast.protos.feast.core.DefinitionTracking_pb2 import (
            DefinitionTrackingState,
        )

        tracker = DefinitionTracker("p", DefinitionTrackingState())
        previous = self._proto()
        current = self._proto(extra_fields=(Field(name="acc_rate", dtype=Float32),))
        record = tracker.record_apply(
            feature_view_name="driver_stats",
            feature_view_type="feature_view",
            current_proto=current,
            previous_proto=previous,
            version_number=1,
            previous_version_number=0,
        )
        assert record is not None
        names = {change.feature_name for change in record.feature_changes}
        assert "acc_rate" in names
        assert record.sequence_number == 1

    def test_identical_content_records_nothing(self):
        from feast.protos.feast.core.DefinitionTracking_pb2 import (
            DefinitionTrackingState,
        )

        tracker = DefinitionTracker("p", DefinitionTrackingState())
        proto = self._proto()
        assert (
            tracker.record_apply(
                feature_view_name="driver_stats",
                feature_view_type="feature_view",
                current_proto=proto,
                previous_proto=copy.deepcopy(proto),
                version_number=0,
            )
            is None
        )


# ------------------------------------------------------------------- CLI surface


class TestCli:
    def _invoke(self, tmp_path, args):
        # Invoke via the CLI group with the --chdir global option so the
        # group-level callback populates the context exactly as production does.
        from click.testing import CliRunner

        from feast.cli.cli import cli

        return CliRunner().invoke(cli, ["--chdir", str(tmp_path), *args])

    def test_list_and_analyze(self, tracked_store, tmp_path):
        store, path, config_path = tracked_store
        _materialize(store, 1, 2)
        store.apply(
            [
                _entity(),
                _feature_view(
                    path, extra_fields=(Field(name="acc_rate", dtype=Float32),)
                ),
            ]
        )

        result = self._invoke(tmp_path, ["definition-changes", "list"])
        assert result.exit_code == 0, result.output
        assert "driver_stats" in result.output
        assert "acc_rate:added" in result.output

        result = self._invoke(
            tmp_path, ["definition-changes", "analyze", "driver_stats"]
        )
        assert result.exit_code == 0, result.output
        assert "must recompute: 1" in result.output

        result = self._invoke(
            tmp_path, ["definition-changes", "analyze", "driver_stats", "--json"]
        )
        assert result.exit_code == 0, result.output
        import json

        payload = json.loads(result.output)
        assert payload["logic_changed"] is True
        assert len(payload["must_recompute"]) == 1


# ----------------------------------------------------- end-to-end SQL registry


class TestSqlRegistryEndToEnd:
    def test_change_and_impact_through_sql_registry(self, tmp_path):
        from feast.infra.registry.sql import SqlRegistry, SqlRegistryConfig

        config = SqlRegistryConfig(
            path=f"sqlite:///{tmp_path}/registry.db",
            definition_tracking={"enabled": True},
        )
        registry = SqlRegistry(config, PROJECT, None)
        path = _write_data(tmp_path, columns=("acc_rate",))

        registry.apply_feature_view(_feature_view(path), PROJECT)
        start = datetime(2024, 1, 1, tzinfo=timezone.utc)
        end = datetime(2024, 1, 2, tzinfo=timezone.utc)
        registry.apply_materialization(
            registry.get_feature_view("driver_stats", PROJECT), PROJECT, start, end
        )
        first_fp = (
            registry.get_definition_tracking_state(PROJECT).partitions[0].fingerprint
        )

        registry.apply_feature_view(
            _feature_view(path, extra_fields=(Field(name="acc_rate", dtype=Float32),)),
            PROJECT,
        )
        state = registry.get_definition_tracking_state(PROJECT)
        assert len(state.changes) == 2
        assert len(state.partitions) == 1
        assert state.partitions[0].fingerprint == first_fp
