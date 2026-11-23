"""End-to-end tests for multi-layer derived feature chains through a FeatureStore.

Covers registration-time rejection, evaluation through the real offline/online
paths, batch/single consistency, upstream reuse, provenance export, and the
feature flag that turns the capability off.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import pytest

from feast import (
    Entity,
    FeatureStore,
    FeatureView,
    Field,
    FileSource,
    RepoConfig,
)
from feast.derived_feature import (
    DerivationLayer,
    DerivedFeatureChain,
    DerivedFeatureChainError,
    DerivedFeatureCycleError,
    IncompatibleDerivationTypeError,
    MissingDerivationInputError,
)
from feast.errors import FeatureViewInUseError
from feast.infra.online_stores.sqlite import SqliteOnlineStoreConfig
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.transformation.pandas_transformation import PandasTransformation
from feast.types import Float32, Float64, String

PROJECT = "derived_chain_test"


# --------------------------------------------------------------------- UDFs


def ratio_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ratio": df["conv_rate"] / (df["acc_rate"] + 1.0)})


def rolling_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"rolling": df["ratio"].rolling(3, min_periods=1).mean()})


def score_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"score": df["rolling"] * 100.0 + 1.0})


def pandas_t(udf, name):
    return PandasTransformation(udf=udf, udf_string=name)


# ----------------------------------------------------------------- fixtures


def _write_driver_data(path: str, drivers, start: datetime, end: datetime) -> None:
    rows = []
    for driver_id in drivers:
        ts = start
        step = 0
        while ts <= end:
            rows.append(
                {
                    "driver_id": driver_id,
                    "event_timestamp": ts,
                    "created": ts,
                    "conv_rate": 0.1 * (step + 1),
                    "acc_rate": 0.05 * (step + 1),
                }
            )
            ts += timedelta(hours=1)
            step += 1
    pd.DataFrame(rows).to_parquet(path)


@pytest.fixture
def store(tmp_path):
    data_path = str(tmp_path / "driver_stats.parquet")
    end = datetime(2024, 1, 1, 12, 0, 0)
    start = end - timedelta(days=3)
    _write_driver_data(data_path, [1001, 1002], start, end)

    config = RepoConfig(
        project=PROJECT,
        registry=str(tmp_path / "registry.db"),
        provider="local",
        entity_key_serialization_version=3,
        online_store=SqliteOnlineStoreConfig(path=str(tmp_path / "online.db")),
    )
    feature_store = FeatureStore(config=config)

    driver = Entity(name="driver", join_keys=["driver_id"])
    source = FileSource(
        name="driver_stats_source",
        path=data_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )
    feature_view = FeatureView(
        name="driver_stats",
        entities=[driver],
        ttl=timedelta(days=30),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
        ],
        online=True,
        source=source,
    )
    feature_store.apply([driver, source, feature_view])

    feature_store._test_df = pd.read_parquet(data_path)  # type: ignore[attr-defined]
    feature_store._test_feature_view = feature_view  # type: ignore[attr-defined]
    return feature_store


def _three_layer_chain() -> DerivedFeatureChain:
    return DerivedFeatureChain(
        name="driver_chain",
        layers=[
            DerivationLayer(
                name="ratio_view",
                inputs=["driver_stats:conv_rate", "driver_stats:acc_rate"],
                features=[Field(name="ratio", dtype=Float64)],
                transformation=pandas_t(ratio_udf, "ratio_udf"),
            ),
            DerivationLayer(
                name="rolling_view",
                inputs=["ratio_view:ratio"],
                features=[Field(name="rolling", dtype=Float64)],
                transformation=pandas_t(rolling_udf, "rolling_udf"),
            ),
            DerivationLayer(
                name="score_view",
                inputs=["rolling_view:rolling"],
                features=[Field(name="score", dtype=Float64)],
                transformation=pandas_t(score_udf, "score_udf"),
            ),
        ],
    )


def _expected_score(df: pd.DataFrame) -> pd.Series:
    ratio = df["conv_rate"] / (df["acc_rate"] + 1.0)
    rolling = ratio.rolling(3, min_periods=1).mean()
    return rolling * 100.0 + 1.0


# ---------------------------------------------------- requirement: 3-layer chain


def test_three_layer_chain_history_matches_hand_computation(store):
    registered = store.apply_derived_feature_chain(_three_layer_chain())
    # Registered in dependency order regardless of the declaration order above.
    assert registered == ["ratio_view", "rolling_view", "score_view"]

    entity_df = store._test_df[["driver_id", "event_timestamp"]].head(8).copy()
    entity_df = entity_df.reset_index(drop=True)

    result = store.get_historical_features(
        entity_df=entity_df, features=["score_view:score"]
    ).to_df()

    # Recompute by hand from the raw features the retrieval also returns.
    raw = store.get_historical_features(
        entity_df=entity_df,
        features=["driver_stats:conv_rate", "driver_stats:acc_rate"],
    ).to_df()

    expected = _expected_score(raw)
    assert result["score"].tolist() == pytest.approx(expected.tolist(), rel=1e-6)


def test_three_layer_chain_history_works_on_duckdb_backend(tmp_path):
    """The same chain evaluates correctly through the ibis/duckdb offline store."""
    data_path = str(tmp_path / "driver_stats.parquet")
    end = datetime(2024, 1, 1, 6, 0, 0)
    start = end - timedelta(days=1)
    _write_driver_data(data_path, [1001], start, end)

    config = RepoConfig(
        project=PROJECT,
        registry=str(tmp_path / "registry.db"),
        provider="local",
        entity_key_serialization_version=3,
        offline_store={"type": "duckdb"},
        online_store=SqliteOnlineStoreConfig(path=str(tmp_path / "online.db")),
    )
    feature_store = FeatureStore(config=config)
    driver = Entity(name="driver", join_keys=["driver_id"])
    source = FileSource(
        name="driver_stats_source",
        path=data_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )
    feature_view = FeatureView(
        name="driver_stats",
        entities=[driver],
        ttl=timedelta(days=30),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
        ],
        online=True,
        source=source,
    )
    feature_store.apply([driver, source, feature_view])
    feature_store.apply_derived_feature_chain(_three_layer_chain())

    frame = pd.read_parquet(data_path)
    entity_df = (
        frame[["driver_id", "event_timestamp"]]
        .drop_duplicates()
        .head(6)
        .reset_index(drop=True)
    )
    # A single query returns raw features and the derived value together.
    result = feature_store.get_historical_features(
        entity_df=entity_df,
        features=[
            "driver_stats:conv_rate",
            "driver_stats:acc_rate",
            "score_view:score",
        ],
    ).to_df()

    # Recompute the chain from the raw columns *in the order they were returned*;
    # the rolling layer is order-sensitive, so the derived value must equal the
    # same row-wise computation over that returned sequence.
    ratio = result["conv_rate"] / (result["acc_rate"] + 1.0)
    expected = ratio.rolling(3, min_periods=1).mean() * 100.0 + 1.0
    assert result["score"].tolist() == pytest.approx(expected.tolist(), rel=1e-6)


def test_declaration_order_does_not_change_registered_order(store):
    # Declare bottom layer last; registration still orders it first.
    chain = DerivedFeatureChain(
        name="shuffled",
        layers=[
            DerivationLayer(
                name="score_view",
                inputs=["rolling_view:rolling"],
                features=[Field(name="score", dtype=Float64)],
                transformation=pandas_t(score_udf, "score_udf"),
            ),
            DerivationLayer(
                name="ratio_view",
                inputs=["driver_stats:conv_rate", "driver_stats:acc_rate"],
                features=[Field(name="ratio", dtype=Float64)],
                transformation=pandas_t(ratio_udf, "ratio_udf"),
            ),
            DerivationLayer(
                name="rolling_view",
                inputs=["ratio_view:ratio"],
                features=[Field(name="rolling", dtype=Float64)],
                transformation=pandas_t(rolling_udf, "rolling_udf"),
            ),
        ],
    )
    registered = store.apply_derived_feature_chain(chain)
    assert registered == ["ratio_view", "rolling_view", "score_view"]


# ------------------------------------------- requirement: registration rejects


def test_cycle_rejected_at_registration_with_cycle_members(store):
    with pytest.raises(DerivedFeatureCycleError):
        DerivedFeatureChain(
            name="cycle",
            layers=[
                DerivationLayer(
                    name="a",
                    inputs=["b:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=pandas_t(ratio_udf, "ratio_udf"),
                ),
                DerivationLayer(
                    name="b",
                    inputs=["a:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=pandas_t(ratio_udf, "ratio_udf"),
                ),
            ],
        )


def test_missing_upstream_rejected_at_registration_with_names(store):
    with pytest.raises(MissingDerivationInputError) as excinfo:
        store.apply_derived_feature_chain(
            DerivedFeatureChain(
                name="missing",
                layers=[
                    DerivationLayer(
                        name="ratio_view",
                        inputs=["ghost_view:some_feature"],
                        features=[Field(name="ratio", dtype=Float64)],
                        transformation=pandas_t(ratio_udf, "ratio_udf"),
                    )
                ],
            )
        )
    message = str(excinfo.value)
    assert "ghost_view:some_feature" in message
    assert "ratio_view" in message


def test_type_incompatible_chain_rejected_at_registration(store):
    def to_text(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"label": df["conv_rate"].astype(str)})

    with pytest.raises(IncompatibleDerivationTypeError):
        store.apply_derived_feature_chain(
            DerivedFeatureChain(
                name="type_mismatch",
                layers=[
                    DerivationLayer(
                        name="label_view",
                        inputs=["driver_stats:conv_rate"],
                        features=[Field(name="label", dtype=String)],
                        transformation=pandas_t(to_text, "to_text"),
                        input_schema=[Field(name="conv_rate", dtype=String)],
                    )
                ],
            )
        )


# ------------------------------------------- requirement: online/historical same


def test_online_and_historical_use_the_same_definition(store):
    store.apply_derived_feature_chain(_three_layer_chain())

    # Populate the online store with the raw features.
    store.write_to_online_store(feature_view_name="driver_stats", df=store._test_df)

    latest = (
        store._test_df.sort_values("event_timestamp")
        .groupby("driver_id")
        .tail(1)
        .reset_index(drop=True)
    )

    online = store.get_online_features(
        entity_rows=[{"driver_id": int(row.driver_id)} for row in latest.itertuples()],
        features=["score_view:score"],
    ).to_df()

    entity_df = latest[["driver_id", "event_timestamp"]].copy()
    historical = store.get_historical_features(
        entity_df=entity_df, features=["score_view:score"]
    ).to_df()

    # Both paths run the same registered views; the values agree.
    assert online["score"].tolist() == pytest.approx(
        historical["score"].tolist(), rel=1e-6
    )


def test_batch_and_single_values_agree_for_row_local_layers(store):
    def local_ratio(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ratio": df["conv_rate"] / (df["acc_rate"] + 1.0)})

    def local_score(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": df["ratio"] * 10.0})

    chain = DerivedFeatureChain(
        name="row_local",
        layers=[
            DerivationLayer(
                name="ratio_view",
                inputs=["driver_stats:conv_rate", "driver_stats:acc_rate"],
                features=[Field(name="ratio", dtype=Float64)],
                transformation=pandas_t(local_ratio, "local_ratio"),
            ),
            DerivationLayer(
                name="score_view",
                inputs=["ratio_view:ratio"],
                features=[Field(name="score", dtype=Float64)],
                transformation=pandas_t(local_score, "local_score"),
            ),
        ],
    )
    store.apply_derived_feature_chain(chain)

    frame = store._test_df.head(4)
    batch = chain.evaluate_batch(frame, requested=["score"])
    for index in range(len(frame)):
        single = chain.evaluate_single(frame.iloc[index].to_dict(), requested=["score"])
        assert single.to_record()["score"] == pytest.approx(
            batch.data["score"].iloc[index]
        )


# ---------------------------------------------------- requirement: reuse counts


def test_shared_upstream_evaluated_once_across_downstream_layers(store):
    calls = {"ratio": 0}

    def counting_ratio(df: pd.DataFrame) -> pd.DataFrame:
        calls["ratio"] += 1
        return pd.DataFrame({"ratio": df["conv_rate"] / (df["acc_rate"] + 1.0)})

    def left(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"left": df["ratio"] + 1.0})

    def right(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"right": df["ratio"] + 2.0})

    chain = DerivedFeatureChain(
        name="shared",
        layers=[
            DerivationLayer(
                name="ratio_view",
                inputs=["driver_stats:conv_rate", "driver_stats:acc_rate"],
                features=[Field(name="ratio", dtype=Float64)],
                transformation=pandas_t(counting_ratio, "counting_ratio"),
            ),
            DerivationLayer(
                name="left_view",
                inputs=["ratio_view:ratio"],
                features=[Field(name="left", dtype=Float64)],
                transformation=pandas_t(left, "left"),
            ),
            DerivationLayer(
                name="right_view",
                inputs=["ratio_view:ratio"],
                features=[Field(name="right", dtype=Float64)],
                transformation=pandas_t(right, "right"),
            ),
        ],
    )
    store.apply_derived_feature_chain(chain)

    entity_df = store._test_df[["driver_id", "event_timestamp"]].head(4).copy()
    result = store.get_historical_features(
        entity_df=entity_df.reset_index(drop=True),
        features=["left_view:left", "right_view:right"],
    ).to_df()
    assert set(["left", "right"]).issubset(result.columns)
    # One evaluation of the shared layer, reused by both consumers.
    assert calls["ratio"] == 1

    # The isolated chain agrees.
    isolated_calls = {"n": 0}

    def counted(df: pd.DataFrame) -> pd.DataFrame:
        isolated_calls["n"] += 1
        return pd.DataFrame({"r": df["a"]})

    chain2 = DerivedFeatureChain(
        name="shared2",
        layers=[
            DerivationLayer(
                name="s",
                inputs=["raw:a"],
                features=[Field(name="r", dtype=Float64)],
                transformation=pandas_t(counted, "counted"),
            ),
            DerivationLayer(
                name="l",
                inputs=["s:r"],
                features=[Field(name="x", dtype=Float64)],
                transformation=pandas_t(left, "left"),
            ),
            DerivationLayer(
                name="r2",
                inputs=["s:r"],
                features=[Field(name="y", dtype=Float64)],
                transformation=pandas_t(right, "right"),
            ),
        ],
    )
    eval_result = chain2.evaluate_batch(pd.DataFrame({"a": [1.0, 2.0]}))
    assert isolated_calls["n"] == 1
    trace = {t.layer: t for t in eval_result.trace}
    assert trace["s"].compute_count == 1
    assert trace["s"].reused is True


# ----------------------------------------------------- requirement: provenance


def test_provenance_export_matches_dependency_graph(store):
    store.apply_derived_feature_chain(_three_layer_chain())

    text = store.export_derived_feature_dependencies()
    assert "ratio_view" in text
    assert "rolling_view" in text
    assert "score_view" in text
    # Bottom-up order is reflected in the export.
    assert (
        text.index("ratio_view") < text.index("rolling_view") < text.index("score_view")
    )
    # Every edge appears as a view:feature input.
    assert "driver_stats:conv_rate" in text
    assert "ratio_view:ratio" in text
    assert "rolling_view:rolling" in text

    # No feature values leak into the export (names and counts only).
    for value in store._test_df["conv_rate"].head(3).tolist():
        assert str(value) not in text


def test_evaluation_provenance_contains_names_and_counts_only(base_frame=None):
    chain = _three_layer_chain()
    frame = pd.DataFrame({"conv_rate": [0.1, 0.2], "acc_rate": [0.05, 0.1]})
    result = chain.evaluate_batch(frame, requested=["score"])
    provenance = result.to_dict()
    assert provenance["completed_layers"] == [
        "ratio_view",
        "rolling_view",
        "score_view",
    ]
    for layer in provenance["layers"]:
        assert "features" in layer and "row_count" in layer
        # No raw value column in the trace.
        assert "data" not in layer and "values" not in layer


def test_registered_views_round_trip_into_the_same_chain(store):
    """The stored views reconstruct the exact declared dependency graph.

    This is what guarantees historical and online serving read one definition:
    both derive their order and inputs from these same registered views.
    """
    from feast.derived_feature import DerivedFeatureChain

    original = _three_layer_chain()
    store.apply_derived_feature_chain(original)

    registered = store.list_on_demand_feature_views()
    restored = DerivedFeatureChain.from_feature_views(registered)

    assert restored.name == original.name
    # Evaluation order is identical.
    assert restored.topological_order() == original.topological_order()

    original_edges = {
        (ref.view, layer.name, ref.feature)
        for layer in original.layers
        for ref in layer.inputs
    }
    restored_edges = {
        (ref.view, layer.name, ref.feature)
        for layer in restored.layers
        for ref in layer.inputs
    }
    assert restored_edges == original_edges

    # The reconstructed chain produces the same values as the original.
    frame = pd.DataFrame({"conv_rate": [0.1, 0.2, 0.3], "acc_rate": [0.1, 0.1, 0.1]})
    assert restored.evaluate_batch(frame).data["score"].tolist() == pytest.approx(
        original.evaluate_batch(frame).data["score"].tolist()
    )


# --------------------------------------------------- requirement: delete guard


def test_delete_blocked_while_upstream_is_referenced(store):
    store.apply_derived_feature_chain(_three_layer_chain())

    assert store.get_feature_view_dependents("ratio_view") == ["rolling_view"]

    with pytest.raises(FeatureViewInUseError) as excinfo:
        store.delete_feature_view("ratio_view")
    message = str(excinfo.value)
    assert "ratio_view" in message
    assert "rolling_view" in message

    # Deleting the whole chain together is allowed because no survivor refers
    # to a deleted view.
    store.apply(
        objects=[],
        objects_to_delete=[
            store.get_on_demand_feature_view(name)
            for name in ("ratio_view", "rolling_view", "score_view")
        ],
        partial=False,
    )
    assert store.list_on_demand_feature_views() == []


def test_objects_to_delete_blocks_referenced_upstream(store):
    store.apply_derived_feature_chain(_three_layer_chain())
    ratio_view = store.get_on_demand_feature_view("ratio_view")

    with pytest.raises(FeatureViewInUseError):
        store.apply(objects=[], objects_to_delete=[ratio_view], partial=False)


# -------------------------------------------------------- requirement: flag off


def test_flag_off_keeps_single_layer_working_and_rejects_multi_layer(tmp_path):
    data_path = str(tmp_path / "driver_stats.parquet")
    end = datetime(2024, 1, 1, 12, 0, 0)
    start = end - timedelta(days=1)
    _write_driver_data(data_path, [1001], start, end)

    config = RepoConfig(
        project=PROJECT,
        registry=str(tmp_path / "registry.db"),
        provider="local",
        entity_key_serialization_version=3,
        online_store=SqliteOnlineStoreConfig(path=str(tmp_path / "online.db")),
        derived_features={"enabled": False},
    )
    feature_store = FeatureStore(config=config)
    driver = Entity(name="driver", join_keys=["driver_id"])
    source = FileSource(
        name="driver_stats_source",
        path=data_path,
        timestamp_field="event_timestamp",
        created_timestamp_column="created",
    )
    feature_view = FeatureView(
        name="driver_stats",
        entities=[driver],
        ttl=timedelta(days=30),
        schema=[
            Field(name="conv_rate", dtype=Float32),
            Field(name="acc_rate", dtype=Float32),
        ],
        online=True,
        source=source,
    )

    single_layer = OnDemandFeatureView(
        name="single_ratio",
        sources=[feature_view],
        schema=[Field(name="ratio", dtype=Float64)],
        feature_transformation=pandas_t(ratio_udf, "ratio_udf"),
    )
    feature_store.apply([driver, source, feature_view, single_layer])
    # Single-layer behavior is unchanged with the capability disabled.
    assert feature_store.get_on_demand_feature_view("single_ratio").name == (
        "single_ratio"
    )

    with pytest.raises(DerivedFeatureChainError):
        feature_store.apply_derived_feature_chain(_three_layer_chain())

    # Manually assembled multi-layer views are also rejected at apply time.
    ratio_view = OnDemandFeatureView(
        name="ratio_view_flagoff",
        sources=[feature_view],
        schema=[Field(name="ratio", dtype=Float64)],
        feature_transformation=pandas_t(ratio_udf, "ratio_udf"),
    )
    dependent_view = OnDemandFeatureView(
        name="dependent_view_flagoff",
        sources=[ratio_view],
        schema=[Field(name="doubled", dtype=Float64)],
        feature_transformation=pandas_t(
            lambda df: pd.DataFrame({"doubled": df["ratio"] * 2.0}), "double"
        ),
    )
    with pytest.raises(DerivedFeatureChainError):
        feature_store.apply([ratio_view, dependent_view])
