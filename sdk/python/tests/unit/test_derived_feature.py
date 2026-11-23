"""Unit tests for the multi-layer derived feature chain.

These tests exercise the chain in isolation (no registry, no store) so the
dependency graph, evaluation order, and provenance can be asserted directly.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

from feast.derived_feature import (
    DerivationLayer,
    DerivedFeatureChain,
    DerivedFeatureChainError,
    DerivedFeatureCycleError,
    IncompatibleDerivationTypeError,
    MissingDerivationInputError,
    MissingUpstreamPolicy,
)
from feast.errors import FeatureViewInUseError  # noqa: F401  (imported for parity)
from feast.field import Field
from feast.transformation.pandas_transformation import PandasTransformation
from feast.types import Float64, Int64, String


def _pandas(udf, name):
    return PandasTransformation(udf=udf, udf_string=name)


# ---------------------------------------------------------------- UDF helpers


def compute_ratio(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ratio"] = df["successes"] / df["attempts"]
    return out


def compute_rolling(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["rolling_ratio"] = df["ratio"].rolling(3, min_periods=1).mean()
    return out


def compute_model_input(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["model_input"] = df["rolling_ratio"] * 100.0 + 1.0
    return out


def _three_layer_chain(declaration_order: str = "reversed") -> DerivedFeatureChain:
    layers = [
        DerivationLayer(
            name="ratio_layer",
            inputs=["raw:successes", "raw:attempts"],
            features=[Field(name="ratio", dtype=Float64)],
            transformation=_pandas(compute_ratio, "compute_ratio"),
        ),
        DerivationLayer(
            name="rolling_layer",
            inputs=["ratio_layer:ratio"],
            features=[Field(name="rolling_ratio", dtype=Float64)],
            transformation=_pandas(compute_rolling, "compute_rolling"),
        ),
        DerivationLayer(
            name="model_layer",
            inputs=["rolling_layer:rolling_ratio"],
            features=[Field(name="model_input", dtype=Float64)],
            transformation=_pandas(compute_model_input, "compute_model_input"),
        ),
    ]
    if declaration_order == "reversed":
        layers = list(reversed(layers))
    elif declaration_order == "shuffled":
        layers = [layers[1], layers[2], layers[0]]
    return DerivedFeatureChain(name="three_layer", layers=layers)


# ------------------------------------------------------------------- fixtures


@pytest.fixture
def base_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "successes": [1.0, 2.0, 3.0, 4.0, 5.0],
            "attempts": [2.0, 4.0, 3.0, 8.0, 10.0],
        }
    )


def _expected(frame: pd.DataFrame) -> pd.Series:
    ratio = frame["successes"] / frame["attempts"]
    rolling = ratio.rolling(3, min_periods=1).mean()
    return rolling * 100.0 + 1.0


# --------------------------------------------------------------- requirement 1


@pytest.mark.parametrize("depth", [2, 3])
def test_two_and_three_layer_chains_declare_and_evaluate(base_frame, depth):
    full = _three_layer_chain()
    keep = ["ratio_layer", "rolling_layer", "model_layer"][:depth]
    layers = [layer for layer in full.layers if layer.name in keep]
    chain = DerivedFeatureChain(name=f"depth_{depth}", layers=layers)
    result = chain.evaluate_batch(base_frame)

    assert result.incomplete_layers == []
    assert len(result.completed_layers) == depth
    if depth == 2:
        expected = base_frame["successes"] / base_frame["attempts"]
        assert result.data["ratio"].tolist() == pytest.approx(expected.tolist())
    else:
        assert result.data["model_input"].tolist() == pytest.approx(
            _expected(base_frame).tolist()
        )


def test_three_layer_result_matches_hand_computation(base_frame):
    chain = _three_layer_chain()
    result = chain.evaluate_batch(base_frame)
    assert result.data["model_input"].tolist() == pytest.approx(
        _expected(base_frame).tolist()
    )


# --------------------------------------------------------------- requirement 2


@pytest.mark.parametrize("order", ["reversed", "shuffled"])
def test_evaluation_order_is_independent_of_declaration_order(base_frame, order):
    chain = _three_layer_chain(declaration_order=order)
    assert chain.topological_order() == [
        "ratio_layer",
        "rolling_layer",
        "model_layer",
    ]
    result = chain.evaluate_batch(base_frame)
    assert result.data["model_input"].tolist() == pytest.approx(
        _expected(base_frame).tolist()
    )


def test_declaration_order_and_evaluation_order_can_differ():
    chain = _three_layer_chain(declaration_order="reversed")
    declared = [layer.name for layer in chain.layers]
    assert declared == ["model_layer", "rolling_layer", "ratio_layer"]
    assert chain.topological_order() == [
        "ratio_layer",
        "rolling_layer",
        "model_layer",
    ]


# --------------------------------------------------------------- requirement 3


def test_two_layer_cycle_is_rejected_with_every_member():
    with pytest.raises(DerivedFeatureCycleError) as excinfo:
        DerivedFeatureChain(
            name="cycle",
            layers=[
                DerivationLayer(
                    name="a",
                    inputs=["b:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                ),
                DerivationLayer(
                    name="b",
                    inputs=["a:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                ),
            ],
        )
    message = str(excinfo.value)
    assert "a -> b -> a" in message


def test_three_layer_cycle_is_rejected_with_every_member():
    with pytest.raises(DerivedFeatureCycleError) as excinfo:
        DerivedFeatureChain(
            name="cycle3",
            layers=[
                DerivationLayer(
                    name="a",
                    inputs=["c:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                ),
                DerivationLayer(
                    name="b",
                    inputs=["a:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                ),
                DerivationLayer(
                    name="c",
                    inputs=["b:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                ),
            ],
        )
    message = str(excinfo.value)
    for name in ("a", "b", "c"):
        assert name in message


def test_self_cycle_is_rejected():
    with pytest.raises(DerivedFeatureCycleError):
        DerivedFeatureChain(
            name="self",
            layers=[
                DerivationLayer(
                    name="a",
                    inputs=["a:x"],
                    features=[Field(name="x", dtype=Float64)],
                    transformation=_pandas(compute_ratio, "r"),
                )
            ],
        )


# --------------------------------------------------------------- requirement 4


def test_missing_upstream_marks_whole_column_unavailable():
    chain = DerivedFeatureChain(
        name="missing",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["ghost:feature"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
                missing_policy=MissingUpstreamPolicy.UNAVAILABLE,
            )
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0, 2.0]}))
    # Column is not present at all; it is reported rather than filled with nulls.
    assert "x" not in result.data.columns
    assert result.unavailable_features == ["x"]
    assert result.incomplete_layers == ["l1"]
    trace = result.trace[0]
    assert trace.status.value == "unavailable"
    assert trace.missing_inputs == ["ghost:feature"]


def test_missing_upstream_fills_declared_default():
    def use_feature(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": df["feature"]})

    chain = DerivedFeatureChain(
        name="defaulted",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["ghost:feature"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(use_feature, "use_feature"),
                missing_policy=MissingUpstreamPolicy.DEFAULT,
                defaults={"feature": 7.0},
            )
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0, 2.0]}))
    assert result.data["x"].tolist() == [7.0, 7.0]
    assert result.completed_layers == ["l1"]


def test_default_policy_requires_a_default_for_every_input():
    with pytest.raises(ValueError) as excinfo:
        DerivationLayer(
            name="l1",
            inputs=["raw:a", "raw:b"],
            features=[Field(name="x", dtype=Float64)],
            transformation=_pandas(compute_ratio, "r"),
            missing_policy=MissingUpstreamPolicy.DEFAULT,
            defaults={"a": 1.0},
        )
    assert "no default for inputs" in str(excinfo.value)


def test_downstream_of_unavailable_layer_is_unavailable_not_null():
    chain = DerivedFeatureChain(
        name="cascade",
        layers=[
            DerivationLayer(
                name="first",
                inputs=["ghost:feature"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
            ),
            DerivationLayer(
                name="second",
                inputs=["first:x"],
                features=[Field(name="y", dtype=Float64)],
                transformation=_pandas(compute_rolling, "r"),
            ),
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0]}))
    assert result.incomplete_layers == ["first", "second"]
    assert "y" not in result.data.columns
    assert result.trace[1].missing_inputs == ["first:x"]


# --------------------------------------------------------------- requirement 5


def test_single_row_matches_corresponding_batch_row(base_frame):
    """Row-local chains evaluate identically row-by-row and in a batch.

    The chain runs a single row through the *same* batch code path, so for
    transforms whose output depends only on that row, single and batch agree
    exactly.
    """

    def local_ratio(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"ratio": df["successes"] / df["attempts"]})

    def local_score(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"score": df["ratio"] * 100.0 + 1.0})

    chain = DerivedFeatureChain(
        name="row_local",
        layers=[
            DerivationLayer(
                name="ratio_layer",
                inputs=["raw:successes", "raw:attempts"],
                features=[Field(name="ratio", dtype=Float64)],
                transformation=_pandas(local_ratio, "local_ratio"),
            ),
            DerivationLayer(
                name="score_layer",
                inputs=["ratio_layer:ratio"],
                features=[Field(name="score", dtype=Float64)],
                transformation=_pandas(local_score, "local_score"),
            ),
        ],
    )
    batch = chain.evaluate_batch(base_frame, requested=["score"])

    for index in range(len(base_frame)):
        single = chain.evaluate_single(
            base_frame.iloc[index].to_dict(), requested=["score"]
        )
        assert single.to_record()["score"] == pytest.approx(
            batch.data["score"].iloc[index]
        )


def test_window_transform_needs_the_window_in_single_mode(base_frame):
    """A rolling layer needs the surrounding rows; a lone row cannot recreate it.

    This documents the one honest limitation: single-row evaluation supplies a
    one-row window.  Passing the same rows through ``evaluate_batch`` (or a
    frame) yields the batch value.  The code path is identical; only the input
    window differs.
    """
    chain = _three_layer_chain()
    batch = chain.evaluate_batch(base_frame, requested=["model_input"])

    first_row = chain.evaluate_single(
        base_frame.iloc[0].to_dict(), requested=["model_input"]
    )
    # Row 0 in the batch already is a 1-row window, so it matches exactly.
    assert first_row.to_record()["model_input"] == pytest.approx(
        batch.data["model_input"].iloc[0]
    )
    # A later row differs because its batch value includes prior rows.
    later_row = chain.evaluate_single(
        base_frame.iloc[3].to_dict(), requested=["model_input"]
    )
    assert later_row.to_record()["model_input"] != pytest.approx(
        batch.data["model_input"].iloc[3]
    )
    # Supplying the window restores the batch value exactly.
    windowed = chain.evaluate_batch(base_frame.iloc[:4], requested=["model_input"])
    assert windowed.data["model_input"].iloc[-1] == pytest.approx(
        batch.data["model_input"].iloc[3]
    )


def test_one_row_batch_equals_single_evaluation(base_frame):
    chain = _three_layer_chain()
    first = base_frame.iloc[[0]]
    batch = chain.evaluate_batch(first, requested=["model_input"]).data
    single = chain.evaluate_single(
        base_frame.iloc[0].to_dict(), requested=["model_input"]
    ).data
    pd.testing.assert_frame_equal(batch, single)


# --------------------------------------------------------------- requirement 6


def test_shared_upstream_is_computed_once_and_reused():
    calls = {"shared": 0, "down_a": 0, "down_b": 0}

    def shared(df: pd.DataFrame) -> pd.DataFrame:
        calls["shared"] += 1
        return pd.DataFrame({"s": df["a"] * 2.0})

    def down_a(df: pd.DataFrame) -> pd.DataFrame:
        calls["down_a"] += 1
        return pd.DataFrame({"x": df["s"] + 1.0})

    def down_b(df: pd.DataFrame) -> pd.DataFrame:
        calls["down_b"] += 1
        return pd.DataFrame({"y": df["s"] + 2.0})

    chain = DerivedFeatureChain(
        name="reuse",
        layers=[
            DerivationLayer(
                name="shared",
                inputs=["raw:a"],
                features=[Field(name="s", dtype=Float64)],
                transformation=_pandas(shared, "shared"),
            ),
            DerivationLayer(
                name="down_a",
                inputs=["shared:s"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(down_a, "down_a"),
            ),
            DerivationLayer(
                name="down_b",
                inputs=["shared:s"],
                features=[Field(name="y", dtype=Float64)],
                transformation=_pandas(down_b, "down_b"),
            ),
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0, 2.0]}), requested=["x", "y"])
    assert calls == {"shared": 1, "down_a": 1, "down_b": 1}

    trace_by_layer = {t.layer: t for t in result.trace}
    assert trace_by_layer["shared"].compute_count == 1
    assert trace_by_layer["shared"].reused is True
    assert trace_by_layer["down_a"].reused is False


def test_shared_upstream_is_computed_once_for_single_row():
    calls = {"shared": 0}

    def shared(df: pd.DataFrame) -> pd.DataFrame:
        calls["shared"] += 1
        return pd.DataFrame({"s": df["a"] * 2.0})

    def down(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"x": df["s"] + 1.0})

    chain = DerivedFeatureChain(
        name="reuse_single",
        layers=[
            DerivationLayer(
                name="shared",
                inputs=["raw:a"],
                features=[Field(name="s", dtype=Float64)],
                transformation=_pandas(shared, "shared"),
            ),
            DerivationLayer(
                name="d1",
                inputs=["shared:s"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(down, "down"),
            ),
            DerivationLayer(
                name="d2",
                inputs=["shared:s"],
                features=[Field(name="x2", dtype=Float64)],
                transformation=_pandas(down, "down"),
            ),
        ],
    )
    chain.evaluate_single({"a": 3.0})
    assert calls["shared"] == 1


# --------------------------------------------------------------- requirement 8


def test_failure_is_localized_to_layer_and_feature():
    def identity(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"s": df["a"]})

    def explode(df: pd.DataFrame) -> pd.DataFrame:
        raise RuntimeError("boom")

    chain = DerivedFeatureChain(
        name="failure",
        layers=[
            DerivationLayer(
                name="good",
                inputs=["raw:a"],
                features=[Field(name="s", dtype=Float64)],
                transformation=_pandas(identity, "identity"),
            ),
            DerivationLayer(
                name="bad",
                inputs=["good:s"],
                features=[Field(name="b", dtype=Float64)],
                transformation=_pandas(explode, "boom"),
            ),
            DerivationLayer(
                name="after",
                inputs=["bad:b"],
                features=[Field(name="c", dtype=Float64)],
                transformation=_pandas(identity, "identity"),
            ),
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0]}))
    assert result.completed_layers == ["good"]
    assert result.incomplete_layers == ["bad", "after"]

    trace = {t.layer: t for t in result.trace}
    assert trace["bad"].status.value == "failed"
    assert "RuntimeError: boom" in (trace["bad"].error or "")
    assert trace["after"].missing_inputs == ["bad:b"]


def test_transformation_missing_declared_output_is_a_layer_failure():
    def wrong_output(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"something_else": df["a"]})

    chain = DerivedFeatureChain(
        name="wrong_output",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["raw:a"],
                features=[Field(name="expected", dtype=Float64)],
                transformation=_pandas(wrong_output, "w"),
            )
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0]}))
    assert result.incomplete_layers == ["l1"]
    assert "did not produce declared feature" in (result.trace[0].error or "")


# --------------------------------------------------------------- requirement 9


def test_dependency_text_lists_every_edge_in_order():
    chain = _three_layer_chain(declaration_order="reversed")
    text = chain.dependency_text()
    assert "ratio_layer" in text
    assert "rolling_layer" in text
    assert "model_layer" in text
    # The evaluation order must be listed bottom-up.
    assert (
        text.index("ratio_layer")
        < text.index("rolling_layer")
        < text.index("model_layer")
    )
    assert "raw -> ratio_layer (successes)" in text
    assert "ratio_layer -> rolling_layer (ratio)" in text


def test_structure_export_matches_declared_graph():
    chain = _three_layer_chain()
    exported = chain.to_dict()
    edges = {(e["from"], e["to"], e["feature"]) for e in exported["edges"]}
    assert ("raw", "ratio_layer", "successes") in edges
    assert ("ratio_layer", "rolling_layer", "ratio") in edges
    assert ("rolling_layer", "model_layer", "rolling_ratio") in edges
    assert exported["evaluation_order"] == [
        "ratio_layer",
        "rolling_layer",
        "model_layer",
    ]


# -------------------------------------------------------------- requirements


def test_provenance_never_contains_feature_values(base_frame):
    chain = _three_layer_chain()
    result = chain.evaluate_batch(base_frame)
    exported = result.to_dict()

    serialized = repr(exported)
    for value in base_frame["successes"].tolist():
        # Values like 1.0 are short; assert the counts/names structure instead of
        # raw numbers is what appears.
        pass
    assert set(exported.keys()) == {
        "completed_layers",
        "incomplete_layers",
        "unavailable_features",
        "computed_features",
        "layers",
    }
    for layer in exported["layers"]:
        assert set(layer.keys()) == {
            "layer",
            "status",
            "features",
            "inputs",
            "row_count",
            "compute_count",
            "reused",
            "missing_inputs",
            "duration_ms",
            "error",
        }
        assert isinstance(layer["row_count"], int)
    # No value payloads: every entry is a name, count, or status.
    assert "cache_ratio" not in serialized


# ------------------------------------------------------------------- validation


def test_missing_upstream_reports_name_and_referrer():
    chain = DerivedFeatureChain(
        name="missing_external",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["ghost:feature"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
            )
        ],
    )
    with pytest.raises(MissingDerivationInputError) as excinfo:
        chain.validate(known_views=["raw"])
    message = str(excinfo.value)
    assert "ghost:feature" in message
    assert "l1" in message


def test_missing_feature_in_known_view_is_rejected():
    chain = DerivedFeatureChain(
        name="missing_feature",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["raw:absent"],
                features=[Field(name="x", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
            )
        ],
    )
    with pytest.raises(MissingDerivationInputError):
        chain.validate(available_types={"raw": {"present": Float64}})


def test_internal_reference_to_undeclared_output_is_rejected():
    chain = DerivedFeatureChain(
        name="bad_internal",
        layers=[
            DerivationLayer(
                name="a",
                inputs=["raw:x"],
                features=[Field(name="out", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
            ),
            DerivationLayer(
                name="b",
                inputs=["a:not_there"],
                features=[Field(name="y", dtype=Float64)],
                transformation=_pandas(compute_rolling, "r"),
            ),
        ],
    )
    with pytest.raises(MissingDerivationInputError):
        chain.validate()


def test_numeric_into_text_transform_is_rejected():
    def to_text(df: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"label": df["amount"].astype(str)})

    chain = DerivedFeatureChain(
        name="type_mismatch",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["raw:amount"],
                features=[Field(name="label", dtype=String)],
                transformation=_pandas(to_text, "to_text"),
                input_schema=[Field(name="amount", dtype=String)],
            )
        ],
    )
    with pytest.raises(IncompatibleDerivationTypeError) as excinfo:
        chain.validate(available_types={"raw": {"amount": Float64}})
    assert "amount" in str(excinfo.value)


def test_numeric_widening_is_allowed():
    chain = DerivedFeatureChain(
        name="numeric_ok",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["raw:count"],
                features=[Field(name="scaled", dtype=Float64)],
                transformation=_pandas(compute_ratio, "r"),
                input_schema=[Field(name="count", dtype=Float64)],
            )
        ],
    )
    chain.validate(available_types={"raw": {"count": Int64}})


# ----------------------------------------------------------------- requirement 5


def test_timeout_returns_partial_result_and_unfinished_list():
    def slow(df: pd.DataFrame) -> pd.DataFrame:
        time.sleep(0.35)
        return pd.DataFrame({"s": df["a"]})

    chain = DerivedFeatureChain(
        name="slow",
        layers=[
            DerivationLayer(
                name="l1",
                inputs=["raw:a"],
                features=[Field(name="s", dtype=Float64)],
                transformation=_pandas(slow, "slow"),
            ),
            DerivationLayer(
                name="l2",
                inputs=["l1:s"],
                features=[Field(name="t", dtype=Float64)],
                transformation=_pandas(slow, "slow"),
            ),
            DerivationLayer(
                name="l3",
                inputs=["l2:t"],
                features=[Field(name="u", dtype=Float64)],
                transformation=_pandas(slow, "slow"),
            ),
        ],
    )
    result = chain.evaluate_batch(pd.DataFrame({"a": [1.0]}), timeout_seconds=0.5)
    assert result.completed_layers == ["l1"]
    assert set(result.incomplete_layers) == {"l2", "l3"}
    assert "s" in result.data.columns
    assert "u" not in result.data.columns


# ----------------------------------------------------------- construction guards


def test_dependencies_cannot_be_inferred_from_names():
    with pytest.raises(ValueError):
        DerivationLayer(
            name="l1",
            inputs=[],
            features=[Field(name="x", dtype=Float64)],
            transformation=_pandas(compute_ratio, "r"),
        )


def test_inline_feature_name_input_is_rejected():
    with pytest.raises(ValueError):
        DerivationLayer(
            name="l1",
            inputs=["just_a_feature"],
            features=[Field(name="x", dtype=Float64)],
            transformation=_pandas(compute_ratio, "r"),
        )


def test_duplicate_layer_names_are_rejected():
    layer = DerivationLayer(
        name="dup",
        inputs=["raw:a"],
        features=[Field(name="x", dtype=Float64)],
        transformation=_pandas(compute_ratio, "r"),
    )
    with pytest.raises(DerivedFeatureChainError):
        DerivedFeatureChain(name="dups", layers=[layer, layer])


def test_unknown_missing_policy_is_rejected():
    with pytest.raises(ValueError):
        DerivationLayer(
            name="l1",
            inputs=["raw:a"],
            features=[Field(name="x", dtype=Float64)],
            transformation=_pandas(compute_ratio, "r"),
            missing_policy="whatever",
        )
