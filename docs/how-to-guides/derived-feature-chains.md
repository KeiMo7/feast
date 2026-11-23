# Multi-layer derived features

Feast on-demand feature views (ODFVs) compute derived features at request time.
A **derived feature chain** lets you declare several ODFVs as an explicit,
multi-layer pipeline: a first layer computes a value from raw features, a later
layer computes a rolling statistic from that value, and a final layer assembles
the model input. Each layer is an ordinary ODFV; the chain adds a single,
validated dependency definition on top.

## Why a chain instead of hand-wired ODFVs

Wiring layers by hand means the dependency graph, the evaluation order, and any
provenance record are maintained separately and drift apart. A chain makes the
declaration the single source of truth:

- **Dependencies come only from declarations.** A layer reads exactly the
  `view:feature` inputs it lists. Nothing is inferred from naming conventions or
  from the order layers happen to be written in.
- **Evaluation order is derived from the graph.** Layers are ordered so every
  upstream runs first. Declaring the layers in any order produces the same
  result.
- **Registration and serving use the same definition.** `apply` validates the
  chain with the same object that evaluates it, so a chain cannot pass
  registration and then behave differently at serving time.
- **Failures are localized.** When an upstream is missing or a layer fails, the
  result names the exact layer and feature instead of quietly returning a column
  of nulls.

## Declaring a chain

```python
import pandas as pd

from feast import DerivationLayer, DerivedFeatureChain, Field, FeatureStore
from feast.transformation.pandas_transformation import PandasTransformation
from feast.types import Float64


def ratio_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"ratio": df["successes"] / df["attempts"]})


def rolling_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"rolling_ratio": df["ratio"].rolling(3, min_periods=1).mean()})


def model_input_udf(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"model_input": df["rolling_ratio"] * 100.0 + 1.0})


chain = DerivedFeatureChain(
    name="conversion_chain",
    layers=[
        DerivationLayer(
            name="ratio_layer",
            inputs=["raw_stats:successes", "raw_stats:attempts"],
            features=[Field(name="ratio", dtype=Float64)],
            transformation=PandasTransformation(udf=ratio_udf, udf_string="ratio_udf"),
        ),
        DerivationLayer(
            name="rolling_layer",
            inputs=["ratio_layer:ratio"],
            features=[Field(name="rolling_ratio", dtype=Float64)],
            transformation=PandasTransformation(
                udf=rolling_udf, udf_string="rolling_udf"
            ),
        ),
        DerivationLayer(
            name="model_layer",
            inputs=["rolling_layer:rolling_ratio"],
            features=[Field(name="model_input", dtype=Float64)],
            transformation=PandasTransformation(
                udf=model_input_udf, udf_string="model_input_udf"
            ),
        ),
    ],
)

store = FeatureStore(repo_path=".")
store.apply_derived_feature_chain(chain)
```

`apply_derived_feature_chain` validates the chain, then registers one
`OnDemandFeatureView` per layer and returns their names in evaluation order.

## Missing upstreams: choose explicitly

If a declared input is unavailable at evaluation time, a layer must say what to
do. There is no implicit fallback, because silently producing nulls is the
failure mode this feature removes.

```python
DerivationLayer(
    name="rolling_layer",
    inputs=["ratio_layer:ratio"],
    features=[Field(name="rolling_ratio", dtype=Float64)],
    transformation=...,
    missing_policy="unavailable",   # default: mark the column unavailable
)

DerivationLayer(
    name="rolling_layer",
    inputs=["ratio_layer:ratio"],
    features=[Field(name="rolling_ratio", dtype=Float64)],
    transformation=...,
    missing_policy="default",       # explicit: fill from `defaults`
    defaults={"ratio": 0.0},
)
```

With `"unavailable"`, the layer's outputs are not produced and are reported as
unavailable; downstream layers that consume them are reported in turn. With
`"default"`, every declared input must have a default.

## Querying the chain

Chains are queried through the registered ODFVs exactly like any other derived
features:

```python
store.get_historical_features(
    entity_df=entity_df,
    features=["model_layer:model_input"],
).to_df()

store.get_online_features(
    entity_rows=[{"driver_id": 1001}],
    features=["model_layer:model_input"],
).to_df()
```

Intermediate layers are evaluated automatically, in dependency order, and are
dropped from the response unless you request them.

## Registration-time validation

`apply_derived_feature_chain` (and `apply` for manually assembled ODFVs) reject:

- **Cycles**, naming every layer on the cycle.
- **Missing upstreams**, naming the missing `view:feature` and the layers that
  reference it.
- **Type-incompatible inputs**, such as feeding a numeric feature into a layer
  that declared a text input.

## Inspecting the graph

Export the dependency graph for review; it contains view and feature names only,
never feature values:

```bash
feast on-demand-feature-views dependencies
```

or, from Python:

```python
text = store.export_derived_feature_dependencies()
```

Individual layers also expose `dependencies()` / `dependents()`:

```python
store.get_feature_view_dependencies("rolling_layer")   # ["ratio_layer"]
store.get_feature_view_dependents("ratio_layer")       # ["rolling_layer"]
```

Deleting a layer that another registered view still consumes is refused, and the
referring views are listed.

## Evaluation provenance

When you evaluate a chain directly, the result carries a provenance record:

```python
result = chain.evaluate_batch(frame, requested=["model_input"])
provenance = result.to_dict()
# {
#   "completed_layers": ["ratio_layer", "rolling_layer", "model_layer"],
#   "incomplete_layers": [],
#   "layers": [
#     {"layer": "ratio_layer", "status": "ok", "features": ["ratio"],
#      "inputs": ["raw_stats:successes", "raw_stats:attempts"],
#      "row_count": 5, "compute_count": 1, "reused": False, ...},
#     ...
#   ],
# }
```

The record contains feature names, layer statuses, and row counts. It never
contains feature values.

A shared upstream that several layers consume is computed once and reused; the
trace reports it with `compute_count == 1` and `reused == True`.

If an evaluation exceeds a `timeout_seconds` budget, the result is partial: it
contains the layers that completed plus the list of unfinished layers, instead of
failing the whole batch.

## Turning the capability off

Multi-layer chains are enabled by default. To keep exactly the previous
single-layer behavior, disable them in `feature_store.yaml`:

```yaml
derived_features:
  enabled: false
```

With the capability disabled, single-layer on-demand feature views work
unchanged, and registering a multi-layer chain raises an error.
