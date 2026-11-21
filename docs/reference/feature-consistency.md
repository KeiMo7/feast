# Offline/Online Feature Consistency Validation

When a model scores well in offline evaluation but poorly online, a common
root cause is **training/serving skew**: the offline pipeline and the online
serving path do not produce the same feature value for the same entity at the
same point in time. Offline values are batch-computed historical values
(point-in-time join over the source), while online values are the latest
served rows materialized into the online store; their computation windows and
data freshness can diverge.

Feast provides an additive consistency validation entry point that compares
the two paths for the same entities and event times, quantifies the mismatch
rate, and attributes every mismatch to a reason category. It does **not**
change feature computation or any existing retrieval interface.

## When to use it

- After materialization, to confirm the online store mirrors the offline
  point-in-time values.
- Before promoting a new feature definition, to measure how historical and
  online values differ.
- As a scheduled data-quality gate over a representative entity sample.

## Entry points

### Python

```python
from feast import FeatureStore, ConsistencyValidationConfig

store = FeatureStore(repo_path=".")
report = store.validate_feature_consistency(
    entity_df=validation_entities,  # must contain join keys + event_timestamp
    features=["driver_stats:conv_rate", "driver_stats:avg_daily_trips"],
    config=ConsistencyValidationConfig(float_rtol=1e-5, float_atol=1e-8),
)
print(report.mismatch_rate)
print(report.reason_counts)
```

A `FeatureService` may be passed in place of the feature-reference list.

### CLI

```bash
feast validate-consistency \
  --entity-df validation_entities.parquet \
  --features driver_stats:conv_rate \
  --features driver_stats:avg_daily_trips \
  --output consistency_report.json
```

The entity dataset is a Parquet or CSV file containing every required join
key and an `event_timestamp` column.

## Time semantics

The two time notions are kept strictly separate and are never mixed:

- **Event time** — the `event_timestamp` column of the validation entity
  dataframe. It is the only timestamp used for the offline point-in-time join
  (`source_event_ts <= event_time < source_event_ts + ttl`) and for the
  retention decision. Naive timestamps are interpreted as UTC, matching
  Feast's `make_tzaware` convention.
- **Processing time** — wall-clock "now". The online store has no time-travel
  parameter and always serves its latest row; it computes the
  `OUTSIDE_MAX_AGE` field status against processing time. Validation records
  this status as evidence but does not use it as the retention criterion, so
  validating historical event times against aged-out materialized data does
  not produce false retention failures.

## Comparison and tolerance

Non-numeric values use exact equality. Floating-point values use
`numpy.isclose` semantics:

```
|offline - online| <= float_atol + float_rtol * |online|
```

- `float_rtol` (default `1e-5`) absorbs scale-dependent rounding, including
  the float32/float64 difference between offline aggregation and online
  serialization.
- `float_atol` (default `1e-8`) bounds the error for values near zero, where
  a purely relative tolerance is meaningless.

Integers read online are deserialized as floats by Feast; an offline integer
compared with an online float is treated as a compatible numeric round-trip
(and compared under tolerance), not as a type error. Genuine
string-vs-number or list-vs-scalar differences are reported as
`type_mismatch`. Two NaN values count as equal by default
(`nan_matches_nan=True`).

## Mismatch reasons

Every cell receives exactly one reason; only `value_match` counts as
consistent.

| Reason | Meaning |
|---|---|
| `value_match` | Values agree (exactly, or within tolerance). |
| `out_of_retention` | The validation event time is older than the source data's retention floor (`max_source_event_ts - ttl`); the offline join has no in-window row. |
| `online_unavailable` | The online field status is `NOT_FOUND`/`INVALID` — the feature is not served. Counted as a mismatch, never as a pass. |
| `value_missing` | A value exists on only one side (the other side is null while the feature is served). |
| `window_mismatch` | The online value is marked `OUTSIDE_MAX_AGE` and differs from the offline value — the online store holds a differently windowed/older row. |
| `semantic_mismatch` | Numeric values differ beyond tolerance while both are present and within TTL — typically different batch vs streaming computation semantics. |
| `type_mismatch` | Incompatible value types (e.g. string vs number). |
| `value_mismatch` | Exact-equality mismatch for non-numeric values. |

## Report and the mismatch-rate definition

The mismatch rate is defined per feature cell and is directly recomputable:

```
mismatch_rate = total_mismatch_cells / total_compared_cells
```

- **Denominator** — every requested feature for every input row, including
  cells that were unavailable online or out of retention. There is no silent
  exclusion.
- **Numerator** — every cell whose reason is not `value_match`.

The report also contains per-feature statistics (`compared_cells`,
`mismatch_cells`, `mismatch_rate`, per-reason counts) so a problematic feature
can be pinpointed, plus up to `max_sample_mismatches` evidence records with
both values, their types, the online field status, and a human-readable
detail string.

### Batching and sharding for million-entity scale

`online_read_batch_size` (default `1000`) bounds each online request; the
offline join runs once over the shard. Split a large dataset into shards,
validate each shard independently (possibly on different machines), and merge
the JSON reports:

```python
merged = shard_report_0.merge(shard_report_1)
# or persist/relocate shards:
payload = report.to_dict()
restored = ConsistencyReport.from_dict(payload)
```

Counts and per-feature statistics add associatively, so shards can be reduced
in any order; mismatch samples are concatenated and trimmed to the configured
limit.

## Error and degradation behavior

- **Empty dataset** raises `EmptyConsistencyDatasetError`. A zero-row input
  can never prove consistency, so no zero-mismatch report is produced.
- **Missing `event_timestamp`** raises
  `ConsistencyTimestampColumnMissingError`.
- **Out-of-retention event time** is reported explicitly as
  `out_of_retention` with the TTL in the detail string.
- **Feature unavailable online** is marked `online_unavailable` and included
  in the mismatch numerator.
- If the offline point-in-time join drops an input row, validation raises
  rather than shrinking the denominator.
- If a local source cannot be introspected for its retention floor (remote
  warehouse, SQL source), validation falls back to a processing-time floor;
  all other checks are unaffected.

## Configuration reference

`ConsistencyValidationConfig` fields and their defaults:

| Field | Default | Meaning |
|---|---|---|
| `float_rtol` | `1e-5` | Relative float tolerance. |
| `float_atol` | `1e-8` | Absolute float tolerance. |
| `max_sample_mismatches` | `100` | Evidence records retained per report. |
| `check_out_of_retention` | `True` | Classify out-of-retention cells. |
| `nan_matches_nan` | `True` | Treat NaN-vs-NaN as equal. |
| `prepare_online_store` | `False` | Materialize each view up to the max event time before validating. |
| `online_read_batch_size` | `1000` | Rows per online-read request. |

## Compatibility boundaries and existing behavior

- This is a **new, opt-in entry point**. Existing
  `get_historical_features`, `get_online_features`, and materialization
  behavior are unchanged; when validation is not invoked, Feast behaves
  exactly as before. Old callers need no changes.
- No new `feature_store.yaml` configuration is required. All tuning lives on
  `ConsistencyValidationConfig`; not constructing one means every default
  above applies.
- The check reads features through the existing offline/online interfaces and
  never alters feature computation logic. With the default
  `prepare_online_store=False` it is read-only against the online store.
- It works against data and online-store entries produced by older Feast
  versions, because it only consumes retrieval results and the standard
  `FieldStatus` enum; no registry or proto schema changes are involved.

### Why not reuse the existing test helper?

`tests/utils/e2e_test_validation.py::validate_offline_online_store_consistency`
is a test scaffold, not a product capability: it is hard-wired to one feature
view and one feature, uses a fixed `1e-6` threshold, asserts and aborts on the
first difference, has no mismatch-rate definition, no reason attribution, no
batching/merging, and cannot run as a user-facing entry point. The two
coexist: the test helper keeps serving the universal integration tests, while
this module provides the reusable, report-based validation. The naming
overlap is historical; the new module lives under `feast/consistency.py`.

### Multi-machine coordination

Validation itself is stateless and requires no cross-node coordination:
partition the entity dataset into disjoint shards and run one validation per
shard. Only the optional `prepare_online_store=True` writes (materializes),
and concurrent materialization of the same feature view can contend on the
online store. The recommended pattern is to materialize once, centrally, then
run read-only validation shards in parallel. If a shard fails, discard and
re-run that shard: partial reports are safe to merge because every report is
an independent additive count, and a missing shard simply means its rows are
absent from the denominator rather than counted as matches.

## Known limitations

- On-demand and stream feature views are supported only through the values
  their underlying regular feature views expose; transformations whose
  inputs are request-time only cannot be replayed offline.
- Retention-floor auto-detection reads local file sources; remote sources use
  the processing-time fallback.
- The online store serves latest values only; validation cannot reconstruct
  the online value that *was* served at a past processing time. It compares
  the current online value against the offline value at the requested event
  time, which is precisely the training/serving-skew check operators need.
- Vector/embedding features are compared element-wise under the same float
  tolerance; semantic (approximate-nearest-neighbor) similarity is out of
  scope.
