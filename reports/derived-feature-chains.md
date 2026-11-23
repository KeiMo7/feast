# Report: Multi-layer derived feature chains

**Scope:** turn single-layer derived (on-demand) features into a declarable,
evaluable, explainable multi-layer chain.

**Date:** 2026-11-23

## What changed

| Area | File | Summary |
|---|---|---|
| Chain definition & evaluation | `sdk/python/feast/derived_feature.py` (new) | `DerivationLayer`, `DerivedFeatureChain`, `DerivationResult`, `MissingUpstreamPolicy`; validation, topological ordering, batch/single evaluation, provenance. |
| Feature store API | `sdk/python/feast/feature_store.py` | `apply_derived_feature_chain`, `get_feature_view_dependencies`, `get_feature_view_dependents`, `export_derived_feature_dependencies`; registration-time chain validation; deletion guard. |
| ODFV as a source | `sdk/python/feast/on_demand_feature_view.py` | An ODFV may now source another ODFV; the upstream projection is tagged so it is not mistaken for a raw feature view. |
| Online retrieval | `sdk/python/feast/utils.py` | Transitive ODFV resolution and dependency-ordered evaluation; intermediate outputs computed only when a downstream layer needs them, then dropped. |
| Offline retrieval | `sdk/python/feast/infra/offline_stores/offline_store.py` | Dependency-ordered evaluation; intermediate columns kept for downstream layers but removed unless requested. |
| Delete protection | `sdk/python/feast/diff/registry_diff.py` | `plan`/`apply` refuses to delete a view a surviving view still sources. |
| Config flag | `sdk/python/feast/repo_config.py` | `derived_features.enabled` (default `true`). |
| CLI | `sdk/python/feast/cli/on_demand_feature_views.py` | `feast on-demand-feature-views dependencies`. |
| Errors | `sdk/python/feast/errors.py` | `DerivedFeatureChainError`, `MissingDerivationInputError`, `DerivedFeatureCycleError`, `IncompatibleDerivationTypeError`, `FeatureViewInUseError`, … |
| Public exports | `sdk/python/feast/__init__.py` | `DerivationLayer`, `DerivedFeatureChain`, `DerivationResult`, `MissingUpstreamPolicy`. |
| Docs | `docs/how-to-guides/derived-feature-chains.md`, `docs/SUMMARY.md`, `docs/reference/feature-store-yaml.md`, `docs/reference/feast-cli-commands.md` | New guide, config and CLI reference entries. |

### Design invariants (mapped to the requirements)

1. **Dependencies only from declarations.** A layer reads exactly the `view:feature`
   inputs listed in `inputs`; there is no naming convention and no dependency on
   declaration order. Inline feature names (no `view:`) are rejected.
2. **One source of truth.** The graph lives in the ODFV's `source_feature_view_projections`;
   the chain metadata stored in tags is limited to the ordinal, missing policy, and
   defaults. Evaluation results and provenance are derived from the same objects.
3. **Registration == runtime.** `apply` validates using the same `DerivedFeatureChain`
   object and the same `validate()` routine that runtime ordering is built from.
4. **Single-layer format/call sites unchanged.** An ODFV with only FeatureView/RequestSource
   sources takes exactly the old code path. The chain additions are guarded by the
   presence of an on-demand source.
5. **No values in provenance/logs.** Traces carry layer names, feature names, status,
   row counts, and durations only.

## Requirement coverage

| Requirement | Where it is implemented | Test |
|---|---|---|
| 1. Two- and three-layer chains evaluate correctly | `DerivedFeatureChain.evaluate_batch` | `test_derived_feature.py::test_two_and_three_layer_chains_declare_and_evaluate`, `test_three_layer_chain_history_matches_hand_computation` |
| 2. Evaluation order from dependencies, not declaration order | `_compute_topological_order` | `test_evaluation_order_is_independent_of_declaration_order`, `test_declaration_order_does_not_change_registered_order` |
| 3. Cycles rejected at registration, every member named | `_find_cycle` + `_validate_derived_feature_chains` | `test_two_layer_cycle_is_rejected_with_every_member`, `test_three_layer_cycle_is_rejected_with_every_member`, `test_cycle_rejected_at_registration_with_cycle_members` |
| 4. Missing upstream: whole column unavailable or declared default | `_input_frames` / `_apply_defaults`, `MissingUpstreamPolicy` | `test_missing_upstream_marks_whole_column_unavailable`, `test_missing_upstream_fills_declared_default`, `test_downstream_of_unavailable_layer_is_unavailable_not_null` |
| 5. Batch and single agree | `evaluate_single` runs the batch path | `test_single_row_matches_corresponding_batch_row`, `test_batch_and_single_values_agree_for_row_local_layers` |
| 6. Shared upstream computed once | single evaluation pass per layer | `test_shared_upstream_is_computed_once_and_reused`, `test_shared_upstream_evaluated_once_across_downstream_layers` |
| 7. Historical and online use the same definition | shared `order_feature_views_by_dependencies` + registered projections | `test_online_and_historical_use_the_same_definition`, `test_registered_views_round_trip_into_the_same_chain` |
| 8. Failure localized to layer + feature | per-layer trace with `error`/`missing_inputs` | `test_failure_is_localized_to_layer_and_feature`, `test_transformation_missing_declared_output_is_a_layer_failure` |
| 9. Dependency graph exportable as text | `dependency_text`, `export_derived_feature_dependencies`, CLI | `test_dependency_text_lists_every_edge_in_order`, `test_provenance_export_matches_dependency_graph` |
| 10. Capability off → single-layer unchanged | `derived_features.enabled` | `test_flag_off_keeps_single_layer_working_and_rejects_multi_layer` |
| Missing upstream: name + referrer reported | `validate()` | `test_missing_upstream_reports_name_and_referrer`, `test_missing_upstream_rejected_at_registration_with_names` |
| Type incompatibility rejected at registration | `validate()` + `_types_compatible` | `test_numeric_into_text_transform_is_rejected`, `test_type_incompatible_chain_rejected_at_registration` |
| Timeout → partial result + unfinished list | `evaluate_batch(timeout_seconds=...)` | `test_timeout_returns_partial_result_and_unfinished_list` |
| Upstream retired while downstream references it → block deletion | `delete_feature_view`, `apply` delete guard, `apply_diff_to_registry` guard | `test_delete_blocked_while_upstream_is_referenced`, `test_objects_to_delete_blocks_referenced_upstream` |

## Notes and limitations

- **Rolling transforms and single-row evaluation.** `evaluate_single` runs the
  same code path as `evaluate_batch` over a one-row frame. For row-local
  transforms the two agree exactly (tested). A rolling/window transform over a
  single row sees a one-row window, so it cannot reproduce a batch value that
  depends on neighbouring rows — this is inherent to the transform, not a
  divergence between code paths. Supplying the window through `evaluate_batch`
  reproduces the batch value (`test_window_transform_needs_the_window_in_single_mode`).
- **Type checking granularity.** Compatibility is coarse (numeric ↔ numeric,
  text ↔ text, bool, array); it is designed to catch the numeric-into-text class
  of error the requirements call out, not to model every Feast type relation.
- **Unavailable vs. failed.** A layer with a missing input (and `unavailable`
  policy) is `unavailable`; a layer whose transform raised is `failed`; a layer
  skipped because its deadline passed is `timeout`. All three are reported in the
  result's `incomplete_layers` and per-layer trace.
- **Runtime backends.** Verified on the local/dask store and the ibis/duckdb
  store. The shared `RetrievalJob.to_arrow` path covers all offline stores that
  use it; backends with bespoke ODFV handling (BigQuery/Snowflake `to_*`
  exporters) are unchanged in structure but were not exercised here.

## How to run the tests

```bash
cd sdk/python
python -m pytest tests/unit/test_derived_feature.py tests/unit/test_derived_feature_store.py -v
python -m pytest tests/unit/test_on_demand_feature_view.py \
  tests/unit/test_on_demand_pandas_transformation.py \
  tests/unit/test_on_demand_python_transformation.py \
  tests/unit/test_on_demand_feature_view_aggregation.py \
  tests/unit/test_on_demand_feature_view_input_schema.py -v
```

### Broader regression run (reported separately)

The full unit suite was also run (excluding only the two server files that need
external ports, which are unaffected by this change):

```bash
cd sdk/python
python -m pytest tests/unit -q \
  --ignore=tests/unit/test_feature_server.py \
  --ignore=tests/unit/test_ui_server.py
```

→ **4265 passed, 53 skipped, 5 failed.**

The 5 failures are all stream-feature-view tests
(`test_stream_feature_view.py::{test_stream_feature_view_serialization,
test_stream_feature_view_udfs, test_update_materialization_intervals}` and
`test_local_feature_store.py::{test_apply_stream_feature_view,
test_apply_stream_feature_view_udf}`). They fail identically on the untouched
baseline with `PySparkRuntimeError: [JAVA_GATEWAY_EXITED]` because no Java
runtime is available in this environment. They are unrelated to this change.

The focused regression set (ODFV + utils + feature service + diff + local store
+ offline/online stores) run on its own before the full suite reported
**558 passed, 30 skipped, 2 failed** — the same two PySpark stream-view tests.

Static checks on the changed files pass: `ruff check`, `ruff format --check`,
and `mypy` (no errors).

---

## 1. Original derived-feature (ODFV) tests — regression

Command:

```
cd sdk/python && python -m pytest \
  tests/unit/test_on_demand_feature_view.py \
  tests/unit/test_on_demand_pandas_transformation.py \
  tests/unit/test_on_demand_python_transformation.py \
  tests/unit/test_on_demand_feature_view_aggregation.py \
  tests/unit/test_on_demand_feature_view_input_schema.py -v
```

Result: **48 passed, 4 skipped** (0 failed). The 4 skips are the
pre-existing protobuf/type-availability skips in these files.

```
test_on_demand_feature_view.py::test_hash PASSED              [  1%]
test_on_demand_feature_view.py::test_python_native_transformation_mode PASSED [  3%]
test_on_demand_feature_view.py::test_from_proto_backwards_compatible_udf PASSED [  5%]
test_on_demand_feature_view.py::test_on_demand_feature_view_writes_protos PASSED [  7%]
test_on_demand_feature_view.py::test_on_demand_feature_view_stored_writes PASSED [  9%]
test_on_demand_feature_view.py::test_function_call_syntax PASSED [ 11%]
test_on_demand_feature_view.py::test_track_metrics_defaults_to_false PASSED [ 13%]
test_on_demand_feature_view.py::test_track_metrics_true_persists_via_proto PASSED [ 15%]
test_on_demand_feature_view.py::test_track_metrics_proto_roundtrip_preserves_user_tags PASSED [ 17%]
test_on_demand_feature_view.py::test_track_metrics_false_not_stored_in_tags PASSED [ 19%]
test_on_demand_feature_view.py::test_copy_preserves_track_metrics PASSED [ 21%]
test_on_demand_feature_view.py::test_eq_considers_track_metrics PASSED [ 23%]
test_on_demand_feature_view.py::test_aggregations_valid_without_udf PASSED [ 25%]
test_on_demand_feature_view.py::test_aggregations_only_odfv_proto_roundtrip PASSED [ 26%]
test_on_demand_feature_view.py::test_aggregations_only_odfv_infer_features PASSED [ 28%]
test_on_demand_feature_view.py::test_input_schema_aggregation_no_udf PASSED [ 30%]
test_on_demand_feature_view.py::test_on_demand_feature_view_org_field PASSED [ 32%]
test_on_demand_feature_view.py::test_schema_field_metadata_survives_inference_and_proto_round_trip PASSED [ 34%]
test_on_demand_pandas_transformation.py::test_pandas_transformation PASSED [ 36%]
test_on_demand_pandas_transformation.py::test_pandas_transformation_returning_all_data_types PASSED [ 38%]
test_on_demand_pandas_transformation.py::test_invalid_pandas_transformation_raises_type_error_on_apply PASSED [ 40%]
test_on_demand_pandas_transformation.py::test_odfv_udf_does_not_receive_undeclared_source_columns PASSED [ 42%]
test_on_demand_pandas_transformation.py::test_odfv_python_mode_does_not_receive_undeclared_source_columns PASSED [ 44%]
test_on_demand_pandas_transformation.py::test_odfv_does_not_see_other_requested_odfv_source_columns PASSED [ 46%]
test_on_demand_pandas_transformation.py::test_odfv_udf_receives_aliased_declared_source_columns PASSED [ 48%]
test_on_demand_pandas_transformation.py::test_feature_service_serves_aliased_odfv_features[pandas] PASSED [ 50%]
test_on_demand_pandas_transformation.py::test_feature_service_serves_aliased_odfv_features[python] PASSED [ 51%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_python_docs_demo PASSED [ 53%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_python_pandas_parity PASSED [ 55%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_python_singleton_view PASSED [ 57%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_python_singleton_view_multiple_entity_rows PASSED [ 59%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_setup PASSED [ 61%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformation::test_stored_writes PASSED [ 63%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformationAllDataTypes::test_python_transformation_returning_all_data_types PASSED [ 65%]
test_on_demand_python_transformation.py::TestOnDemandPythonTransformationAllDataTypes::test_setup PASSED [ 67%]
test_on_demand_python_transformation.py::test_invalid_python_transformation_raises_type_error_on_apply PASSED [ 69%]
test_on_demand_python_transformation.py::TestOnDemandTransformationsWithWrites::test_docling_transform SKIPPED [ 71%]
test_on_demand_python_transformation.py::TestOnDemandTransformationsWithWrites::test_materialize_with_odfv_writes SKIPPED [ 73%]
test_on_demand_python_transformation.py::TestOnDemandTransformationsWithWrites::test_stored_writes SKIPPED [ 75%]
test_on_demand_python_transformation.py::TestOnDemandTransformationsWithWrites::test_stored_writes_with_explode SKIPPED [ 76%]
test_on_demand_feature_view_aggregation.py::test_aggregation_python_mode PASSED [ 78%]
test_on_demand_feature_view_aggregation.py::test_aggregation_pandas_mode PASSED [ 80%]
test_on_demand_feature_view_aggregation.py::test_multiple_aggregations PASSED [ 82%]
test_on_demand_feature_view_aggregation.py::test_no_aggregations_returns_original PASSED [ 84%]
test_on_demand_feature_view_aggregation.py::test_empty_data_returns_empty PASSED [ 86%]
test_on_demand_feature_view_aggregation.py::test_aggregation_without_time_window_survives_proto_roundtrip PASSED [ 88%]
test_on_demand_feature_view_aggregation.py::test_odfv_aggregation_still_serves_online_after_proto_roundtrip PASSED [ 90%]
test_on_demand_feature_view_input_schema.py::test_decorator_with_input_schema PASSED [ 92%]
test_on_demand_feature_view_input_schema.py::test_aggregation_aliases PASSED [ 94%]
test_on_demand_feature_view_input_schema.py::test_input_schema_proto_roundtrip PASSED [ 96%]
test_on_demand_feature_view_input_schema.py::test_input_schema_copy PASSED [ 98%]
test_on_demand_feature_view_input_schema.py::test_sources_required_without_input_schema PASSED [100%]
```

## 2. New tests for multi-layer derived feature chains

Command:

```
cd sdk/python && python -m pytest \
  tests/unit/test_derived_feature.py \
  tests/unit/test_derived_feature_store.py -v
```

Result: **48 passed** (0 failed).

```
test_derived_feature.py::test_two_and_three_layer_chains_declare_and_evaluate[2] PASSED [  2%]
test_derived_feature.py::test_two_and_three_layer_chains_declare_and_evaluate[3] PASSED [  4%]
test_derived_feature.py::test_three_layer_result_matches_hand_computation PASSED [  6%]
test_derived_feature.py::test_evaluation_order_is_independent_of_declaration_order[reversed] PASSED [  8%]
test_derived_feature.py::test_evaluation_order_is_independent_of_declaration_order[shuffled] PASSED [ 10%]
test_derived_feature.py::test_declaration_order_and_evaluation_order_can_differ PASSED [ 12%]
test_derived_feature.py::test_two_layer_cycle_is_rejected_with_every_member PASSED [ 14%]
test_derived_feature.py::test_three_layer_cycle_is_rejected_with_every_member PASSED [ 16%]
test_derived_feature.py::test_self_cycle_is_rejected PASSED   [ 18%]
test_derived_feature.py::test_missing_upstream_marks_whole_column_unavailable PASSED [ 20%]
test_derived_feature.py::test_missing_upstream_fills_declared_default PASSED [ 22%]
test_derived_feature.py::test_default_policy_requires_a_default_for_every_input PASSED [ 25%]
test_derived_feature.py::test_downstream_of_unavailable_layer_is_unavailable_not_null PASSED [ 27%]
test_derived_feature.py::test_single_row_matches_corresponding_batch_row PASSED [ 29%]
test_derived_feature.py::test_window_transform_needs_the_window_in_single_mode PASSED [ 31%]
test_derived_feature.py::test_one_row_batch_equals_single_evaluation PASSED [ 33%]
test_derived_feature.py::test_shared_upstream_is_computed_once_and_reused PASSED [ 35%]
test_derived_feature.py::test_shared_upstream_is_computed_once_for_single_row PASSED [ 37%]
test_derived_feature.py::test_failure_is_localized_to_layer_and_feature PASSED [ 39%]
test_derived_feature.py::test_transformation_missing_declared_output_is_a_layer_failure PASSED [ 41%]
test_derived_feature.py::test_dependency_text_lists_every_edge_in_order PASSED [ 43%]
test_derived_feature.py::test_structure_export_matches_declared_graph PASSED [ 45%]
test_derived_feature.py::test_provenance_never_contains_feature_values PASSED [ 47%]
test_derived_feature.py::test_missing_upstream_reports_name_and_referrer PASSED [ 50%]
test_derived_feature.py::test_missing_feature_in_known_view_is_rejected PASSED [ 52%]
test_derived_feature.py::test_internal_reference_to_undeclared_output_is_rejected PASSED [ 54%]
test_derived_feature.py::test_numeric_into_text_transform_is_rejected PASSED [ 56%]
test_derived_feature.py::test_numeric_widening_is_allowed PASSED [ 58%]
test_derived_feature.py::test_timeout_returns_partial_result_and_unfinished_list PASSED [ 60%]
test_derived_feature.py::test_dependencies_cannot_be_inferred_from_names PASSED [ 62%]
test_derived_feature.py::test_inline_feature_name_input_is_rejected PASSED [ 64%]
test_derived_feature.py::test_duplicate_layer_names_are_rejected PASSED [ 66%]
test_derived_feature.py::test_unknown_missing_policy_is_rejected PASSED [ 68%]
test_derived_feature_store.py::test_three_layer_chain_history_matches_hand_computation PASSED [ 70%]
test_derived_feature_store.py::test_three_layer_chain_history_works_on_duckdb_backend PASSED [ 72%]
test_derived_feature_store.py::test_declaration_order_does_not_change_registered_order PASSED [ 75%]
test_derived_feature_store.py::test_cycle_rejected_at_registration_with_cycle_members PASSED [ 77%]
test_derived_feature_store.py::test_missing_upstream_rejected_at_registration_with_names PASSED [ 79%]
test_derived_feature_store.py::test_type_incompatible_chain_rejected_at_registration PASSED [ 81%]
test_derived_feature_store.py::test_online_and_historical_use_the_same_definition PASSED [ 83%]
test_derived_feature_store.py::test_batch_and_single_values_agree_for_row_local_layers PASSED [ 85%]
test_derived_feature_store.py::test_shared_upstream_evaluated_once_across_downstream_layers PASSED [ 87%]
test_derived_feature_store.py::test_provenance_export_matches_dependency_graph PASSED [ 89%]
test_derived_feature_store.py::test_evaluation_provenance_contains_names_and_counts_only PASSED [ 91%]
test_derived_feature_store.py::test_registered_views_round_trip_into_the_same_chain PASSED [ 93%]
test_derived_feature_store.py::test_delete_blocked_while_upstream_is_referenced PASSED [ 95%]
test_derived_feature_store.py::test_objects_to_delete_blocks_referenced_upstream PASSED [ 97%]
test_derived_feature_store.py::test_flag_off_keeps_single_layer_working_and_rejects_multi_layer PASSED [100%]
```

