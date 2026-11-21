# Feature definition change tracking

Feast stores feature views declaratively: `feast apply` overwrites the active
definition, and materialization watermarks move forward. Before this feature
there was no durable record of *which definition produced which materialized
data*, so when a view's logic changed nobody could say whether historical
partitions needed to be recomputed, where the impact stopped, or how much
recomputation was left.

Definition tracking makes that answerable. It is **opt-in** and is a pure
observer: it never changes how features are computed, materialized or served.

## Enable it

```yaml
# feature_store.yaml
registry:
  path: data/registry.db
  definition_tracking:
    enabled: true
```

Defaults and behavior:

| Setting | Default | When unset |
|---|---|---|
| `enabled` | `false` | No fingerprints are computed, no records are written, and every output (online, historical, materialization intervals) is unchanged. The public APIs raise `ValueError` explaining that tracking must be enabled. |
| `require_explicit_partition_provenance` | `false` | Partitions materialized before tracking was enabled are reported as **undetermined**. Set to `true` to raise instead of reporting a partial answer. |

Existing callers need no changes. `store.apply(...)`, `store.materialize(...)`,
`get_online_features` and `get_historical_features` accept the same arguments and
produce the same results with tracking on or off (covered by tests).

Tracking state is stored inside the registry:

* file / S3 / GCS / HDFS registries: `Registry.definition_tracking` in the same
  proto blob as the rest of the metadata;
* SQL registries: a dedicated `definition_tracking` table with one row per
  project.

## What is recorded

Every `feast apply` (and every direct `registry.apply_feature_view`) whose
definition **content** changed appends a `DefinitionChangeRecord`:

* the feature view and its change sequence number;
* the content fingerprint before and after;
* the *logic* fingerprint before and after;
* a per-feature diff: `added`, `removed`, `type_changed`, or `logic_changed`;
* the event-time range the change can affect (earliest materialized partition
  start up to the moment the change was recorded);
* whether a previous baseline existed (`baseline_missing`).

Two fingerprints are derived by SHA-256 over canonical JSON:

* **content fingerprint** covers the authored definition (schema, entities,
  TTL, sources, transformation, description, tags, owner, …). Any authored edit
  produces a new record, so the audit trail is complete.
* **logic fingerprint** covers only the parts that can change produced *values*
  (schema, join keys, TTL, sources, transformation/UDF, mode, aggregations,
  derived-view inputs). A description or owner edit changes the content
  fingerprint but not the logic fingerprint.

The fingerprint is a function of content only. Version strings, apply order and
the machine running `feast apply` are never inputs, so two identical definitions
always compare equal.

Every successful materialization additionally records a `PartitionRecord`: one
materialization interval plus the fingerprints and version number that were
active when it was materialized.

## Assess the impact

```python
impact = store.analyze_definition_change("driver_stats")

impact.must_recompute   # stale partitions, with the reason for each
impact.can_skip         # partitions already produced by equivalent logic
impact.undetermined     # partitions whose provenance was never recorded
impact.impact_known     # False when anything is undetermined
impact.downstream_views # visible dependencies of the changed view
```

Each verdict is one of exactly three states, never a guess:

| Verdict | Meaning |
|---|---|
| **must recompute** | The partition's recorded logic fingerprint differs from the current one. |
| **can skip** | The partition was materialized by the current definition, or by an older one whose logic fingerprint matches. |
| **undetermined** | The partition has no recorded provenance (it predates tracking). Its impact cannot be derived from registry metadata. |

Verdicts are derived from two registry facts only — the view's
`materialization_intervals` and the recorded partition provenance — plus the
dependency graph for downstream views. No manual annotation is consulted.

### Downstream views

Dependents (derived feature views and on-demand feature views) are found by
walking `source_views` / `sources`. A dependent's partition is flagged only when
the upstream *logic* changed **and** the dependent partition was materialized
before the change took effect. When a dependent's materialization time is not
recorded, its impact is reported as undetermined rather than assumed stale.

## Recompute plans

```python
plan = store.create_recompute_plan("driver_stats")

# Execute; safe to interrupt and re-run.
store.execute_recompute_plan(plan.plan_id)

# Progress and the resume point.
status = store.get_recompute_plan(plan.plan_id)
status["progress"]                 # counts per status
status["next_pending_partition"]   # (start, end) or None when complete
```

A plan is keyed by `(project, feature_view, from_fingerprint, to_fingerprint)`,
so creating it twice returns the same plan with progress preserved. Each
partition's status (`pending`, `running`, `completed`, `failed`) is persisted
after every step:

* an interrupted run leaves completed partitions completed;
* re-running resumes at the first pending or failed partition;
* a completed plan re-runs as a no-op.

`execute_recompute_plan` only calls the existing `materialize` path for each
affected interval. It does not introduce new computation code.

You can also drive the plan yourself:

```python
store.update_recompute_plan(
    plan.plan_id,
    partition_start=start,
    partition_end=end,
    status="completed",
)
```

## Two definitions at once

Version-qualified references let two versions coexist:

```python
store.list_feature_view_versions("driver_stats")   # v0, v1, ...

# Historical retrieval pinned to a specific version.
job = store.get_historical_features(
    entity_df=entity_df,
    features=["driver_stats@v0:conv_rate"],
)
```

`driver_stats@v0` resolves to the v0 snapshot while the unqualified name keeps
resolving to the active definition. Pinning to a version whose content no
longer matches the active definition raises `FeatureViewPinConflict` — the
same-name/different-logic case is an explicit error, not a silent merge.

## CLI

```bash
feast definition-changes list
feast definition-changes analyze driver_stats --sequence 2
feast definition-changes analyze driver_stats --json
feast definition-changes plan driver_stats
feast definition-changes plan-status <plan-id>
feast definition-changes execute <plan-id>
```

## Limitations

* **Pre-tracking data.** Partitions materialized before tracking was enabled
  have no provenance. They are reported as undetermined (or raise, in strict
  mode) — never silently classified. To establish a baseline, enable tracking
  and materialize the affected range once.
* **The first tracked apply.** When a view already exists but has no tracked
  history, the first recorded change has `baseline_missing=True`; its effect on
  pre-existing partitions cannot be classified and is reported as undetermined.
* **Multi-machine deployments.** Tracking state is written through the registry.
  The file registry is last-write-wins for concurrent writers (a pre-existing
  limitation of that backend); the SQL registry persists each project's tracking
  state in its own row, so concurrent writers on different projects do not
  collide. A recompute plan is a sequence of ordinary `materialize` calls, so
  run it from a single coordinator; if two coordinators run the same plan, both
  will materialize the same intervals (idempotent for online values, but
  redundant). Tracking never participates in the serving path, so a tracking
  write failure cannot break serving.
* **Impact is per materialization interval.** Feast records materialization as
  intervals, not as separately addressable partitions. "Partition" in this
  document means one recorded interval.
* **Remote registry clients.** When a client talks to a remote registry server,
  tracking runs on the server (the server's `feature_store.yaml` must enable
  it); the client's `definition_tracking` block does not toggle server-side
  recording. Tracking state is then read over the existing registry API.
* **Go feature server / Go SDK.** Definition tracking is a Python SDK and
  registry feature. The Go server reads definitions and serves online features
  as before and does not write or consult tracking state.
