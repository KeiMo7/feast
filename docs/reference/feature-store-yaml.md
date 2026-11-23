# feature\_store.yaml

## Overview

`feature_store.yaml` is a file that is placed at the root of the [Feature Repository](feature-repository.md). This file contains configuration about how the feature store runs. An example `feature_store.yaml` is shown below:

{% code title="feature\_store.yaml" %}
```yaml
project: loyal_spider
registry: data/registry.db
provider: local
online_store:
    type: sqlite
    path: data/online_store.db
```
{% endcode %}

## Fields in feature\_store.yaml

* **provider** \("local" or "gcp"\)  — Defines the environment in which Feast will execute data flows.
* **registry** \(a local or GCS filepath\) — Defines the location of the feature registry.
* **online\_store** — Configures the online store. This field will have various subfields depending on the type of online store:
  * **type** \("sqlite" or "datastore"\) — Defines the type of online store.
  * **path** \(a local filepath\) — Parameter for the sqlite online store. Defines the path to the SQLite database file.
  * **project\_id**  — Optional parameter for the datastore online store. Sets the GCP project id used by Feast, if not set Feast will use the default GCP project id in the local environment.
* **project** — Defines a namespace for the entire feature store. Can be used to isolate multiple deployments in a single installation of Feast.

### feature_server

The `feature_server` block configures the Python Feature Server when it is used
to serve online features and handle `/push` requests. This section is optional
and only applies when running the Python feature server.

An example configuration:

```yaml
feature_server:
  type: local
  metrics: # Prometheus metrics configuration. Also achievable via `feast serve --metrics`.
    enabled: true             # Enable Prometheus metrics server on port 8000
    resource: true            # CPU / memory gauges
    request: true             # endpoint latency histograms & request counters
    online_features: true     # online feature retrieval counters + store read & ODFV transform timing
    push: true                # push request counters
    materialization: true     # materialization counters & duration histograms
    freshness: true           # per-feature-view freshness gauges
  offline_push_batching_enabled: true # Enables batching of offline writes processed by /push. Online writes are unaffected.
  offline_push_batching_batch_size: 100 # Maximum number of buffered rows before writing to the offline store.
  offline_push_batching_batch_interval_seconds: 5 # Maximum time rows may remain buffered before a forced flush.
```

### registry

The `registry` field can be a simple path string or an object with additional
configuration. When using the REST registry server, MCP support can be enabled:

```yaml
registry:
  registry_type: sql
  path: postgresql+psycopg://feast:feast@localhost:5432/feast #pragma: allowlist secret
  mcp:
    enabled: true
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `registry_type` | string | `file` | Registry backend (`file`, `sql`, etc.) |
| `path` | string | — | Connection string or file path |
| `schema_mode` | string | `auto` | SQL registry only. `auto`: create tables on startup; `verify`: check tables exist, error if missing; `skip`: no DDL or verification. See [SQL Registry docs](registries/sql.md#schema-management-schema_mode). |
| `mcp.enabled` | bool | `false` | Enable MCP (Model Context Protocol) on the REST registry server |
| `serve_features_while_materializing` | bool | `false` | Keep serving a feature view's last-materialized values while it is in the `MATERIALIZING` state, instead of rejecting online requests during materialization |

When `serve_features_while_materializing` is `true`, online serving continues for a
feature view while `feast materialize` runs against a shared registry (which
transitions the feature view to `MATERIALIZING`). This avoids serving interruptions
from routine incremental materialization; feature views serve their
last-materialized values until materialization completes. States that never had
online data (e.g. `CREATED`, `GENERATED`) remain gated.

When `registry.mcp.enabled` is `true`, the REST registry server exposes registry
metadata (entities, feature views, feature services) as MCP tool endpoints for
LLM agents. Requires `feast[mcp]` to be installed.

### online_status

The optional `online_status` block enables per-row, per-feature status tracking
for `get_online_features`. When enabled, every `OnlineResponse` carries a
`status_report` side-car explaining *why* each value is present or missing
(entity not found, feature never materialized, materialized-but-null,
unsupported type, unregistered feature view, outside TTL, or store
unavailable). The existing response structure (`to_dict()` / `to_df()` /
`to_arrow()`) is unchanged; when the block is absent or `enabled: false`,
behavior is byte-for-byte identical to before.

```yaml
online_status:
  enabled: true           # Attach a status report to every online response
  fill_defaults: false    # Explicitly opt in to filling missing values with defaults
  defaults:               # Optional per-feature fill overrides
    avg_orders_day: 0.0
  read_batch_size: 500    # Split each feature view's read into batches of this size
  max_batch_retries: 2    # Retries per failed batch before marking it store_unavailable
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `enabled` | bool | `false` | Attach an `OnlineStatusReport` to every online response |
| `fill_defaults` | bool | `false` | Replace fillable missing values with defaults in the returned values. The original status is always preserved in the report |
| `defaults` | map | `{}` | Per-feature default overrides keyed by result column name; features without an override get a type-appropriate zero value |
| `read_batch_size` | int | unset | Split large reads into batches so one failing batch does not fail the whole request. Unset keeps today's single-read behavior |
| `max_batch_retries` | int | `2` | Retry cap per failed read batch; exceeded batches are listed in the report's `failed_batches` |

Missing-reason counts are exported through the existing
`feast_online_features_status_total{feature_view, status}` Prometheus counter
with fine-grained `status` labels (`entity_not_found`,
`feature_not_materialized`, `null_value`, `unsupported_type`,
`feature_view_not_registered`, `store_unavailable`, ...).

## Providers

The `provider` field defines the environment in which Feast will execute data flows. As a result, it also determines the default values for other fields.

### Local

When using the local provider:

* Feast can read from **local Parquet data sources.**
* Feast performs historical feature retrieval \(point-in-time joins\) using **pandas.**
* Feast performs online feature serving from a **SQLite database.**

### **GCP**

When using the GCP provider:

* Feast can read data from **BigQuery data sources.**
* Feast performs historical feature retrieval \(point-in-time joins\) in **BigQuery.**
* Feast performs online feature serving from **Google Cloud Datastore.**

**Permissions**

<table>
  <thead>
    <tr>
      <th style="text-align:left"><b>Command</b>
      </th>
      <th style="text-align:left">Component</th>
      <th style="text-align:left">Permissions</th>
      <th style="text-align:left">Recommended Role</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:left"><b>Apply</b>
      </td>
      <td style="text-align:left">BigQuery (source)</td>
      <td style="text-align:left">
        <p>bigquery.jobs.create</p>
        <p>bigquery.readsessions.create</p>
        <p>bigquery.readsessions.getData</p>
      </td>
      <td style="text-align:left">roles/bigquery.user</td>
    </tr>
    <tr>
      <td style="text-align:left"><b>Apply</b>
      </td>
      <td style="text-align:left">Datastore (destination)</td>
      <td style="text-align:left">
        <p>datastore.entities.allocateIds</p>
        <p>datastore.entities.create</p>
        <p>datastore.entities.delete</p>
        <p>datastore.entities.get</p>
        <p>datastore.entities.list</p>
        <p>datastore.entities.update</p>
      </td>
      <td style="text-align:left">roles/datastore.owner</td>
    </tr>
    <tr>
      <td style="text-align:left"><b>Materialize</b>
      </td>
      <td style="text-align:left">BigQuery (source)</td>
      <td style="text-align:left">bigquery.jobs.create</td>
      <td style="text-align:left">roles/bigquery.user</td>
    </tr>
    <tr>
      <td style="text-align:left"><b>Materialize</b>
      </td>
      <td style="text-align:left">Datastore (destination)</td>
      <td style="text-align:left">
        <p>datastore.entities.allocateIds</p>
        <p>datastore.entities.create</p>
        <p>datastore.entities.delete</p>
        <p>datastore.entities.get</p>
        <p>datastore.entities.list</p>
        <p>datastore.entities.update</p>
        <p>datastore.databases.get</p>
      </td>
      <td style="text-align:left">roles/datastore.owner</td>
    </tr>
    <tr>
      <td style="text-align:left"><b>Get Online Features</b>
      </td>
      <td style="text-align:left">Datastore</td>
      <td style="text-align:left">datastore.entities.get</td>
      <td style="text-align:left">roles/datastore.user</td>
    </tr>
    <tr>
      <td style="text-align:left"><b>Get Historical Features</b>
      </td>
      <td style="text-align:left">BigQuery (source)</td>
      <td style="text-align:left">
        <p>bigquery.datasets.get</p>
        <p>bigquery.tables.get</p>
        <p>bigquery.tables.create</p>
        <p>bigquery.tables.updateData</p>
        <p>bigquery.tables.update</p>
        <p>bigquery.tables.delete</p>
        <p>bigquery.tables.getData</p>
      </td>
      <td style="text-align:left">roles/bigquery.dataEditor</td>
    </tr>
  </tbody>
</table>

