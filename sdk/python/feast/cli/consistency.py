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

import json

import click
import pandas as pd

from feast.consistency import ConsistencyValidationConfig
from feast.repo_operations import create_feature_store


@click.command(name="validate-consistency")
@click.option(
    "--entity-df",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to a Parquet/CSV entity dataset. Must contain an "
    "'event_timestamp' column (event time) and every required join key.",
)
@click.option(
    "--features",
    multiple=True,
    required=True,
    help="Feature reference in 'feature_view:feature' form. Repeatable.",
)
@click.option(
    "--output",
    type=click.Path(dir_okay=False),
    default=None,
    help="Optional path to write the full JSON report. When omitted, the "
    "summary is printed to stdout only.",
)
@click.option(
    "--float-rtol",
    type=float,
    default=1e-5,
    show_default=True,
    help="Relative tolerance for float comparison (numpy.isclose semantics).",
)
@click.option(
    "--float-atol",
    type=float,
    default=1e-8,
    show_default=True,
    help="Absolute tolerance for float comparison (numpy.isclose semantics).",
)
@click.option(
    "--batch-size",
    type=int,
    default=1000,
    show_default=True,
    help="Number of entity rows per online-read request.",
)
@click.option(
    "--max-samples",
    type=int,
    default=100,
    show_default=True,
    help="Maximum mismatch samples retained in the report.",
)
@click.option(
    "--prepare-online-store",
    is_flag=True,
    default=False,
    show_default=True,
    help="Materialize each involved feature view up to the max event "
    "timestamp before validating. Off by default: validation then never "
    "writes to the online store.",
)
@click.option(
    "--full-feature-names",
    is_flag=True,
    default=False,
    show_default=True,
    help="Use 'feature_view__feature' naming in the report.",
)
@click.pass_context
def validate_consistency_cmd(
    ctx: click.Context,
    entity_df: str,
    features: tuple,
    output: str,
    float_rtol: float,
    float_atol: float,
    batch_size: int,
    max_samples: int,
    prepare_online_store: bool,
    full_feature_names: bool,
):
    """Validate offline point-in-time feature values against online values.

    Event time and processing time are kept distinct: event timestamps in
    the entity dataset drive the offline point-in-time join and TTL checks,
    while the online store serves its latest value and reports its
    FieldStatus. An empty entity dataset errors out instead of reporting a
    zero mismatch rate.
    """
    store = create_feature_store(ctx)
    if entity_df.endswith(".csv"):
        dataset = pd.read_csv(entity_df, parse_dates=["event_timestamp"])
    else:
        dataset = pd.read_parquet(entity_df)

    config = ConsistencyValidationConfig(
        float_rtol=float_rtol,
        float_atol=float_atol,
        max_sample_mismatches=max_samples,
        prepare_online_store=prepare_online_store,
        online_read_batch_size=batch_size,
    )
    report = store.validate_feature_consistency(
        entity_df=dataset,
        features=list(features),
        config=config,
        full_feature_names=full_feature_names,
    )

    payload = report.to_dict()
    click.echo(
        json.dumps(
            {
                "total_rows": payload["total_rows"],
                "total_compared_cells": payload["total_compared_cells"],
                "total_mismatch_cells": payload["total_mismatch_cells"],
                "mismatch_rate": payload["mismatch_rate"],
                "reason_counts": payload["reason_counts"],
            },
            indent=2,
        )
    )
    if output:
        with open(output, "w") as f:
            f.write(report.to_json())
        click.echo(f"Full report written to {output}")
    return report
