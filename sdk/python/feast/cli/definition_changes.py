"""CLI commands for opt-in feature-definition change tracking."""

import json

import click
from tabulate import tabulate

from feast.definition_tracking import change_type_name
from feast.definition_tracking.tracker import FingerprintUnavailable
from feast.repo_operations import create_feature_store


@click.group(name="definition-changes")
def definition_changes_cmd():
    """
    Inspect tracked feature-definition changes and their materialized impact.

    Requires ``definition_tracking.enabled: true`` under ``registry`` in
    feature_store.yaml.
    """
    pass


@definition_changes_cmd.command(name="list")
@click.option(
    "--feature-view", "feature_view_name", default=None, help="Only this view."
)
@click.pass_context
def definition_changes_list(ctx: click.Context, feature_view_name: str | None):
    """List recorded definition changes."""
    store = create_feature_store(ctx)
    changes = store.list_definition_changes(feature_view_name)

    if not changes:
        click.echo("No definition changes recorded.")
        return

    rows = []
    for change in changes:
        diff = ", ".join(
            f"{item['feature_name']}:{item['change_type']}"
            for item in change["feature_changes"]
        )
        rows.append(
            [
                change["feature_view_name"],
                change["sequence_number"],
                "yes" if change["logic_changed"] else "no",
                diff or "(schema unchanged)",
                "missing"
                if change["baseline_missing"]
                else change["previous_fingerprint"][:12],
                change["fingerprint"][:12],
            ]
        )
    click.echo(
        tabulate(
            rows,
            headers=[
                "FEATURE_VIEW",
                "SEQ",
                "LOGIC_CHANGED",
                "FEATURE_CHANGES",
                "PREVIOUS",
                "FINGERPRINT",
            ],
            tablefmt="plain",
        )
    )


@definition_changes_cmd.command(name="analyze")
@click.argument("feature_view_name", type=click.STRING)
@click.option("--sequence", "sequence_number", type=int, default=None)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
@click.pass_context
def definition_changes_analyze(
    ctx: click.Context,
    feature_view_name: str,
    sequence_number: int | None,
    as_json: bool,
):
    """Show which materialized partitions a change affects."""
    store = create_feature_store(ctx)
    try:
        impact = store.analyze_definition_change(feature_view_name, sequence_number)
    except FingerprintUnavailable as exc:
        raise click.ClickException(str(exc)) from exc

    payload = {
        "feature_view_name": impact.feature_view_name,
        "sequence_number": impact.sequence_number,
        "logic_changed": impact.logic_changed,
        "impact_known": impact.impact_known,
        "feature_changes": [
            {
                "feature_name": name,
                "change_type": change_type_name(change_type),
            }
            for name, change_type, _prev, _new in impact.feature_changes
        ],
        "must_recompute": [
            {
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "reason": item.reason,
            }
            for item in impact.must_recompute
        ],
        "can_skip": [
            {
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "reason": item.reason,
            }
            for item in impact.can_skip
        ],
        "undetermined": [
            {
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "reason": item.reason,
            }
            for item in impact.undetermined
        ],
        "downstream_views": impact.downstream_views,
        "notes": impact.notes,
    }
    if as_json:
        click.echo(json.dumps(payload, indent=2, default=str))
        return

    click.echo(f"{impact.feature_view_name} change #{impact.sequence_number}")
    click.echo(
        f"  logic changed: {'yes' if impact.logic_changed else 'no'}; "
        f"impact known: {'yes' if impact.impact_known else 'NO (metadata missing)'}"
    )
    for title, items in (
        ("must recompute", payload["must_recompute"]),
        ("can skip", payload["can_skip"]),
        ("undetermined", payload["undetermined"]),
    ):
        click.echo(f"  {title}: {len(items)}")
        for item in items:
            click.echo(f"    [{item['start']} .. {item['end']}] {item['reason']}")
    if impact.downstream_views:
        click.echo(f"  downstream views: {', '.join(impact.downstream_views)}")
    for note in impact.notes:
        click.echo(f"  note: {note}")


@definition_changes_cmd.command(name="plan")
@click.argument("feature_view_name", type=click.STRING)
@click.option("--sequence", "sequence_number", type=int, default=None)
@click.pass_context
def definition_changes_plan(
    ctx: click.Context, feature_view_name: str, sequence_number: int | None
):
    """Create (or fetch) a resumable recompute plan for a change."""
    store = create_feature_store(ctx)
    try:
        plan = store.create_recompute_plan(feature_view_name, sequence_number)
    except FingerprintUnavailable as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(plan.plan_id)


@definition_changes_cmd.command(name="plan-status")
@click.argument("plan_id", type=click.STRING)
@click.pass_context
def definition_changes_plan_status(ctx: click.Context, plan_id: str):
    """Show progress for a recompute plan."""
    store = create_feature_store(ctx)
    plan = store.get_recompute_plan(plan_id)
    rows = [
        [item["start"], item["end"], item["status"], item["reason"]]
        for item in plan["partitions"]
    ]
    click.echo(
        tabulate(rows, headers=["START", "END", "STATUS", "REASON"], tablefmt="plain")
    )
    click.echo(f"progress: {plan['progress']}")
    if plan["next_pending_partition"] is not None:
        start, end = plan["next_pending_partition"]
        click.echo(f"resume at: [{start} .. {end}]")
    else:
        click.echo("plan complete")


@definition_changes_cmd.command(name="execute")
@click.argument("plan_id", type=click.STRING)
@click.pass_context
def definition_changes_execute(ctx: click.Context, plan_id: str):
    """Re-materialize the pending partitions of a plan (resumable)."""
    store = create_feature_store(ctx)
    result = store.execute_recompute_plan(plan_id)
    click.echo(
        f"processed={result['processed']} completed={result['completed']} "
        f"failed={result['failed']}"
    )
    for error in result["errors"]:
        click.echo(
            f"  failed {error['feature_view']} [{error['start']}]: {error['error']}"
        )
