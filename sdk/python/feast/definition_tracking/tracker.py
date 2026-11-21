"""Definition change tracking, impact analysis and resumable recompute plans.

The tracker is a thin, opt-in layer on top of the registry proto. It is
*read-only* with respect to feature definitions and materialization: nothing in
this module changes how features are computed, materialized or served. It only
appends records to ``Registry.definition_tracking``.

Design notes
------------
* Change detection is fingerprint based (see :mod:`fingerprint`); no version
  string or human input is consulted.
* Impact analysis is derived from two pieces of registry metadata only:
  materialization intervals (the partitions a feature view owns) and the
  dependency graph (derived views / ODFV sources). It never guesses.
* Recompute plans are keyed by ``(project, feature_view, from_fp, to_fp)`` and
  store per-partition status, so an interrupted run can be resumed without
  recomputing finished partitions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from google.protobuf.timestamp_pb2 import Timestamp

from feast.definition_tracking import fingerprint as fp
from feast.protos.feast.core.DefinitionTracking_pb2 import (
    DefinitionChangeRecord,
    DefinitionTrackingState,
    FeatureChange,
    PartitionRecord,
    RecomputePartition,
    RecomputePartitionStatus,
    RecomputePlan,
)
from feast.utils import _utc_now, make_tzaware

# Re-exported so callers do not need to import the proto module directly.
__all__ = [
    "DefinitionTracker",
    "ChangeImpact",
    "PartitionImpact",
    "PartitionStatus",
    "FingerprintUnavailable",
]

PartitionStatus = RecomputePartitionStatus


class FingerprintUnavailable(RuntimeError):
    """Raised when a conclusion would require a fingerprint we do not have.

    Impact analysis must never fabricate an answer. When the metadata needed to
    decide a partition's provenance (or the previous definition) is missing,
    callers get this instead of a guess.
    """


@dataclass
class PartitionImpact:
    """Impact verdict for one partition.

    A verdict is exactly one of three states:

    * ``required=True``  -- stale relative to the current definition, must be
      recomputed.
    * ``required=False`` -- already produced by the current (or an
      equivalent-logic) definition, can be skipped.
    * ``undetermined=True`` -- the partition's provenance was never recorded,
      so no correct answer exists. Nothing is inferred; the caller is told the
      metadata is missing.

    ``required`` and ``undetermined`` are mutually exclusive. A missing
    provenance is never silently collapsed into either of the other two.
    """

    start: datetime
    end: datetime
    required: bool
    reason: str
    fingerprint: Optional[str] = None
    version_number: Optional[int] = None
    undetermined: bool = False

    @property
    def can_skip(self) -> bool:
        return not self.required and not self.undetermined

    def as_proto(self) -> RecomputePartition:
        if self.undetermined:
            status = RecomputePartitionStatus.RECOMPUTE_STATUS_UNSPECIFIED
        elif self.required:
            status = RecomputePartitionStatus.RECOMPUTE_PENDING
        else:
            status = RecomputePartitionStatus.RECOMPUTE_COMPLETED
        partition = RecomputePartition(
            status=status,
            reason=self.reason,
            unknown_provenance=self.undetermined or self.fingerprint is None,
        )
        partition.start_time.FromDatetime(self.start)
        partition.end_time.FromDatetime(self.end)
        if not self.required and not self.undetermined:
            partition.completed_at.FromDatetime(_utc_now())
        return partition


@dataclass
class ChangeImpact:
    """Impact report for a single recorded definition change."""

    project: str
    feature_view_name: str
    feature_view_type: str
    sequence_number: int
    previous_fingerprint: str
    fingerprint: str
    logic_changed: bool
    affected_start: Optional[datetime]
    affected_end: Optional[datetime]
    feature_changes: List[Tuple[str, int, str, str]] = field(default_factory=list)
    # Partitions of the changed view itself, plus affected downstream views.
    partitions: List[PartitionImpact] = field(default_factory=list)
    downstream: Dict[str, List[PartitionImpact]] = field(default_factory=dict)
    # Names of views whose own definition is unchanged but which consume the
    # changed view (transitively).
    downstream_views: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def must_recompute(self) -> List[PartitionImpact]:
        return [
            partition
            for partition in self.partitions
            if partition.required and not partition.undetermined
        ]

    @property
    def can_skip(self) -> List[PartitionImpact]:
        return [partition for partition in self.partitions if partition.can_skip]

    @property
    def undetermined(self) -> List[PartitionImpact]:
        """Partitions whose provenance is missing, impact cannot be decided."""
        return [partition for partition in self.partitions if partition.undetermined]

    @property
    def impact_known(self) -> bool:
        """False when at least one partition's impact cannot be determined."""
        return not self.undetermined

    def downstream_must_recompute(self, view: str) -> List[PartitionImpact]:
        return [
            partition
            for partition in self.downstream.get(view, [])
            if partition.required and not partition.undetermined
        ]


def _datetime(stamp: Timestamp) -> Optional[datetime]:
    if stamp is None or not stamp.FromDatetime:  # pragma: no cover - defensive
        return None
    if stamp.seconds == 0 and stamp.nanos == 0:
        return None
    return make_tzaware(stamp.ToDatetime())


def _timestamp_nanos(stamp: Timestamp) -> int:
    return stamp.seconds * 1_000_000_000 + stamp.nanos


def _to_nanos(value: datetime) -> int:
    stamp = Timestamp()
    stamp.FromDatetime(make_tzaware(value))
    return _timestamp_nanos(stamp)


class DefinitionTracker:
    """Reads and writes definition-tracking state on a registry proto.

    The tracker is stateless; it operates on the passed-in
    :class:`DefinitionTrackingState` message, which is owned by the registry.
    """

    def __init__(
        self,
        project: str,
        state: DefinitionTrackingState,
        materialization_intervals: Optional[
            Mapping[str, Sequence[Tuple[datetime, datetime]]]
        ] = None,
    ):
        self.project = project
        self.state = state
        self._materialization_intervals = materialization_intervals or {}

    # ------------------------------------------------------------------ reads

    def records_for(self, feature_view_name: str) -> List[DefinitionChangeRecord]:
        return [
            record
            for record in self.state.changes
            if record.project == self.project
            and record.feature_view_name == feature_view_name
        ]

    def latest_record(self, feature_view_name: str) -> Optional[DefinitionChangeRecord]:
        records = self.records_for(feature_view_name)
        if not records:
            return None
        return max(records, key=lambda record: record.sequence_number)

    def _next_sequence(self, feature_view_name: str) -> int:
        records = self.records_for(feature_view_name)
        if not records:
            return 1
        return max(record.sequence_number for record in records) + 1

    def recorded_partition_fingerprints(
        self, feature_view_name: str
    ) -> Dict[Tuple[datetime, datetime], PartitionRecord]:
        keyed: Dict[Tuple[datetime, datetime], PartitionRecord] = {}
        for record in self.state.partitions:
            if (
                record.project == self.project
                and record.feature_view_name == feature_view_name
            ):
                start = _datetime(record.start_time)
                end = _datetime(record.end_time)
                if start is None or end is None:
                    continue
                keyed[(start, end)] = record
        return keyed

    # ----------------------------------------------------------------- writes

    def record_apply(
        self,
        *,
        feature_view_name: str,
        feature_view_type: str,
        current_proto: Any,
        previous_proto: Optional[Any],
        version_number: Optional[int],
        previous_version_number: Optional[int] = None,
        recorded_at: Optional[datetime] = None,
        affected_end: Optional[datetime] = None,
    ) -> Optional[DefinitionChangeRecord]:
        """Append a change record if the definition content changed.

        ``previous_proto`` is the definition that was active before this apply.
        When it is ``None`` (the view is new, or a pre-tracking registry has no
        recorded baseline), a record with ``baseline_missing=True`` is written
        so the change history still has an anchor, and no per-feature diff is
        invented.

        Returns the new record, or ``None`` when nothing changed.
        """
        recorded_at = recorded_at or _utc_now()
        current_fp = fp.definition_fingerprint(current_proto)
        current_logic_fp = fp.logic_fingerprint(current_proto)

        if previous_proto is not None:
            previous_fp = fp.definition_fingerprint(previous_proto)
            previous_logic_fp = fp.logic_fingerprint(previous_proto)
            if previous_fp == current_fp:
                return None
            logic_changed = previous_logic_fp != current_logic_fp
            changes = fp.diff_feature_sets(
                previous_proto, current_proto, logic_changed=logic_changed
            )
            baseline_missing = False
        else:
            previous_fp = ""
            previous_logic_fp = ""
            logic_changed = True
            changes = []
            baseline_missing = True

        record = DefinitionChangeRecord(
            sequence_number=self._next_sequence(feature_view_name),
            project=self.project,
            feature_view_name=feature_view_name,
            feature_view_type=feature_view_type,
            previous_fingerprint=previous_fp,
            fingerprint=current_fp,
            previous_logic_fingerprint=previous_logic_fp,
            logic_fingerprint=current_logic_fp,
            previous_version_number=previous_version_number
            if previous_version_number is not None
            else -1,
            version_number=version_number if version_number is not None else -1,
            baseline_missing=baseline_missing,
        )
        for name, change_type, previous_type, new_type in changes:
            record.feature_changes.append(
                FeatureChange(
                    feature_name=name,
                    change_type=change_type,  # type: ignore[arg-type]
                    previous_type=previous_type,
                    new_type=new_type,
                )
            )

        affected_start = self._affected_start(feature_view_name)
        affected_end = affected_end or recorded_at
        if affected_start is not None:
            record.affected_start.FromDatetime(make_tzaware(affected_start))
        record.affected_end.FromDatetime(make_tzaware(affected_end))
        record.recorded_at.FromDatetime(make_tzaware(recorded_at))

        self.state.changes.append(record)
        return record

    def _affected_start(self, feature_view_name: str) -> Optional[datetime]:
        """Earliest materialized partition start for the view, if any.

        Reads the ``materialization_intervals`` passed in by the registry (not
        only the partitions recorded while tracking was on): a view that was
        materialized before tracking was enabled still has an affected range.
        """
        starts = [
            make_tzaware(start)
            for (start, _end) in self._materialization_intervals.get(
                feature_view_name, []
            )
            if start is not None
        ]
        return min(starts) if starts else None

    def record_partition(
        self,
        *,
        feature_view_name: str,
        start: datetime,
        end: datetime,
        fingerprint: Optional[str],
        logic_fingerprint: Optional[str] = None,
        version_number: Optional[int] = None,
        materialized_at: Optional[datetime] = None,
    ) -> Optional[PartitionRecord]:
        """Record the provenance of one materialized partition.

        A later materialization of the same interval with a different
        fingerprint replaces the earlier record: the partition now holds data
        produced by the newer definition.
        """
        materialized_at = materialized_at or _utc_now()
        start = make_tzaware(start)
        end = make_tzaware(end)

        existing = None
        for record in self.state.partitions:
            if (
                record.project == self.project
                and record.feature_view_name == feature_view_name
                and _datetime(record.start_time) == start
                and _datetime(record.end_time) == end
            ):
                existing = record
                break

        if existing is not None and fingerprint is not None:
            if (
                existing.fingerprint == fingerprint
                and _datetime(existing.materialized_at) is not None
            ):
                return existing

        record = existing or self.state.partitions.add()
        record.project = self.project
        record.feature_view_name = feature_view_name
        record.fingerprint = fingerprint or ""
        record.logic_fingerprint = logic_fingerprint or ""
        record.version_number = version_number if version_number is not None else -1
        record.start_time.FromDatetime(start)
        record.end_time.FromDatetime(end)
        record.materialized_at.FromDatetime(make_tzaware(materialized_at))
        return record

    # ------------------------------------------------------- impact analysis

    def analyze(
        self,
        record: DefinitionChangeRecord,
        *,
        partitions_by_view: Mapping[str, Sequence[Tuple[datetime, datetime]]],
        definitions_by_view: Optional[Dict[str, Any]] = None,
    ) -> ChangeImpact:
        """Compute the impact of a recorded change.

        Args:
            record: The change record to analyse.
            partitions_by_view: Materialization intervals per feature view, as
                read from the registry (``materialization_intervals``). This is
                the authoritative source of which partitions exist.
            definitions_by_view: Optional current protos, used to discover
                downstream dependents of the changed view.

        Raises:
            FingerprintUnavailable: when the record's own fingerprint is missing
                or the partition metadata needed to decide is absent.
        """
        if not record.fingerprint:
            raise FingerprintUnavailable(
                f"Cannot determine the impact of the change to "
                f"'{record.feature_view_name}': the change record has no "
                f"content fingerprint."
            )

        impact = ChangeImpact(
            project=record.project,
            feature_view_name=record.feature_view_name,
            feature_view_type=record.feature_view_type,
            sequence_number=record.sequence_number,
            previous_fingerprint=record.previous_fingerprint,
            fingerprint=record.fingerprint,
            logic_changed=record.previous_logic_fingerprint != record.logic_fingerprint,
            affected_start=_datetime(record.affected_start),
            affected_end=_datetime(record.affected_end),
            feature_changes=[
                (
                    change.feature_name,
                    change.change_type,
                    change.previous_type,
                    change.new_type,
                )
                for change in record.feature_changes
            ],
        )

        direct = partitions_by_view.get(record.feature_view_name)
        if direct is None:
            # The caller has no partition metadata at all for this view. We can
            # still say the definition changed, but we must not claim "nothing
            # to recompute" or "everything to recompute".
            raise FingerprintUnavailable(
                f"No materialization metadata is available for "
                f"'{record.feature_view_name}', so its affected partitions "
                f"cannot be determined."
            )
        if record.baseline_missing:
            # The view has never had a tracked baseline, so there is no
            # previous definition to compare against. Any existing partition
            # was produced by an unknown definition.
            impact.notes.append(
                f"No previous tracked definition exists for "
                f"'{record.feature_view_name}' (this is its first tracked "
                f"apply), so individual partitions cannot be compared."
            )
            for start, end in direct:
                impact.partitions.append(
                    PartitionImpact(
                        start=make_tzaware(start),
                        end=make_tzaware(end),
                        required=False,
                        undetermined=True,
                        reason=(
                            "no previous tracked definition exists; the "
                            "partition predates tracking and its impact cannot "
                            "be determined"
                        ),
                    )
                )
            return impact

        for start, end in direct:
            impact.partitions.append(
                self._verdict(
                    record.feature_view_name,
                    start,
                    end,
                    current_fingerprint=record.fingerprint,
                    current_logic_fingerprint=record.logic_fingerprint,
                )
            )

        if definitions_by_view:
            dependents = self._downstream_views(
                record.feature_view_name, definitions_by_view
            )
            impact.downstream_views = dependents
            changed_at = _datetime(record.recorded_at)
            for dependent in dependents:
                dependent_partitions = partitions_by_view.get(dependent)
                if dependent_partitions is None:
                    impact.notes.append(
                        f"Dependent view '{dependent}' has no materialization "
                        f"metadata; its affected partitions cannot be determined."
                    )
                    continue
                impact.downstream[dependent] = [
                    self._downstream_verdict(
                        dependent,
                        start,
                        end,
                        upstream=record.feature_view_name,
                        upstream_logic_changed=impact.logic_changed,
                        changed_at=changed_at,
                    )
                    for start, end in dependent_partitions
                ]

        if not impact.partitions:
            impact.notes.append(
                f"'{record.feature_view_name}' has no materialized partitions, "
                f"so this change affects no existing data."
            )
        return impact

    def _verdict(
        self,
        feature_view_name: str,
        start: datetime,
        end: datetime,
        *,
        current_fingerprint: str,
        current_logic_fingerprint: str,
    ) -> PartitionImpact:
        """Decide one partition from its recorded provenance.

        The verdict compares the partition's *logic* fingerprint (the part that
        determines values) with the current one. This makes a metadata-only
        change (description, owner, tags, online/offline flags) correctly
        skippable instead of forcing a recompute.
        """
        start = make_tzaware(start)
        end = make_tzaware(end)
        provenance = self.recorded_partition_fingerprints(feature_view_name).get(
            (start, end)
        )

        if provenance is None or not provenance.fingerprint:
            # No recorded provenance: the partition may have been produced by
            # any definition. Report the gap; do not invent a verdict.
            return PartitionImpact(
                start=start,
                end=end,
                required=False,
                undetermined=True,
                reason=(
                    "partition provenance is missing (the range was materialized "
                    "before definition tracking was enabled); impact cannot be "
                    "determined from registry metadata"
                ),
                fingerprint=None,
            )

        version_number = (
            provenance.version_number if provenance.version_number >= 0 else None
        )

        if provenance.fingerprint == current_fingerprint:
            return PartitionImpact(
                start=start,
                end=end,
                required=False,
                reason="already materialized with the current definition",
                fingerprint=provenance.fingerprint,
                version_number=version_number,
            )

        if provenance.logic_fingerprint:
            if provenance.logic_fingerprint == current_logic_fingerprint:
                return PartitionImpact(
                    start=start,
                    end=end,
                    required=False,
                    reason=(
                        "materialized with a definition whose computation logic "
                        "matches the current one; only metadata changed"
                    ),
                    fingerprint=provenance.fingerprint,
                    version_number=version_number,
                )
            return PartitionImpact(
                start=start,
                end=end,
                required=True,
                reason=(
                    "materialized with a definition whose computation logic "
                    "differs from the current one"
                ),
                fingerprint=provenance.fingerprint,
                version_number=version_number,
            )

        # Provenance exists but has no logic fingerprint (written by an older
        # tracking version). Fall back to the full fingerprint, which is exact
        # for the previous definition but cannot rule out an intermediate one.
        return PartitionImpact(
            start=start,
            end=end,
            required=True,
            reason=(
                "materialized with a definition other than the current one; its "
                "logic fingerprint was not recorded, so a recompute is required"
            ),
            fingerprint=provenance.fingerprint,
            version_number=version_number,
        )

    def _downstream_verdict(
        self,
        dependent: str,
        start: datetime,
        end: datetime,
        *,
        upstream: str,
        upstream_logic_changed: bool,
        changed_at: Optional[datetime],
    ) -> PartitionImpact:
        """Decide a dependent view's partition from dependency + partition time."""
        start = make_tzaware(start)
        end = make_tzaware(end)

        if not upstream_logic_changed:
            return PartitionImpact(
                start=start,
                end=end,
                required=False,
                reason=(
                    f"depends on '{upstream}', but only non-value metadata "
                    f"changed upstream"
                ),
            )

        provenance = self.recorded_partition_fingerprints(dependent).get((start, end))
        if provenance is None or _datetime(provenance.materialized_at) is None:
            return PartitionImpact(
                start=start,
                end=end,
                required=False,
                undetermined=True,
                reason=(
                    f"depends on '{upstream}' whose logic changed, but this "
                    f"partition's materialization time is not recorded; impact "
                    f"cannot be determined"
                ),
            )

        materialized_at = _datetime(provenance.materialized_at)
        if (
            changed_at is not None
            and materialized_at is not None
            and materialized_at >= changed_at
        ):
            return PartitionImpact(
                start=start,
                end=end,
                required=False,
                reason=(
                    f"depends on '{upstream}', but was materialized after the "
                    f"upstream change took effect"
                ),
                fingerprint=provenance.fingerprint or None,
            )

        return PartitionImpact(
            start=start,
            end=end,
            required=True,
            reason=(
                f"depends on '{upstream}' whose logic changed after this "
                f"partition was materialized"
            ),
            fingerprint=provenance.fingerprint or None,
        )

    def _downstream_views(
        self, feature_view_name: str, definitions_by_view: Dict[str, Any]
    ) -> List[str]:
        """Transitive dependents of ``feature_view_name`` (sorted, stable)."""
        reverse: Dict[str, List[str]] = {}
        for name, proto in definitions_by_view.items():
            for upstream in fp.upstream_view_names(proto):
                reverse.setdefault(upstream, []).append(name)

        found: set[str] = set()
        frontier = [feature_view_name]
        while frontier:
            current = frontier.pop()
            for dependent in reverse.get(current, []):
                if dependent not in found:
                    found.add(dependent)
                    frontier.append(dependent)
        found.discard(feature_view_name)
        return sorted(found)

    # ------------------------------------------------------ recompute plans

    def create_plan(
        self,
        impact: ChangeImpact,
        *,
        change_sequence_number: Optional[int] = None,
    ) -> RecomputePlan:
        """Create (or return the existing) recompute plan for an impact report.

        The plan is keyed by the source/target fingerprints, so generating it
        twice is idempotent. Existing partition statuses are preserved, which is
        what makes an interrupted recompute resumable.
        """
        plan = self._find_plan(
            impact.feature_view_name,
            impact.previous_fingerprint,
            impact.fingerprint,
        )
        if plan is not None:
            return plan

        now = _utc_now()
        plan = self.state.plans.add(
            plan_id=str(uuid.uuid4()),
            project=self.project,
            feature_view_name=impact.feature_view_name,
            from_fingerprint=impact.previous_fingerprint,
            to_fingerprint=impact.fingerprint,
            change_sequence_number=change_sequence_number or impact.sequence_number,
        )
        plan.created_at.FromDatetime(now)
        plan.updated_at.FromDatetime(now)

        for partition in impact.must_recompute:
            plan.partitions.append(partition.as_proto())
        for dependent, verdicts in impact.downstream.items():
            for verdict in verdicts:
                if not verdict.required:
                    continue
                proto = plan.partitions.add()
                proto.CopyFrom(verdict.as_proto())
                proto.reason = f"[{dependent}] {proto.reason}"
        return plan

    def _find_plan(
        self, feature_view_name: str, from_fingerprint: str, to_fingerprint: str
    ) -> Optional[RecomputePlan]:
        for plan in self.state.plans:
            if (
                plan.project == self.project
                and plan.feature_view_name == feature_view_name
                and plan.from_fingerprint == from_fingerprint
                and plan.to_fingerprint == to_fingerprint
            ):
                return plan
        return None

    def plans_for(self, feature_view_name: str) -> List[RecomputePlan]:
        return [
            plan
            for plan in self.state.plans
            if plan.project == self.project
            and plan.feature_view_name == feature_view_name
        ]

    def mark_partition(
        self,
        plan: RecomputePlan,
        *,
        start: datetime,
        end: datetime,
        status: int,
        error: str = "",
    ) -> bool:
        """Update the status of one partition in a plan.

        Returns ``False`` when the plan has no matching partition.
        """
        start_ns = _to_nanos(start)
        end_ns = _to_nanos(end)
        for partition in plan.partitions:
            if (
                _timestamp_nanos(partition.start_time) == start_ns
                and _timestamp_nanos(partition.end_time) == end_ns
            ):
                partition.status = status
                partition.error = error
                if status == RecomputePartitionStatus.RECOMPUTE_RUNNING:
                    partition.started_at.FromDatetime(_utc_now())
                elif status in (
                    RecomputePartitionStatus.RECOMPUTE_COMPLETED,
                    RecomputePartitionStatus.RECOMPUTE_FAILED,
                ):
                    partition.completed_at.FromDatetime(_utc_now())
                plan.updated_at.FromDatetime(_utc_now())
                return True
        return False

    def next_pending(self, plan: RecomputePlan) -> Optional[RecomputePartition]:
        """First partition that still needs work (pending or interrupted)."""
        for partition in plan.partitions:
            if partition.status in (
                RecomputePartitionStatus.RECOMPUTE_PENDING,
                RecomputePartitionStatus.RECOMPUTE_RUNNING,
                RecomputePartitionStatus.RECOMPUTE_FAILED,
            ):
                return partition
        return None

    def plan_progress(self, plan: RecomputePlan) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for partition in plan.partitions:
            name = RecomputePartitionStatus.Name(partition.status).lower()
            counts[name] = counts.get(name, 0) + 1
        return counts


def new_tracking_state() -> DefinitionTrackingState:
    return DefinitionTrackingState()
