"""Opt-in tracking of feature-definition changes and their materialized impact.

Enable it by adding a ``definition_tracking`` block to ``feature_store.yaml``::

    registry:
      path: data/registry.db
      definition_tracking:
        enabled: true

When disabled (the default), importing this package has no effect on any
existing behavior: no fingerprints are computed, no records are written, and
materialization/retrieval are byte-for-byte unchanged.

See :mod:`feast.definition_tracking.fingerprint` for how content fingerprints
are derived and :mod:`feast.definition_tracking.tracker` for impact analysis and
resumable recompute plans.
"""

from feast.definition_tracking.fingerprint import (
    change_type_name,
    definition_fingerprint,
    diff_feature_sets,
    logic_fingerprint,
    upstream_view_names,
)
from feast.definition_tracking.tracker import (
    ChangeImpact,
    DefinitionTracker,
    FingerprintUnavailable,
    PartitionImpact,
    PartitionStatus,
)

__all__ = [
    "ChangeImpact",
    "DefinitionTracker",
    "FingerprintUnavailable",
    "PartitionImpact",
    "PartitionStatus",
    "change_type_name",
    "definition_fingerprint",
    "diff_feature_sets",
    "logic_fingerprint",
    "upstream_view_names",
]
