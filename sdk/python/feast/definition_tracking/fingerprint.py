"""Content fingerprints for feature definitions.

Tracking is opt-in, but the fingerprint itself is a pure function of the
definition's content, so it never depends on a manually supplied version
string. The same definition always produces the same fingerprint regardless of
which machine or process computed it.

Two fingerprints are computed for every tracked view:

* ``definition_fingerprint`` -- the definition's content (schema, entities,
  TTL, sources, transformation, ...). Descriptions, tags, owners and lifecycle
  flags are excluded: they cannot change a feature's name, type or values, and
  treating a typo fix as a definition change would bury the real ones.
* ``logic_fingerprint`` -- only the parts that can change the *values* a view
  produces (transformation/UDF, derived-view inputs, field mapping, mode,
  aggregation windows, TTL). A metadata-only edit does not change it. This is
  what separates "must recompute" from "no recompute needed".

Fingerprints are SHA-256 over a canonical JSON document with sorted keys, so
they are deterministic and independent of protobuf field ordering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Tuple

from feast.protos.feast.core.DefinitionTracking_pb2 import FeatureChangeType

# Bump when the canonicalization itself changes. Fingerprints are only compared
# within a single scheme version; a version mismatch is treated as "unknown".
FINGERPRINT_SCHEME_VERSION = 1

# Spec fields that are operational state rather than authored definition
# content, and are therefore excluded from the content fingerprint. Everything
# else the user can edit (description, tags, owner, ttl, sources, ...) is part
# of the definition: a change to any of it is recorded, even when it cannot
# change feature values. That keeps a complete audit trail while the separate
# logic fingerprint decides whether recomputation is actually needed.
_NON_CONTENT_SPEC_FIELDS = {
    "project",
    "version",
}

# Volatile keys dropped at *any* nesting depth. Nested messages (e.g. a data
# source's ``meta``) carry wall-clock timestamps that differ on every apply.
_VOLATILE_NESTED_FIELDS = {
    "created_timestamp",
    "last_updated_timestamp",
    "materialization_intervals",
    "current_version_number",
    "version_id",
    "state",
}

# Fields whose change can alter produced values (not just the feature set).
_VALUE_AFFECTING_SPEC_FIELDS = {
    # Join keys define which rows a partition holds, so they change values.
    "entities",
    "entity_columns",
    "ttl",
    "batch_source",
    "stream_source",
    "source_views",
    "sources",
    "feature_transformation",
    "user_defined_function",
    "udf",
    "mode",
    "aggregations",
    "timestamp_field",
    "category_field",
    "date_partition_column",
    "labeler_field",
    "reference_feature_view",
    "reference_feature_view_projection",
    "conflict_policy",
}

_CHANGE_TYPE_NAMES = {
    int(FeatureChangeType.FEATURE_ADDED): "added",
    int(FeatureChangeType.FEATURE_REMOVED): "removed",
    int(FeatureChangeType.FEATURE_TYPE_CHANGED): "type_changed",
    int(FeatureChangeType.FEATURE_LOGIC_CHANGED): "logic_changed",
}


def _canonical_json(document: Any) -> str:
    return json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def sha256_hex(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _message_to_dict(message: Any) -> Dict[str, Any]:
    """Convert a proto message to a plain dict.

    ``MessageToDict`` renders enums by name (stable across unrelated
    renumbering). Repeated fields keep their declared order, which is part of
    the definition's semantics in Feast (e.g. derived-view input order).
    """
    from google.protobuf.json_format import MessageToDict

    return MessageToDict(
        message,
        preserving_proto_field_name=True,
        use_integers_for_enums=False,
        always_print_fields_with_no_presence=False,
    )


def _strip_volatile(document: Any) -> Any:
    if isinstance(document, dict):
        return {
            key: _strip_volatile(value)
            for key, value in document.items()
            if key not in _VOLATILE_NESTED_FIELDS
        }
    if isinstance(document, list):
        return [_strip_volatile(value) for value in document]
    return document


def _content_spec(proto: Any) -> Dict[str, Any]:
    document = _strip_volatile(_message_to_dict(proto))
    spec = document.get("spec", {})
    return {
        key: value for key, value in spec.items() if key not in _NON_CONTENT_SPEC_FIELDS
    }


def _feature_schema_payload(features: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {"name": feature.get("name", ""), "type": feature.get("value_type", "")}
        for feature in features
    ]


def definition_fingerprint(proto: Any) -> str:
    """Fingerprint the content of a feature view proto."""
    body = {
        "spec": _content_spec(proto),
        "features": _feature_schema_payload(
            _message_to_dict(proto).get("spec", {}).get("features", [])
        ),
    }
    return sha256_hex(
        _canonical_json(
            {"v": FINGERPRINT_SCHEME_VERSION, "kind": "definition", "body": body}
        )
    )


def logic_fingerprint(proto: Any) -> str:
    """Fingerprint only the value-affecting parts of a feature view proto."""
    spec = _content_spec(proto)
    body: Dict[str, Any] = {
        "spec": {
            key: value
            for key, value in spec.items()
            if key in _VALUE_AFFECTING_SPEC_FIELDS
        },
        "features": _feature_schema_payload(spec.get("features", [])),
    }
    return sha256_hex(
        _canonical_json(
            {"v": FINGERPRINT_SCHEME_VERSION, "kind": "logic", "body": body}
        )
    )


def _feature_type_map(proto: Any) -> Dict[str, str]:
    types: Dict[str, str] = {}
    for feature in proto.spec.features:
        types[feature.name] = str(feature.value_type)
    return types


def diff_feature_sets(
    previous_proto: Any,
    current_proto: Any,
    *,
    logic_changed: bool,
) -> List[Tuple[str, int, str, str]]:
    """Diff the feature sets of two definitions.

    Returns ``(feature_name, change_type, previous_type, new_type)`` tuples
    sorted by feature name.

    A feature whose type is unchanged is reported as ``FEATURE_LOGIC_CHANGED``
    only when ``logic_changed`` is true: Feast pulls values through a
    declarative view, so a change to the transformation or the derived-view
    inputs can alter every feature's values without touching the schema.
    """
    previous_types = _feature_type_map(previous_proto)
    current_types = _feature_type_map(current_proto)

    changes: List[Tuple[str, int, str, str]] = []
    for name in sorted(set(previous_types) | set(current_types)):
        in_previous = name in previous_types
        in_current = name in current_types
        if not in_previous:
            changes.append(
                (name, FeatureChangeType.FEATURE_ADDED, "", current_types[name])
            )
        elif not in_current:
            changes.append(
                (name, FeatureChangeType.FEATURE_REMOVED, previous_types[name], "")
            )
        elif previous_types[name] != current_types[name]:
            changes.append(
                (
                    name,
                    FeatureChangeType.FEATURE_TYPE_CHANGED,
                    previous_types[name],
                    current_types[name],
                )
            )
        elif logic_changed:
            changes.append(
                (
                    name,
                    FeatureChangeType.FEATURE_LOGIC_CHANGED,
                    previous_types[name],
                    current_types[name],
                )
            )
    return changes


def change_type_name(change_type: int) -> str:
    return _CHANGE_TYPE_NAMES.get(change_type, "unspecified")


def upstream_view_names(proto: Any) -> List[str]:
    """Names of feature views this definition depends on.

    Covers derived ``FeatureView``s (``source_views``) and ODFV source
    projections (``sources``). Sorted and de-duplicated so dependency traversal
    is deterministic.
    """
    names: set[str] = set()
    spec = proto.spec
    if not hasattr(spec, "DESCRIPTOR"):
        return []

    field_names = spec.DESCRIPTOR.fields_by_name
    if "source_views" in field_names:
        for source_view in spec.source_views:
            if source_view.name:
                names.add(source_view.name)

    if "sources" in field_names:
        for source in spec.sources.values():
            which = source.WhichOneof("source")
            if which == "feature_view" and source.feature_view.spec.name:
                names.add(source.feature_view.spec.name)
            elif which == "feature_view_projection":
                projection = source.feature_view_projection
                if projection.feature_view_name:
                    names.add(projection.feature_view_name)

    return sorted(names)


def describe_logic(proto: Any) -> Dict[str, Any]:
    """Human-readable description of a definition's value-affecting inputs.

    Stored on change records so a reviewer can see *why* a logic fingerprint
    changed without diffing raw protos.
    """
    spec = proto.spec
    description: Dict[str, Any] = {
        "mode": getattr(spec, "mode", "") or "",
        "upstream_views": upstream_view_names(proto),
    }
    if hasattr(spec, "feature_transformation"):
        transformation = spec.feature_transformation
        which = transformation.WhichOneof("transformation")
        if which == "user_defined_function":
            udf = transformation.user_defined_function
            description["udf_name"] = udf.name
            # ``body_text`` is the UDF source. It is already stored verbatim in
            # the registry, so recording it here adds no new exposure.
            description["udf_body_text"] = udf.body_text
        elif which == "substrait_transformation":
            description["substrait"] = True
    return description
