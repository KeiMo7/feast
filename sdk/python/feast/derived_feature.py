"""Declarative, multi-layer derived features for Feast.

A derived feature chain declares one or more *layers*.  Each layer computes one or
more output features from explicitly named input features, which may come either
from a raw feature view or from an earlier layer in the same chain.

The chain is the single source of truth for three things that used to drift
apart:

* the dependency graph (declared through :class:`DerivationLayer.inputs`),
* the evaluation order (derived from that graph, never from declaration order),
* the provenance record emitted when the chain is evaluated.

Nothing here relies on naming conventions or on the order layers happen to be
declared in.  A dependency exists if and only if it is written in the layer
definition.

The chain is deliberately small and dependency-free (pandas + pyarrow, both of
which Feast already requires).  It can be used on its own, and it is also used
by :class:`feast.feature_store.FeatureStore` to build and order the
``OnDemandFeatureView`` objects that back a chain at registration and serving
time.
"""

from __future__ import annotations

import json
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import pandas as pd

from feast.errors import (
    DerivedFeatureChainError,
    DerivedFeatureCycleError,
    IncompatibleDerivationTypeError,
    MissingDerivationInputError,
)
from feast.field import Field, from_value_type
from feast.transformation.base import Transformation
from feast.types import FeastType
from feast.value_type import ValueType

# Tags used to persist chain-level metadata on the generated OnDemandFeatureViews.
# The dependency graph itself is *not* duplicated here: it lives in the ODFV's
# ``source_feature_view_projections``, exactly the same place runtime resolution
# reads it from.
DERIVED_CHAIN_TAG = "feast:derived_chain"
DERIVED_LAYER_INDEX_TAG = "feast:derived_layer_index"
DERIVED_MISSING_POLICY_TAG = "feast:derived_missing_policy"
DERIVED_DEFAULTS_TAG = "feast:derived_defaults"
DERIVED_CHAIN_VERSION_TAG = "feast:derived_chain_version"

# Bumped only if the on-tag encoding changes.
_DERIVED_CHAIN_FORMAT_VERSION = "1"


class MissingUpstreamPolicy(str, Enum):
    """What a layer does when one of its declared inputs is not available.

    The policy must be chosen explicitly per layer; there is no implicit
    fallback, because silently producing a column of nulls is the exact failure
    mode this feature is meant to remove.
    """

    UNAVAILABLE = "unavailable"
    """Mark every output of the layer unavailable; do not invent values."""

    DEFAULT = "default"
    """Fill missing inputs from the layer's declared ``defaults`` mapping."""

    @classmethod
    def parse(cls, value: Union[str, MissingUpstreamPolicy]) -> MissingUpstreamPolicy:
        if isinstance(value, MissingUpstreamPolicy):
            return value
        normalized = str(value).strip().lower()
        for member in cls:
            if member.value == normalized:
                return member
        raise ValueError(
            f"Unknown missing-upstream policy '{value}'. "
            f"Expected one of: {', '.join(m.value for m in cls)}."
        )


class LayerStatus(str, Enum):
    OK = "ok"
    UNAVAILABLE = "unavailable"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class UpstreamRef:
    """An explicit ``view:feature`` reference to a layer input."""

    view: str
    feature: str

    def __str__(self) -> str:
        return f"{self.view}:{self.feature}"

    @classmethod
    def parse(cls, ref: Union[str, UpstreamRef]) -> UpstreamRef:
        if isinstance(ref, UpstreamRef):
            return ref
        if not isinstance(ref, str) or ":" not in ref:
            raise ValueError(
                f"Derived feature input must be written as 'view:feature', got {ref!r}. "
                "Dependencies are only taken from explicit declarations."
            )
        view, feature = ref.split(":", 1)
        view, feature = view.strip(), feature.strip()
        if not view or not feature:
            raise ValueError(
                f"Derived feature input must name both a view and a feature, got {ref!r}."
            )
        return cls(view=view, feature=feature)


@dataclass
class LayerTrace:
    """Provenance for a single layer.  Carries names and counts, never values."""

    layer: str
    status: LayerStatus
    features: List[str] = field(default_factory=list)
    inputs: List[str] = field(default_factory=list)
    row_count: int = 0
    compute_count: int = 0
    reused: bool = False
    missing_inputs: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer": self.layer,
            "status": self.status.value,
            "features": list(self.features),
            "inputs": list(self.inputs),
            "row_count": self.row_count,
            "compute_count": self.compute_count,
            "reused": self.reused,
            "missing_inputs": list(self.missing_inputs),
            "duration_ms": round(self.duration_ms, 3),
            "error": self.error,
        }


@dataclass
class DerivationResult:
    """Output of evaluating a chain.

    ``data`` always contains the columns that were successfully computed.  When a
    layer is unavailable or the evaluation timed out, the result is *partial*
    rather than an exception: ``completed_layers`` and ``incomplete_layers`` say
    exactly what happened.
    """

    data: pd.DataFrame
    trace: List[LayerTrace]
    completed_layers: List[str] = field(default_factory=list)
    incomplete_layers: List[str] = field(default_factory=list)
    unavailable_features: List[str] = field(default_factory=list)
    computed_features: List[str] = field(default_factory=list)

    @property
    def partial(self) -> bool:
        return bool(self.incomplete_layers)

    def to_record(self) -> Dict[str, Any]:
        """Return the single evaluated row as a dict (or ``{}`` if empty)."""
        if len(self.data) == 0:
            return {}
        return self.data.iloc[0].to_dict()

    def to_dict(self) -> Dict[str, Any]:
        """Provenance export.  Contains feature names and counts only."""
        return {
            "completed_layers": list(self.completed_layers),
            "incomplete_layers": list(self.incomplete_layers),
            "unavailable_features": list(self.unavailable_features),
            "computed_features": list(self.computed_features),
            "layers": [t.to_dict() for t in self.trace],
        }


def _dtype_kind(dtype: Any) -> str:
    """Classify a Feast type into a coarse kind used for compatibility checks."""
    try:
        value_type = dtype.to_value_type()
    except Exception:
        value_type = dtype if isinstance(dtype, ValueType) else None

    numeric = {
        ValueType.INT32,
        ValueType.INT64,
        ValueType.FLOAT,
        ValueType.DOUBLE,
        ValueType.UNIX_TIMESTAMP,
    }
    text = {ValueType.STRING, ValueType.BYTES}
    boolean = {ValueType.BOOL}
    if value_type in numeric:
        return "numeric"
    if value_type in text:
        return "text"
    if value_type in boolean:
        return "bool"
    name = str(value_type or dtype)
    if "LIST" in name or "SET" in name or "ARRAY" in name.upper():
        return "array"
    return "other"


def _types_compatible(upstream: FeastType, expected: FeastType) -> bool:
    """Return True when an upstream value can feed a declared input.

    Numeric widening/narrowing is allowed; crossing between numeric, text and
    boolean is not.  This is the check that rejects feeding a numeric feature to
    a transform that declared a text input.
    """
    a, b = _dtype_kind(upstream), _dtype_kind(expected)
    if a == b:
        return True
    return a == "numeric" and b == "numeric"


class DerivationLayer:
    """One layer of a derived feature chain.

    Args:
        name: Name of the layer.  Becomes the name of the backing
            ``OnDemandFeatureView``.
        inputs: Explicit ``view:feature`` references this layer reads.  At least
            one is required; dependencies are never inferred.
        features: Declared output features of the layer.
        transformation: Feast ``Transformation`` (pandas or python mode).
        mode: ``"pandas"`` or ``"python"``.
        missing_policy: What to do when a declared input is unavailable.
        defaults: Default values used when ``missing_policy`` is ``DEFAULT``.
            Every declared input must have an entry.
        input_schema: Optional declared input types.  Used to reject
            type-incompatible chains at registration time.
        description/tags/entities: Metadata mirrored onto the backing view.
    """

    def __init__(
        self,
        *,
        name: str,
        inputs: Sequence[Union[str, UpstreamRef]],
        features: Sequence[Field],
        transformation: Transformation,
        mode: str = "pandas",
        missing_policy: Union[
            str, MissingUpstreamPolicy
        ] = MissingUpstreamPolicy.UNAVAILABLE,
        defaults: Optional[Mapping[str, Any]] = None,
        input_schema: Optional[Sequence[Field]] = None,
        description: str = "",
        tags: Optional[Mapping[str, str]] = None,
        entities: Optional[Sequence[str]] = None,
        singleton: bool = False,
    ):
        if not name:
            raise ValueError("DerivationLayer requires a non-empty name.")
        if not inputs:
            raise ValueError(
                f"DerivationLayer '{name}' must declare at least one input. "
                "Dependencies cannot be inferred from naming conventions."
            )
        if not features:
            raise ValueError(
                f"DerivationLayer '{name}' must declare at least one output feature."
            )

        self.name = name
        self.inputs: List[UpstreamRef] = [UpstreamRef.parse(i) for i in inputs]
        self.features: List[Field] = list(features)
        self.output_names: List[str] = [f.name for f in self.features]
        if len(set(self.output_names)) != len(self.output_names):
            raise ValueError(
                f"DerivationLayer '{name}' declares duplicate output feature names."
            )
        self.transformation = transformation
        self.mode = mode.lower()
        if self.mode not in ("pandas", "python"):
            raise ValueError(
                f"DerivationLayer '{name}' has unsupported mode '{mode}'. "
                "Multi-layer chains support 'pandas' and 'python'."
            )
        self.missing_policy = MissingUpstreamPolicy.parse(missing_policy)
        self.defaults: Dict[str, Any] = dict(defaults or {})
        self.input_schema: List[Field] = list(input_schema or [])
        self.description = description
        self.tags = dict(tags or {})
        self.entities = list(entities or [])
        self.singleton = singleton

        if self.missing_policy is MissingUpstreamPolicy.DEFAULT:
            missing_defaults = [
                str(ref) for ref in self.inputs if ref.feature not in self.defaults
            ]
            if missing_defaults:
                raise ValueError(
                    f"DerivationLayer '{name}' uses missing_policy='default' but has "
                    f"no default for inputs: {sorted(missing_defaults)}."
                )

        self._validate_input_schema()

    def _validate_input_schema(self) -> None:
        declared = {f.name: f for f in self.input_schema}
        input_features = {ref.feature for ref in self.inputs}
        unknown = sorted(set(declared) - input_features)
        if unknown:
            raise ValueError(
                f"DerivationLayer '{self.name}' declares input types for features it "
                f"does not depend on: {unknown}. Add them to 'inputs' or remove them "
                "from 'input_schema'."
            )
        conflicts = defaultdict(set)
        for ref in self.inputs:
            conflicts[ref.feature].add(ref.view)
        ambiguous = sorted(name for name, views in conflicts.items() if len(views) > 1)
        if ambiguous:
            raise DerivedFeatureChainError(
                f"DerivationLayer '{self.name}' reads feature(s) {ambiguous} from more "
                "than one upstream view. Rename or split the layer so each input "
                "feature comes from exactly one declared upstream."
            )

    @property
    def input_names(self) -> List[str]:
        return [ref.feature for ref in self.inputs]

    @property
    def upstream_views(self) -> List[str]:
        return list(OrderedDict.fromkeys(ref.view for ref in self.inputs))

    def dependency_text(self) -> str:
        produced = ", ".join(self.output_names)
        consumed = ", ".join(str(ref) for ref in self.inputs)
        return f"{self.name} [{produced}] <- {consumed}"

    def input_type_expectations(self) -> Dict[str, FeastType]:
        return {f.name: f.dtype for f in self.input_schema}


class DerivedFeatureChain:
    """A validated, evaluable multi-layer derived feature graph.

    Build one from :class:`DerivationLayer` objects and pass it either directly to
    :meth:`evaluate_batch` / :meth:`evaluate_single`, or to
    :meth:`feast.feature_store.FeatureStore.apply_derived_feature_chain` to
    register it as a set of linked ``OnDemandFeatureView`` objects.

    Args:
        name: Chain name, used for metadata and provenance.
        layers: Layers in any order.  Evaluation order is derived from the
            declared dependencies.
        description/tags: Metadata mirrored onto the generated views.
    """

    def __init__(
        self,
        *,
        name: str,
        layers: Sequence[DerivationLayer],
        description: str = "",
        tags: Optional[Mapping[str, str]] = None,
        entities: Optional[Sequence[Any]] = None,
    ):
        if not name:
            raise ValueError("DerivedFeatureChain requires a non-empty name.")
        if not layers:
            raise ValueError("DerivedFeatureChain requires at least one layer.")

        self.name = name
        self.description = description
        self.tags = dict(tags or {})
        self.layers: List[DerivationLayer] = list(layers)
        # Entity objects inherited by every generated ODFV.  Layers may override
        # with their own names; supplying real Entity objects here means the
        # backing views participate in normal registration/inference.
        self.entities: List[Any] = list(entities or [])

        self._layer_index: "OrderedDict[str, DerivationLayer]" = OrderedDict()
        for layer in self.layers:
            if layer.name in self._layer_index:
                raise DerivedFeatureChainError(
                    f"Derived feature chain '{name}' declares layer '{layer.name}' "
                    "more than once."
                )
            self._layer_index[layer.name] = layer

        # Validated eagerly so that an invalid chain cannot be built, let alone
        # registered.  Registration and runtime both consume this same object.
        self._topological: List[str] = self._compute_topological_order()

    # ------------------------------------------------------------------ graph

    def get_layer(self, name: str) -> DerivationLayer:
        try:
            return self._layer_index[name]
        except KeyError:
            raise KeyError(f"Unknown layer '{name}' in chain '{self.name}'")

    def layer_names(self) -> List[str]:
        return list(self._layer_index.keys())

    def output_features(self, layer: Optional[str] = None) -> List[str]:
        if layer is not None:
            return list(self.get_layer(layer).output_names)
        out: List[str] = []
        for name in self._topological:
            out.extend(self.get_layer(name).output_names)
        return out

    def _is_chain_layer(self, view_name: str) -> bool:
        return view_name in self._layer_index

    def _compute_topological_order(self) -> List[str]:
        """Order layers so every dependency precedes its dependents.

        Declaration order never matters.  Cycles are rejected with every node on
        the cycle named so the error is actionable.
        """
        # Only intra-chain edges participate in ordering; refs to raw feature
        # views are external inputs.
        dependencies: Dict[str, Set[str]] = {name: set() for name in self._layer_index}
        for layer in self.layers:
            for ref in layer.inputs:
                if self._is_chain_layer(ref.view):
                    dependencies[layer.name].add(ref.view)

        # Detect self-cycles and report the whole cycle path.
        for name, deps in dependencies.items():
            if name in deps:
                raise DerivedFeatureCycleError(
                    f"Derived feature chain '{self.name}' has a cycle: {name} -> {name}."
                )

        order: List[str] = []
        # Kahn's algorithm, seeded in declaration order only to keep output
        # deterministic for independent layers.  It plays no part in resolving
        # which layer must run before which.
        indegree = {name: len(deps) for name, deps in dependencies.items()}
        dependents: Dict[str, Set[str]] = defaultdict(set)
        for name, deps in dependencies.items():
            for dep in deps:
                dependents[dep].add(name)

        ready = [name for name in self._layer_index if indegree[name] == 0]
        while ready:
            name = ready.pop(0)
            order.append(name)
            for dependent in sorted(dependents[name]):
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)

        if len(order) != len(self._layer_index):
            cycle = self._find_cycle(dependencies)
            raise DerivedFeatureCycleError(
                f"Derived feature chain '{self.name}' contains a dependency cycle "
                f"involving: {cycle}. Break the cycle by removing one of the "
                "declared inputs."
            )

        return order

    def _find_cycle(self, dependencies: Mapping[str, Set[str]]) -> str:
        """Return one concrete cycle as ``a -> b -> ... -> a``.

        Depth-first search over the intra-chain dependency edges; any back edge
        closes a cycle, and the stack slice from the repeated node is exactly the
        set of layers on that cycle.
        """
        WHITE, GREY, BLACK = 0, 1, 2
        color = {name: WHITE for name in dependencies}
        parent: Dict[str, str] = {}
        cycle: List[str] = []

        def dfs(node: str) -> bool:
            color[node] = GREY
            for neighbor in sorted(dependencies.get(node, ())):
                if color.get(neighbor) == GREY:
                    # ``neighbor`` is an ancestor on the current DFS stack.  Walk
                    # parent pointers from ``node`` back to it; reversing gives
                    # the cycle in dependency direction, then close the loop.
                    path = [node]
                    current = node
                    while current != neighbor and current in parent:
                        current = parent[current]
                        path.append(current)
                    path.reverse()
                    cycle.extend(path + [neighbor])
                    return True
                if color.get(neighbor) == WHITE:
                    parent[neighbor] = node
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for name in sorted(dependencies):
            if color[name] == WHITE and dfs(name):
                break
        return " -> ".join(cycle) if cycle else ", ".join(sorted(dependencies))

    def topological_order(self) -> List[str]:
        """Layer names in dependency order, independent of declaration order."""
        return list(self._topological)

    def downstream_of(self, view_name: str) -> List[str]:
        """Layer names that (transitively) consume ``view_name``, ordered."""
        dependents: Dict[str, Set[str]] = defaultdict(set)
        for layer in self.layers:
            for ref in layer.inputs:
                if self._is_chain_layer(ref.view):
                    dependents[ref.view].add(layer.name)
        result: List[str] = []
        stack = list(dependents.get(view_name, ()))
        seen: Set[str] = set()
        while stack:
            node = stack.pop()
            if node in seen:
                continue
            seen.add(node)
            result.append(node)
            stack.extend(dependents.get(node, ()))
        return [n for n in self._topological if n in seen]

    def reference_counts(self) -> Dict[str, int]:
        """How many downstream layer inputs reference each layer."""
        counts: Dict[str, int] = {name: 0 for name in self._layer_index}
        for layer in self.layers:
            for ref in layer.inputs:
                if self._is_chain_layer(ref.view):
                    counts[ref.view] += 1
        return counts

    # -------------------------------------------------------------- validation

    def validate(
        self,
        available_types: Optional[Mapping[str, Mapping[str, FeastType]]] = None,
        known_views: Optional[Iterable[str]] = None,
    ) -> None:
        """Validate the chain, raising on any problem.

        This is the *only* validation routine; ``FeatureStore.apply`` calls it
        before writing anything to the registry, so registration and runtime can
        never disagree about whether a chain is valid.

        Args:
            available_types: Mapping of ``view_name -> {feature_name: FeastType}``
                for external (non-chain) upstream views.  Enables type checking
                and detection of unknown features.
            known_views: Names of views that exist, used to detect references to
                views that were never declared.  Chain layers are always known.
                When both this and ``available_types`` are omitted, external
                upstream existence is not checked (only intra-chain structure is).
        """
        known = set(self._layer_index)
        external_check_enabled = known_views is not None or available_types is not None
        if known_views is not None:
            known |= set(known_views)
        if available_types:
            known |= set(available_types)

        # Missing upstreams: report the missing name and every downstream that
        # asked for it.
        missing_by_upstream: Dict[str, List[str]] = defaultdict(list)
        if external_check_enabled:
            for layer in self.layers:
                for ref in layer.inputs:
                    if ref.view in known:
                        continue
                    missing_by_upstream[str(ref)].append(layer.name)

        if missing_by_upstream:
            details = "; ".join(
                f"'{ref}' referenced by {sorted(set(downstreams))}"
                for ref, downstreams in sorted(missing_by_upstream.items())
            )
            raise MissingDerivationInputError(
                f"Derived feature chain '{self.name}' declares inputs whose upstream "
                f"does not exist: {details}."
            )

        # Output collisions: two layers (or a layer and a raw feature) may not
        # produce the same feature name, otherwise downstream refs are ambiguous.
        produced: Dict[str, str] = {}
        collisions: List[str] = []
        for layer in self.layers:
            for feature in layer.output_names:
                if feature in produced:
                    collisions.append(
                        f"'{feature}' produced by both '{produced[feature]}' and "
                        f"'{layer.name}'"
                    )
                produced[feature] = layer.name
        if collisions:
            raise DerivedFeatureChainError(
                f"Derived feature chain '{self.name}' has duplicate output "
                f"features: {collisions}."
            )

        # Type compatibility, checked against declared upstream types.
        upstream_types: Dict[Tuple[str, str], FeastType] = {}
        if available_types:
            for view_name, feature_map in available_types.items():
                for feature_name, dtype in feature_map.items():
                    upstream_types[(view_name, feature_name)] = dtype

        # Chain-layer outputs are themselves typed, so a layer can check against
        # an earlier layer even when no external map is supplied.
        for layer_name in self._topological:
            layer = self.get_layer(layer_name)
            for output_field in layer.features:
                upstream_types[(layer_name, output_field.name)] = output_field.dtype

        # A reference into another layer must name a feature that layer declares;
        # otherwise the dependency is unresolvable at runtime and should be
        # rejected now, at registration.
        bad_internal_refs: List[str] = []
        bad_external_refs: List[str] = []
        for layer in self.layers:
            for ref in layer.inputs:
                if self._is_chain_layer(ref.view):
                    upstream_layer = self.get_layer(ref.view)
                    if ref.feature not in upstream_layer.output_names:
                        bad_internal_refs.append(
                            f"'{ref}' (layer '{ref.view}' declares "
                            f"{sorted(upstream_layer.output_names)})"
                        )
                elif available_types is not None and ref.view in available_types:
                    # The view is known *and* its schema is known: the feature
                    # must exist there.
                    if ref.feature not in available_types[ref.view]:
                        bad_external_refs.append(
                            f"'{ref}' (view '{ref.view}' declares "
                            f"{sorted(available_types[ref.view])})"
                        )
        if bad_internal_refs:
            raise MissingDerivationInputError(
                f"Derived feature chain '{self.name}' references undeclared "
                f"upstream features: {'; '.join(bad_internal_refs)}."
            )
        if bad_external_refs:
            raise MissingDerivationInputError(
                f"Derived feature chain '{self.name}' references features that its "
                f"upstream views do not provide: {'; '.join(bad_external_refs)}."
            )

        incompatible: List[str] = []
        for layer in self.layers:
            for expected_field in layer.input_schema:
                upstream_type = upstream_types.get(
                    (
                        next(
                            ref.view
                            for ref in layer.inputs
                            if ref.feature == expected_field.name
                        ),
                        expected_field.name,
                    )
                )
                if upstream_type is None:
                    # Unknown external type: cannot disprove compatibility, so
                    # defer to runtime rather than rejecting a valid chain.
                    continue
                if not _types_compatible(upstream_type, expected_field.dtype):
                    incompatible.append(
                        f"layer '{layer.name}' input '{expected_field.name}' expects "
                        f"{expected_field.dtype} but upstream provides {upstream_type}"
                    )
        if incompatible:
            raise IncompatibleDerivationTypeError(
                f"Derived feature chain '{self.name}' has incompatible input types: "
                + "; ".join(incompatible)
                + "."
            )

    # -------------------------------------------------------------- evaluation

    def _input_frames(
        self,
        layer: DerivationLayer,
        base: pd.DataFrame,
        results: Mapping[str, pd.DataFrame],
        unavailable_layers: Set[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Assemble the layer's input frame and list of missing inputs."""
        columns: Dict[str, pd.Series] = {}
        missing: List[str] = []
        num_rows = len(base)

        for ref in layer.inputs:
            series: Optional[pd.Series] = None
            if self._is_chain_layer(ref.view):
                upstream = results.get(ref.view)
                if ref.view in unavailable_layers or upstream is None:
                    series = None
                elif ref.feature in upstream.columns:
                    series = upstream[ref.feature]
                elif ref.feature in base.columns:
                    # A downstream layer may also read a raw feature with the
                    # same name that the upstream layer happened to echo.
                    series = base[ref.feature]
            else:
                for candidate in (ref.feature, f"{ref.view}__{ref.feature}"):
                    if candidate in base.columns:
                        series = base[candidate]
                        break

            if series is None:
                missing.append(str(ref))
                continue
            columns[ref.feature] = series.reset_index(drop=True)

        if not columns:
            return pd.DataFrame(index=range(num_rows)), missing

        frame = pd.DataFrame(columns)
        # A declared input that is present but entirely null is treated as
        # missing so that a broken upstream is reported instead of silently
        # producing an all-null output column.
        all_null = [name for name in columns if frame[name].isna().all()]
        missing.extend(
            str(ref)
            for ref in layer.inputs
            if ref.feature in all_null and str(ref) not in missing
        )
        return frame, missing

    def _apply_defaults(
        self, frame: pd.DataFrame, layer: DerivationLayer, missing: List[str]
    ) -> pd.DataFrame:
        """Replace every missing input with its declared default.

        A declared input counts as missing when the column is absent *or* every
        value in it is null; both are filled, so a downstream layer never sees a
        silently all-null column where a default was promised.
        """
        frame = frame.copy()
        missing_features = {UpstreamRef.parse(ref).feature for ref in missing}
        for feature_name in missing_features:
            default_value = self.defaults_for(layer)[feature_name]
            if feature_name in frame.columns:
                frame[feature_name] = frame[feature_name].where(
                    frame[feature_name].notna(), other=default_value
                )
            else:
                frame[feature_name] = default_value
        return frame

    @staticmethod
    def defaults_for(layer: DerivationLayer) -> Dict[str, Any]:
        return dict(layer.defaults)

    def _run_layer(
        self,
        layer: DerivationLayer,
        frame: pd.DataFrame,
        num_rows: int,
    ) -> pd.DataFrame:
        if layer.mode == "pandas":
            output = layer.transformation.transform(frame)
        elif layer.singleton and hasattr(layer.transformation, "transform_singleton"):
            output = layer.transformation.transform_singleton(
                frame.to_dict(orient="list")
            )
        else:
            output = layer.transformation.transform(frame.to_dict(orient="list"))
        if not isinstance(output, pd.DataFrame):
            output = pd.DataFrame(output)
        return output.reset_index(drop=True)

    def evaluate_batch(
        self,
        base: pd.DataFrame,
        requested: Optional[Sequence[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> DerivationResult:
        """Evaluate the chain against a batch dataframe.

        Args:
            base: DataFrame containing raw upstream feature columns (and request
                data).  Column names may be either short (``conv_rate``) or
                fully qualified (``driver_stats__conv_rate``).
            requested: Optional subset of output features to return.  Dependencies
                of requested features are still computed.
            timeout_seconds: Optional wall-clock budget.  On expiry, evaluation
                stops and the result reports the computed columns plus the
                unfinished layers instead of failing the whole batch.

        Returns:
            A :class:`DerivationResult`.  Every layer that ran appears in
            ``trace`` with names and counts only.
        """
        base = base.reset_index(drop=True)
        num_rows = len(base)

        requested_set = set(requested) if requested is not None else None
        needed_layers = self._layers_needed_for(requested_set)

        results: Dict[str, pd.DataFrame] = {}
        unavailable_layers: Set[str] = set()
        traces: List[LayerTrace] = []
        compute_counts: Dict[str, int] = defaultdict(int)
        trace_by_layer: Dict[str, LayerTrace] = {}

        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )

        for layer_name in self._topological:
            layer = self.get_layer(layer_name)
            if needed_layers is not None and layer_name not in needed_layers:
                trace = LayerTrace(
                    layer=layer_name,
                    status=LayerStatus.SKIPPED,
                    features=list(layer.output_names),
                    inputs=[str(r) for r in layer.inputs],
                    row_count=num_rows,
                )
                traces.append(trace)
                trace_by_layer[layer_name] = trace
                continue

            if deadline is not None and time.monotonic() > deadline:
                trace = LayerTrace(
                    layer=layer_name,
                    status=LayerStatus.TIMEOUT,
                    features=list(layer.output_names),
                    inputs=[str(r) for r in layer.inputs],
                    row_count=num_rows,
                )
                traces.append(trace)
                trace_by_layer[layer_name] = trace
                unavailable_layers.add(layer_name)
                continue

            started = time.monotonic()
            frame, missing = self._input_frames(
                layer, base, results, unavailable_layers
            )

            if missing:
                if layer.missing_policy is MissingUpstreamPolicy.DEFAULT:
                    frame = self._apply_defaults(frame, layer, missing)
                else:
                    trace = LayerTrace(
                        layer=layer_name,
                        status=LayerStatus.UNAVAILABLE,
                        features=list(layer.output_names),
                        inputs=[str(r) for r in layer.inputs],
                        row_count=num_rows,
                        missing_inputs=sorted(missing),
                    )
                    traces.append(trace)
                    trace_by_layer[layer_name] = trace
                    unavailable_layers.add(layer_name)
                    continue

            compute_counts[layer_name] += 1
            try:
                output = self._run_layer(layer, frame, num_rows)
            except Exception as exc:  # noqa: BLE001 - surfaced in trace, not raised
                trace = LayerTrace(
                    layer=layer_name,
                    status=LayerStatus.FAILED,
                    features=list(layer.output_names),
                    inputs=[str(r) for r in layer.inputs],
                    row_count=num_rows,
                    duration_ms=(time.monotonic() - started) * 1000,
                    error=f"{type(exc).__name__}: {exc}",
                )
                traces.append(trace)
                trace_by_layer[layer_name] = trace
                unavailable_layers.add(layer_name)
                continue

            # Only the declared features are exposed; anything else the UDF
            # returned is dropped so downstream layers see a stable contract.
            missing_outputs = [
                name for name in layer.output_names if name not in output.columns
            ]
            if missing_outputs:
                trace = LayerTrace(
                    layer=layer_name,
                    status=LayerStatus.FAILED,
                    features=list(layer.output_names),
                    inputs=[str(r) for r in layer.inputs],
                    row_count=num_rows,
                    duration_ms=(time.monotonic() - started) * 1000,
                    error=(
                        "transformation did not produce declared feature(s): "
                        + ", ".join(missing_outputs)
                    ),
                )
                traces.append(trace)
                trace_by_layer[layer_name] = trace
                unavailable_layers.add(layer_name)
                continue

            results[layer_name] = output[layer.output_names]
            trace = LayerTrace(
                layer=layer_name,
                status=LayerStatus.OK,
                features=list(layer.output_names),
                inputs=[str(r) for r in layer.inputs],
                row_count=num_rows,
                duration_ms=(time.monotonic() - started) * 1000,
            )
            traces.append(trace)
            trace_by_layer[layer_name] = trace

        # A layer that was computed once but referenced by several downstream
        # layers is a reuse; record that explicitly in the trace.
        ref_counts = self.reference_counts()
        for layer_name, trace in trace_by_layer.items():
            trace.compute_count = compute_counts.get(layer_name, 0)
            trace.reused = ref_counts.get(layer_name, 0) > 1 and trace.status in (
                LayerStatus.OK,
                LayerStatus.UNAVAILABLE,
                LayerStatus.FAILED,
            )

        computed_features: List[str] = []
        unavailable_features: List[str] = []
        completed: List[str] = []
        incomplete: List[str] = []
        for layer_name in self._topological:
            trace = trace_by_layer[layer_name]
            if trace.status is LayerStatus.SKIPPED:
                continue
            if trace.status is LayerStatus.OK:
                completed.append(layer_name)
                computed_features.extend(trace.features)
            else:
                incomplete.append(layer_name)
                unavailable_features.extend(trace.features)

        data = pd.DataFrame(index=range(num_rows))
        for layer_name in self._topological:
            if layer_name in results:
                layer_output = results[layer_name]
                for column in layer_output.columns:
                    data[column] = layer_output[column].reset_index(drop=True)

        if requested_set is not None:
            data = data[[c for c in data.columns if c in requested_set]]

        # Keep the requested order stable and predictable.
        if requested is not None:
            ordered_columns = [c for c in requested if c in data.columns]
            if ordered_columns:
                data = data[ordered_columns]

        return DerivationResult(
            data=data,
            trace=traces,
            completed_layers=completed,
            incomplete_layers=incomplete,
            unavailable_features=unavailable_features,
            computed_features=computed_features,
        )

    def evaluate_single(
        self,
        row: Mapping[str, Any],
        requested: Optional[Sequence[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> DerivationResult:
        """Evaluate the chain for a single row.

        Runs the exact same batch path over a one-row frame, so a single
        evaluation can never disagree with a batch evaluation.  The returned
        ``data`` has exactly one row; use :meth:`DerivationResult.to_record` to
        get it as a dict.
        """
        frame = pd.DataFrame([dict(row)])
        return self.evaluate_batch(
            frame, requested=requested, timeout_seconds=timeout_seconds
        )

    def evaluate(
        self,
        base: Union[pd.DataFrame, Mapping[str, Any]],
        requested: Optional[Sequence[str]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> DerivationResult:
        """Evaluate either a batch frame or a single row mapping."""
        if isinstance(base, pd.DataFrame):
            return self.evaluate_batch(
                base, requested=requested, timeout_seconds=timeout_seconds
            )
        return self.evaluate_single(
            base, requested=requested, timeout_seconds=timeout_seconds
        )

    def _layers_needed_for(self, requested: Optional[Set[str]]) -> Optional[Set[str]]:
        """Transitive closure of layers required to produce ``requested``."""
        if requested is None:
            return None
        output_to_layer = {
            feature: layer.name
            for layer in self.layers
            for feature in layer.output_names
        }
        needed: Set[str] = set()
        stack = [output_to_layer[f] for f in requested if f in output_to_layer]
        while stack:
            name = stack.pop()
            if name in needed:
                continue
            needed.add(name)
            for ref in self.get_layer(name).inputs:
                if self._is_chain_layer(ref.view):
                    stack.append(ref.view)
        return needed

    # -------------------------------------------------------------- provenance

    def dependency_text(self) -> str:
        """Readable, review-friendly export of the dependency graph."""
        lines = [f"derived feature chain: {self.name}"]
        if self.description:
            lines.append(f"description: {self.description}")
        lines.append("evaluation order:")
        for index, layer_name in enumerate(self._topological, start=1):
            lines.append(f"  {index}. {self.get_layer(layer_name).dependency_text()}")
        lines.append("edges:")
        for layer in self.layers:
            for ref in layer.inputs:
                lines.append(f"  {ref.view} -> {layer.name} ({ref.feature})")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Structural export used for tests and provenance review."""
        return {
            "name": self.name,
            "description": self.description,
            "layers": [
                {
                    "name": layer.name,
                    "index": index,
                    "inputs": [str(ref) for ref in layer.inputs],
                    "outputs": list(layer.output_names),
                    "missing_policy": layer.missing_policy.value,
                    "mode": layer.mode,
                }
                for index, layer in enumerate(self.layers)
            ],
            "evaluation_order": self.topological_order(),
            "edges": [
                {"from": ref.view, "to": layer.name, "feature": ref.feature}
                for layer in self.layers
                for ref in layer.inputs
            ],
        }

    # ------------------------------------------------------------ ODFV bridge

    def to_feature_views(
        self,
        type_resolver: Optional[Callable[[str, str], Optional[Field]]] = None,
        entity_resolver: Optional[Callable[[str], Any]] = None,
    ):
        """Build one ``OnDemandFeatureView`` per layer.

        The dependency graph is encoded in each view's ``sources`` (view name +
        the exact features consumed), which is the same structure runtime
        resolution reads.  Chain-level metadata that is not part of the graph
        (missing-input policy, defaults, ordinal) is stored in tags.

        Args:
            type_resolver: Optional ``(view_name, feature_name) -> Field`` used to
                materialise typed projections for external upstreams.
            entity_resolver: Optional ``entity_name -> Entity`` used to resolve
                named entities declared on a layer.  When omitted, only entities
                supplied to the chain constructor are attached.
        """
        from feast.feature_view_projection import FeatureViewProjection
        from feast.on_demand_feature_view import OnDemandFeatureView

        chain_entities = {getattr(e, "name", None): e for e in self.entities}
        views = []
        for index, layer in enumerate(self.layers):
            projections: List[Any] = []
            for view_name in layer.upstream_views:
                feature_fields = self._input_fields_for(layer, view_name, type_resolver)
                projections.append(
                    FeatureViewProjection(
                        name=view_name,
                        name_alias=None,
                        desired_features=[],
                        features=feature_fields,
                    )
                )

            tags = dict(layer.tags)
            tags.update(
                {
                    DERIVED_CHAIN_TAG: self.name,
                    DERIVED_LAYER_INDEX_TAG: str(index),
                    DERIVED_MISSING_POLICY_TAG: layer.missing_policy.value,
                    DERIVED_CHAIN_VERSION_TAG: _DERIVED_CHAIN_FORMAT_VERSION,
                }
            )
            if layer.defaults:
                tags[DERIVED_DEFAULTS_TAG] = json.dumps(
                    layer.defaults, sort_keys=True, default=str
                )

            layer_entities = self._resolve_entities(
                layer, chain_entities, entity_resolver
            )
            odfv = OnDemandFeatureView(
                name=layer.name,
                sources=projections,
                schema=list(layer.features),
                feature_transformation=layer.transformation,
                mode=layer.mode,
                description=layer.description or self.description,
                tags=tags,
                entities=layer_entities or None,
                singleton=layer.singleton,
            )
            views.append(odfv)
        return views

    def _resolve_entities(
        self,
        layer: DerivationLayer,
        chain_entities: Mapping[Optional[str], Any],
        entity_resolver: Optional[Callable[[str], Any]],
    ) -> List[Any]:
        if not layer.entities:
            return list(self.entities)
        resolved: List[Any] = []
        for entity_name in layer.entities:
            entity = None
            if entity_resolver is not None:
                entity = entity_resolver(entity_name)
            if entity is None:
                entity = chain_entities.get(entity_name)
            if entity is None:
                raise DerivedFeatureChainError(
                    f"Layer '{layer.name}' references entity '{entity_name}', which "
                    "could not be resolved. Pass it to the chain's 'entities' or "
                    "provide an entity_resolver."
                )
            resolved.append(entity)
        return resolved

    def _input_fields_for(
        self,
        layer: DerivationLayer,
        view_name: str,
        type_resolver: Optional[Callable[[str, str], Optional[Field]]],
    ) -> List[Field]:
        fields: List[Field] = []
        expected = layer.input_type_expectations()
        for ref in layer.inputs:
            if ref.view != view_name:
                continue
            if self._is_chain_layer(ref.view):
                upstream_layer = self.get_layer(ref.view)
                match = next(
                    (f for f in upstream_layer.features if f.name == ref.feature),
                    None,
                )
                if match is None:
                    raise MissingDerivationInputError(
                        f"Layer '{layer.name}' reads '{ref}' but layer '{ref.view}' "
                        f"does not declare feature '{ref.feature}'."
                    )
                fields.append(match)
                continue
            resolved = None
            if type_resolver is not None:
                resolved = type_resolver(ref.view, ref.feature)
            if resolved is not None:
                fields.append(resolved)
                continue
            if ref.feature in expected:
                fields.append(Field(name=ref.feature, dtype=expected[ref.feature]))
            else:
                # Unknown type: use Feast's own UNKNOWN FeastType so the
                # placeholder behaves exactly like any other declared dtype.
                fields.append(
                    Field(name=ref.feature, dtype=from_value_type(ValueType.UNKNOWN))
                )
        return fields

    @classmethod
    def from_feature_views(
        cls, views: Sequence[Any], chain_name: Optional[str] = None
    ) -> "DerivedFeatureChain":
        """Reconstruct a chain from registered ODFVs.

        Used by serving and by lineage export so they read the same declaration
        that registration validated.
        """
        tagged = [
            view for view in views if DERIVED_CHAIN_TAG in getattr(view, "tags", {})
        ]
        if not tagged:
            raise DerivedFeatureChainError(
                "No derived feature chain metadata found on the supplied feature views."
            )
        resolved_name = chain_name or tagged[0].tags[DERIVED_CHAIN_TAG]
        tagged = [
            view for view in tagged if view.tags.get(DERIVED_CHAIN_TAG) == resolved_name
        ]
        tagged.sort(key=lambda v: int(v.tags.get(DERIVED_LAYER_INDEX_TAG, "0")))
        layers: List[DerivationLayer] = []
        for view in tagged:
            inputs: List[str] = []
            for projection in view.source_feature_view_projections.values():
                for feature in projection.features:
                    inputs.append(f"{projection.name}:{feature.name}")
            defaults: Dict[str, Any] = {}
            if view.tags.get(DERIVED_DEFAULTS_TAG):
                defaults = json.loads(view.tags[DERIVED_DEFAULTS_TAG])
            layers.append(
                DerivationLayer(
                    name=view.name,
                    inputs=inputs,
                    features=list(view.features),
                    transformation=view.feature_transformation,
                    mode=view.mode,
                    missing_policy=view.tags.get(
                        DERIVED_MISSING_POLICY_TAG,
                        MissingUpstreamPolicy.UNAVAILABLE.value,
                    ),
                    defaults=defaults,
                    description=view.description,
                    tags={
                        k: v
                        for k, v in view.tags.items()
                        if not k.startswith("feast:derived")
                    },
                    entities=list(getattr(view, "entities", []) or []),
                    singleton=bool(getattr(view, "singleton", False)),
                )
            )
        return cls(name=resolved_name, layers=layers)


def _safe_projections(view: Any) -> Any:
    """Return a view's source projections as a mapping, tolerating test doubles.

    Production calls always receive a real ``OnDemandFeatureView``.  Some tests
    pass ``Mock`` objects; a ``Mock`` attribute is truthy but not a mapping, so a
    plain truthiness check is not enough.  Only accept an object that exposes a
    real, iterable ``values()``.
    """
    projections = getattr(view, "source_feature_view_projections", None)
    values = getattr(projections, "values", None)
    if not callable(values):
        return {}
    try:
        materialized = list(values())
    except TypeError:
        return {}
    # A ``Mock`` returns another ``Mock`` from ``values()``; that "list" has no
    # recognizable projection shape, so treat it as no sources.
    if not all(hasattr(item, "name") for item in materialized):
        return {}
    return projections


def order_feature_views_by_dependencies(
    views: Sequence[Any],
) -> List[Any]:
    """Stable topological ordering of ODFVs based on their declared sources.

    Views that do not depend on each other keep their input order, so the
    single-layer case behaves exactly as before.  This is intentionally generic
    and does not require chain metadata: any ODFV whose source is another ODFV
    is ordered after it.
    """
    by_name = {view.name: view for view in views}
    result: List[Any] = []
    visiting: Set[str] = set()
    visited: Set[str] = set()

    def visit(view: Any) -> None:
        name = view.name
        if name in visited:
            return
        if name in visiting:
            raise DerivedFeatureCycleError(
                f"On-demand feature views form a cycle at '{name}'."
            )
        visiting.add(name)
        projections = _safe_projections(view)
        for projection in projections.values():
            upstream = by_name.get(projection.name)
            if upstream is not None:
                visit(upstream)
        visiting.discard(name)
        visited.add(name)
        result.append(view)

    for view in views:
        visit(view)
    return result


def required_upstream_outputs(
    views: Sequence[Any],
) -> Dict[str, FrozenSet[str]]:
    """Map ODFV name -> feature names it must expose for downstream ODFVs.

    Used by serving to decide which intermediate outputs to compute and keep
    available, without duplicating the graph: the answer is read straight off
    the declared sources.
    """
    required: Dict[str, Set[str]] = defaultdict(set)
    by_name = {view.name: view for view in views}
    for view in views:
        projections = _safe_projections(view)
        for projection in projections.values():
            if projection.name in by_name and projection.name != view.name:
                for feature in projection.features:
                    required[projection.name].add(feature.name)
    return {name: frozenset(features) for name, features in required.items()}


def transitive_on_demand_dependencies(views: Sequence[Any]) -> List[Any]:
    """Expand a list of ODFVs with all ODFV sources they transitively need."""
    by_name = {view.name: view for view in views}
    result: "OrderedDict[str, Any]" = OrderedDict()
    stack = list(views)
    while stack:
        view = stack.pop()
        if view.name in result:
            continue
        result[view.name] = view
        for projection in _safe_projections(view).values():
            upstream = by_name.get(projection.name)
            if upstream is not None and upstream.name not in result:
                stack.append(upstream)
    return order_feature_views_by_dependencies(list(result.values()))
