"""Immutable model contracts for native binding variant dispatch.

The models in this module describe runtime guard checks and compiled binding
variants without embedding Python source expressions. Guard payloads are
structured so later code generation can render checks from known fields instead
of evaluating arbitrary text from analysis results.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

from atoll.models import Backend, BindingTarget, SymbolId

GuardKind = Literal[
    "exact-type",
    "integer-domain",
    "direct-field",
    "callable-code-identity",
    "buffer-layout",
]
CompiledVariantKind = Literal["safe-int32", "safe-int64", "generic"]
DispatchTargetKind = Literal["compiled", "python-fallback"]
IntegerBitWidth = Literal[32, 64]

_DIGEST_SIZE = 16
_INT32_WIDTH: Literal[32] = 32
_INT64_WIDTH: Literal[64] = 64
_INT32_MIN = -(2**31)
_INT32_MAX = 2**31 - 1
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_VARIANT_KIND_ORDER: dict[CompiledVariantKind, int] = {
    "safe-int32": 0,
    "safe-int64": 1,
    "generic": 2,
}
_BACKEND_ORDER: dict[Backend, int] = {"mypyc": 0, "cython": 1}


def _require_identifier(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be non-empty")


def _canonical_text(parts: tuple[str, ...]) -> str:
    """Return an unambiguous stable text form for digest input.

    Args:
        parts: Ordered scalar fields to include in the serialized form.

    Returns:
        str: Length-prefixed canonical representation of the supplied fields.
    """
    return "".join(f"{len(part)}:{part};" for part in parts)


def _stable_digest(prefix: str, serialization: str) -> str:
    digest = hashlib.blake2b(digest_size=_DIGEST_SIZE)
    digest.update(serialization.encode("utf-8"))
    return f"{prefix}-{digest.hexdigest()}"


def _binding_serialization(binding: BindingTarget) -> str:
    """Return the runtime-destination identity for one compiled binding.

    Args:
        binding: Descriptor-aware source binding promised by a native variant.

    Returns:
        str: Canonical source, owner, descriptor, execution, and target-owner identity.
    """
    return _canonical_text(
        (
            binding.source.stable_id,
            binding.kind,
            binding.owner_class or "",
            binding.target_owner_class or "",
            binding.execution_kind,
        )
    )


@dataclass(frozen=True, slots=True)
class ExactTypeGuardPayload:
    """Payload for a guard that requires an object's runtime type to match exactly.

    The payload names a subject captured by the dispatcher and the fully
    qualified type it must have. It deliberately has no expression field and no
    subclass flag because a subclass-permitting check is not an exact-type guard.

    Attributes:
        subject: Dispatcher-local value name whose type is checked.
        type_module: Importable module that owns the expected type.
        type_qualname: Qualified type name inside `type_module`.
    """

    subject: str
    type_module: str
    type_qualname: str

    def __post_init__(self) -> None:
        """Reject missing subject or expected type identity.

        Raises:
            ValueError: If any identity field is blank.
        """
        _require_identifier(self.subject, "subject")
        _require_identifier(self.type_module, "type_module")
        _require_identifier(self.type_qualname, "type_qualname")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard-payload serialization.

        Returns:
            str: Canonical text containing the guard kind and expected type identity.
        """
        return _canonical_text(("exact-type", self.subject, self.type_module, self.type_qualname))


@dataclass(frozen=True, slots=True)
class IntegerDomainGuardPayload:
    """Payload for a guard that constrains an integer to a safe closed domain.

    Attributes:
        subject: Dispatcher-local integer value name being checked.
        minimum: Inclusive lower bound accepted by the variant.
        maximum: Inclusive upper bound accepted by the variant.
        bit_width: Native integer width whose representable range must contain the bounds.
        signed: Whether the native representation is signed.
    """

    subject: str
    minimum: int
    maximum: int
    bit_width: IntegerBitWidth
    signed: bool = True

    def __post_init__(self) -> None:
        """Reject empty subjects, inverted domains, or unsafe native bounds.

        Raises:
            ValueError: If the closed domain is malformed or does not fit the requested width.
        """
        _require_identifier(self.subject, "subject")
        if self.minimum > self.maximum:
            raise ValueError("integer-domain minimum must be less than or equal to maximum")
        if self.bit_width == _INT32_WIDTH:
            lower = _INT32_MIN if self.signed else 0
            upper = _INT32_MAX if self.signed else 2**32 - 1
        else:
            lower = _INT64_MIN if self.signed else 0
            upper = _INT64_MAX if self.signed else 2**64 - 1
        if self.minimum < lower or self.maximum > upper:
            raise ValueError("integer-domain bounds must fit the requested native width")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard-payload serialization.

        Returns:
            str: Canonical text containing subject, bounds, width, and signedness.
        """
        return _canonical_text(
            (
                "integer-domain",
                self.subject,
                str(self.minimum),
                str(self.maximum),
                str(self.bit_width),
                str(self.signed),
            )
        )


@dataclass(frozen=True, slots=True)
class DirectFieldGuardPayload:
    """Payload for a guard that checks a direct field on a known owner type.

    The field is named structurally as data. Renderers can decide whether to use
    `getattr`, a C-level slot read, or a backend-specific access path, but the
    model never stores a source expression such as `obj.field`.

    Attributes:
        owner_subject: Dispatcher-local owner value name.
        owner_type_module: Importable module that owns the expected owner type.
        owner_type_qualname: Qualified owner type name inside `owner_type_module`.
        field_name: Direct field name expected on the owner.
        field_type: Optional report-facing field type identity required by the variant.
        minimum: Optional inclusive integer lower bound for the field.
        maximum: Optional inclusive integer upper bound for the field.
    """

    owner_subject: str
    owner_type_module: str
    owner_type_qualname: str
    field_name: str
    field_type: str | None = None
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        """Reject missing owner identity or missing direct field name.

        Raises:
            ValueError: If any required owner or field identity field is blank.
        """
        _require_identifier(self.owner_subject, "owner_subject")
        _require_identifier(self.owner_type_module, "owner_type_module")
        _require_identifier(self.owner_type_qualname, "owner_type_qualname")
        _require_identifier(self.field_name, "field_name")
        if "." in self.field_name:
            raise ValueError("direct-field field_name must name one direct field")
        if self.field_type is not None:
            _require_identifier(self.field_type, "field_type")
        if (self.minimum is None) != (self.maximum is None):
            raise ValueError("direct-field integer bounds must be provided together")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("direct-field minimum must not exceed maximum")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard-payload serialization.

        Returns:
            str: Canonical text containing owner, field, and expected field type.
        """
        return _canonical_text(
            (
                "direct-field",
                self.owner_subject,
                self.owner_type_module,
                self.owner_type_qualname,
                self.field_name,
                self.field_type or "",
                "" if self.minimum is None else str(self.minimum),
                "" if self.maximum is None else str(self.maximum),
            )
        )


@dataclass(frozen=True, slots=True)
class CallableCodeIdentityGuardPayload:
    """Payload for a guard that pins a callable to a known code identity.

    Attributes:
        subject: Dispatcher-local callable value name.
        callable_module: Importable module that owns the expected callable.
        callable_qualname: Qualified callable name inside `callable_module`.
        code_fingerprint: Stable digest of the callable code object or source body.
        receiver_subject: Optional instance receiver checked for method shadowing.
        code_firstlineno: Expected declaration start line for the live Python code object.
    """

    subject: str
    callable_module: str
    callable_qualname: str
    code_fingerprint: str
    receiver_subject: str | None = None
    code_firstlineno: int | None = None

    def __post_init__(self) -> None:
        """Reject incomplete callable or code identity fields.

        Raises:
            ValueError: If any callable identity or code fingerprint field is blank.
        """
        _require_identifier(self.subject, "subject")
        _require_identifier(self.callable_module, "callable_module")
        _require_identifier(self.callable_qualname, "callable_qualname")
        _require_identifier(self.code_fingerprint, "code_fingerprint")
        if self.receiver_subject is not None:
            _require_identifier(self.receiver_subject, "receiver_subject")
        if self.code_firstlineno is not None and self.code_firstlineno < 1:
            raise ValueError("code_firstlineno must be positive when provided")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard-payload serialization.

        Returns:
            str: Canonical text containing callable identity and code fingerprint.
        """
        return _canonical_text(
            (
                "callable-code-identity",
                self.subject,
                self.callable_module,
                self.callable_qualname,
                self.code_fingerprint,
                self.receiver_subject or "",
                "" if self.code_firstlineno is None else str(self.code_firstlineno),
            )
        )


@dataclass(frozen=True, slots=True)
class BufferLayoutGuardPayload:
    """Payload for a guard that constrains a buffer-compatible value layout.

    Attributes:
        subject: Dispatcher-local buffer value name.
        format: Buffer protocol format string expected by the variant.
        itemsize: Required item size in bytes.
        ndim: Required dimensionality.
        c_contiguous: Whether C-contiguous layout is required.
        f_contiguous: Whether Fortran-contiguous layout is required.
        readonly: Required buffer mutability, or `None` when the read-only kernel accepts
            either read-only or writable storage.
        minimum_length: Optional inclusive element-count lower bound.
        maximum_length: Optional inclusive element-count upper bound proving index and
            accumulator safety before native entry.
    """

    subject: str
    format: str
    itemsize: int
    ndim: int
    c_contiguous: bool
    f_contiguous: bool = False
    readonly: bool | None = None
    minimum_length: int | None = None
    maximum_length: int | None = None

    def __post_init__(self) -> None:
        """Reject impossible or underspecified buffer layout requirements.

        Raises:
            ValueError: If subject or format is blank, item size is not positive, dimensionality
                is negative, or incompatible contiguity flags are requested for a scalar.
        """
        _require_identifier(self.subject, "subject")
        _require_identifier(self.format, "format")
        if self.itemsize <= 0:
            raise ValueError("buffer-layout itemsize must be positive")
        if self.ndim < 0:
            raise ValueError("buffer-layout ndim must be at least 0")
        if self.ndim == 0 and (self.c_contiguous or self.f_contiguous):
            raise ValueError("buffer-layout scalar buffers cannot require contiguity")
        if (self.minimum_length is None) != (self.maximum_length is None):
            raise ValueError("buffer-layout length bounds must be provided together")
        if self.minimum_length is not None and self.maximum_length is not None:
            if self.minimum_length < 0:
                raise ValueError("buffer-layout minimum length must be non-negative")
            if self.minimum_length > self.maximum_length:
                raise ValueError("buffer-layout minimum length must not exceed maximum")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard-payload serialization.

        Returns:
            str: Canonical text containing buffer shape, format, and mutability constraints.
        """
        return _canonical_text(
            (
                "buffer-layout",
                self.subject,
                self.format,
                str(self.itemsize),
                str(self.ndim),
                str(self.c_contiguous),
                str(self.f_contiguous),
                str(self.readonly),
                "" if self.minimum_length is None else str(self.minimum_length),
                "" if self.maximum_length is None else str(self.maximum_length),
            )
        )


type GuardPayload = (
    ExactTypeGuardPayload
    | IntegerDomainGuardPayload
    | DirectFieldGuardPayload
    | CallableCodeIdentityGuardPayload
    | BufferLayoutGuardPayload
)


@dataclass(frozen=True, slots=True)
class GuardExpression:
    """Structured runtime guard expression for a compiled binding variant.

    A guard is a discriminated pair of kind and typed payload. It can represent
    exact type checks, integer domains, direct field checks, callable/code
    identity checks, and buffer layout checks, but it cannot carry arbitrary
    source expressions or code snippets.

    Attributes:
        kind: Guard family used by dispatch generation.
        payload: Structured guard payload whose dataclass must match `kind`.
        message: Human-readable explanation for reports and diagnostics.
    """

    kind: GuardKind
    payload: GuardPayload
    message: str

    def __post_init__(self) -> None:
        """Reject guard kind and payload combinations that cannot be rendered safely.

        Raises:
            ValueError: If `message` is blank or `payload` does not match `kind`.
        """
        _require_identifier(self.message, "message")
        expected_type = _payload_type_for_kind(self.kind)
        if not isinstance(self.payload, expected_type):
            raise TypeError(f"{self.kind} guard requires {expected_type.__name__}")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable guard serialization.

        Returns:
            str: Canonical text containing the guard kind, payload, and message.
        """
        return _canonical_text((self.kind, self.payload.canonical_serialization, self.message))

    @property
    def fingerprint(self) -> str:
        """Return a stable content-derived guard fingerprint.

        Returns:
            str: Short BLAKE2 digest for guard cache and report identity.
        """
        return _stable_digest("native-guard", self.canonical_serialization)


@dataclass(frozen=True, slots=True)
class CompiledBindingVariant:
    """One compiled implementation candidate for a Python binding.

    Attributes:
        backend: Native compiler backend that produced the variant.
        kind: Dispatch class for ordering and guard interpretation.
        binding: Descriptor-aware Python binding this variant can replace when guards pass.
        compiled_symbol: Importable compiled implementation symbol.
        artifact_fingerprint: Backend artifact or compilation-unit content fingerprint.
        guards: Runtime guards that must pass before this variant is called.
        payload_fingerprint: Optional backend-neutral payload digest covered by the variant.
    """

    backend: Backend
    kind: CompiledVariantKind
    binding: BindingTarget
    compiled_symbol: SymbolId
    artifact_fingerprint: str
    guards: tuple[GuardExpression, ...]
    payload_fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Reject incomplete compiled variant descriptions.

        Raises:
            ValueError: If the artifact fingerprint is blank, guards are missing for a
                specialized integer variant, or a safe integer variant lacks a matching
                integer-domain guard width.
        """
        _require_identifier(self.artifact_fingerprint, "artifact_fingerprint")
        if self.payload_fingerprint is not None:
            _require_identifier(self.payload_fingerprint, "payload_fingerprint")
        if self.kind in {"safe-int32", "safe-int64"} and not self.guards:
            raise ValueError("safe integer variants require guards")
        if self.kind in {"safe-int32", "safe-int64"} and self.backend != "cython":
            raise ValueError("safe integer variants require the Cython backend")
        required_width: IntegerBitWidth = (
            _INT32_WIDTH if self.kind == "safe-int32" else _INT64_WIDTH
        )
        if self.kind in {"safe-int32", "safe-int64"} and not any(
            guard.kind == "integer-domain"
            and isinstance(guard.payload, IntegerDomainGuardPayload)
            and guard.payload.bit_width == required_width
            for guard in self.guards
        ):
            raise ValueError(f"{self.kind} variants require a {required_width}-bit integer guard")

    @property
    def canonical_serialization(self) -> str:
        """Return the stable compiled-variant serialization.

        Returns:
            str: Canonical text covering backend, symbols, artifact, payload, and guards.
        """
        guard_serialization = tuple(guard.canonical_serialization for guard in self.guards)
        return _canonical_text(
            (
                "compiled-variant",
                self.backend,
                self.kind,
                _binding_serialization(self.binding),
                self.compiled_symbol.stable_id,
                self.artifact_fingerprint,
                self.payload_fingerprint or "",
                *guard_serialization,
            )
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable content-derived compiled-variant fingerprint.

        Returns:
            str: Short BLAKE2 digest for variant cache and report identity.
        """
        return _stable_digest("native-variant", self.canonical_serialization)


@dataclass(frozen=True, slots=True)
class BindingDispatchPlan:
    """Deterministic dispatch plan for one Python binding.

    `variants` are canonicalized during construction so dispatch always tries
    safe 32-bit variants, safe 64-bit variants, generic mypyc/Cython variants,
    and finally the original Python fallback represented by `fallback_symbol`.

    Attributes:
        binding: Descriptor-aware Python binding whose calls are dispatched.
        fallback_symbol: Original Python implementation used when all compiled guards fail.
        variants: Compiled variants in deterministic dispatch order.
        dispatch_fingerprint: Static source or call-site fingerprint for this binding dispatch.
    """

    binding: BindingTarget
    fallback_symbol: SymbolId
    variants: tuple[CompiledBindingVariant, ...]
    dispatch_fingerprint: str

    def __post_init__(self) -> None:
        """Canonicalize variant order and reject ambiguous binding dispatch plans.

        Raises:
            ValueError: If the dispatch fingerprint is blank, no compiled variants are present,
                variants target different bindings, or variant fingerprints collide.
        """
        _require_identifier(self.dispatch_fingerprint, "dispatch_fingerprint")
        if not self.variants:
            raise ValueError("binding dispatch plans require at least one compiled variant")
        for variant in self.variants:
            if _binding_serialization(variant.binding) != _binding_serialization(self.binding):
                raise ValueError("compiled variants must target the dispatch binding")
        ordered_variants = tuple(sorted(self.variants, key=_compiled_variant_sort_key))
        fingerprints = tuple(variant.fingerprint for variant in ordered_variants)
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("compiled variant fingerprints must be unique")
        object.__setattr__(self, "variants", ordered_variants)

    @property
    def ordered_targets(self) -> tuple[tuple[DispatchTargetKind, str], ...]:
        """Return compiled target identities followed by the Python fallback.

        Returns:
            tuple[tuple[DispatchTargetKind, str], ...]: Ordered dispatch target kind and stable
            symbol identity pairs.
        """
        compiled_targets: tuple[tuple[DispatchTargetKind, str], ...] = tuple(
            ("compiled", variant.compiled_symbol.stable_id) for variant in self.variants
        )
        return (*compiled_targets, ("python-fallback", self.fallback_symbol.stable_id))

    @property
    def canonical_serialization(self) -> str:
        """Return the stable dispatch-plan serialization.

        Returns:
            str: Canonical text covering binding, fallback, dispatch fingerprint, and variants.
        """
        return _canonical_text(
            (
                "binding-dispatch",
                _binding_serialization(self.binding),
                self.fallback_symbol.stable_id,
                self.dispatch_fingerprint,
                *(variant.canonical_serialization for variant in self.variants),
            )
        )

    @property
    def fingerprint(self) -> str:
        """Return a stable content-derived dispatch-plan fingerprint.

        Returns:
            str: Short BLAKE2 digest for dispatch cache and report identity.
        """
        return _stable_digest("native-dispatch", self.canonical_serialization)


@dataclass(frozen=True, slots=True)
class NativeVariantPlan:
    """Backend-neutral plan containing native dispatch for one optimized owner.

    Attributes:
        owner: Source symbol that owns the native-variant opportunity.
        source_fingerprint: Stable fingerprint of source text and static analysis inputs.
        planner_version: Version of the planner semantics used to build the plan.
        dispatch_plans: Binding dispatch plans in deterministic binding order.
        notes: Optional stable report notes that do not contain runtime measurements.
    """

    owner: SymbolId
    source_fingerprint: str
    planner_version: str
    dispatch_plans: tuple[BindingDispatchPlan, ...]
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Canonicalize dispatch plan order and reject incomplete native plans.

        Raises:
            ValueError: If static fingerprints are blank, no dispatch plans are present, or
                multiple dispatch plans target the same binding.
        """
        _require_identifier(self.source_fingerprint, "source_fingerprint")
        _require_identifier(self.planner_version, "planner_version")
        if not self.dispatch_plans:
            raise ValueError("native variant plans require at least one dispatch plan")
        ordered_dispatch = tuple(
            sorted(
                self.dispatch_plans,
                key=lambda plan: (
                    plan.binding.source.stable_id,
                    plan.binding.target_owner_class or plan.binding.owner_class or "",
                    plan.binding.kind,
                ),
            )
        )
        binding_ids = tuple(_binding_serialization(plan.binding) for plan in ordered_dispatch)
        if len(set(binding_ids)) != len(binding_ids):
            raise ValueError("native variant plans cannot duplicate binding dispatch plans")
        object.__setattr__(self, "dispatch_plans", ordered_dispatch)

    @property
    def canonical_serialization(self) -> str:
        """Return the stable native-plan serialization.

        Returns:
            str: Canonical text covering static owner, source, planner, dispatch, and notes.
        """
        return _canonical_text(
            (
                "native-variant-plan",
                self.owner.stable_id,
                self.source_fingerprint,
                self.planner_version,
                *(plan.canonical_serialization for plan in self.dispatch_plans),
                *self.notes,
            )
        )

    @property
    def stable_id(self) -> str:
        """Return the stable content-derived native-variant plan identifier.

        Returns:
            str: Short BLAKE2 digest for native-variant plan identity.
        """
        return stable_native_variant_plan_id(self)


def stable_native_variant_plan_id(plan: NativeVariantPlan) -> str:
    """Return a deterministic native-variant plan identifier.

    Args:
        plan: Immutable native-variant plan to identify.

    Returns:
        str: Short content-addressed identifier derived from canonical plan serialization.
    """
    return _stable_digest("native-plan", plan.canonical_serialization)


def _payload_type_for_kind(kind: GuardKind) -> type[GuardPayload]:
    if kind == "exact-type":
        return ExactTypeGuardPayload
    if kind == "integer-domain":
        return IntegerDomainGuardPayload
    if kind == "direct-field":
        return DirectFieldGuardPayload
    if kind == "callable-code-identity":
        return CallableCodeIdentityGuardPayload
    return BufferLayoutGuardPayload


def _compiled_variant_sort_key(
    variant: CompiledBindingVariant,
) -> tuple[int, int, str, str]:
    return (
        _VARIANT_KIND_ORDER[variant.kind],
        _BACKEND_ORDER[variant.backend],
        variant.compiled_symbol.stable_id,
        variant.fingerprint,
    )
