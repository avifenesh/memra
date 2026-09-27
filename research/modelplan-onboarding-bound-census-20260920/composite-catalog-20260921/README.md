# Composite tensor catalog

`CompositeTensorCatalog::compile` validates the actual opened repack/fallback components before
applying precedence. It returns immutable metadata only. It does not implement `TensorSource`,
hold source or file handles, expose a materializer, or remove the existing fallback refusal in
`BoundTensorSource`. Its digest covers semantic metadata, not checkpoint payload identity.

The compiler builds each component's schema through its model pack and retains its own dialect,
physical records, auxiliary metadata, contract, interpretation and bound roles. Sparse fragments
use the canonical binder for shape, dtype, auxiliary, alias, duplicate-claim and unexpected-row
validation. Missing roles are resolved only after every component has passed those checks, so an
overlay cannot conceal a malformed shadowed tensor. Component plans must agree.

Completeness follows the base artifact's schema. Sparse overlays do not introduce unrelated
representation requirements: a GGUF norm over an HF Step artifact does not require a GGUF-only
RoPE tensor. A GGUF base still requires that tensor. Declared overlay tensors remain validated
against the overlay dialect's schema. The original incorrect union of required schemas and its
failing Step control are retained in the development logs.

Selections retain the physical component, dialect, transform and owner. Whole tensors use the
first declared replacement. Canonical groups retain member order and require every member;
the compiler never manufactures a missing split member from a lower stacked bank. Independent
GGUF scale rows require an owning weight in their own component and follow that weight's
selection. Folded HF planes remain attached to their own weight. Unselected inventory is fully
validated and excluded from selected execution metadata.

## Executed acceptance controls

The eleven CPU controls cover:

- Two-level override precedence, preserved shadowed roles, pathname unlink, detached metadata,
  and continued fallback runtime/identity refusal.
- A GGUF norm override with an HF Qwen3.5 norm retaining its `NormAddOne` transform.
- Rejection of shadowed shape, integer-weight, AWQ input-width and FP8 scale-grid errors.
- Rejection of unclaimed tensors and missing final roles, with a valid sparse replacement control.
- Digest sensitivity to shadowed storage interpretation and selected load scope.
- Explicit refusal of unimplemented composite masks and GGUF split-member schemas.
- Scale ownership, removal of shadowed scales, and rejection of auxiliary-only overlays.
- Canonical HF expert-member order, missing-member rejection, and explicit whole-bank replacement.
- Alias-collision rejection and configuration-declared tied-head ownership.
- Step vision inventory validation without selection during text loading.
- Continued enforcement of the GGUF base's required RoPE tensor.

Final results: 395 GGUF tests passed with 2 declared ignores; the compiler/CLI suites,
18 identity controls, Clippy, Linux type/lint checks, formatting and diff checks passed.
Command statuses, exact source hashes, binary hashes and lossless logs are in `evidence.json`
and `raw/`. Earlier compiler/test failures are retained under `development/`. Linux checks use
`DOCS_RS=1`: they establish type/lint correctness, not native CUDA execution. The previously
recorded #548 reference bit mismatch remains separate; this stage does not claim a new full
reference-suite pass.

## Remaining acceptance

Independent source review of this catalog is next. Composite retained masks, GGUF split-member
contracts, source-bound payload and native-plane materialization, disk views, and complete opened
composite identity still need implementation and their own controls. Auxiliary-group forms that
cannot be attached unambiguously remain refused. The metadata plan is not a prepared runtime
program; source-factor preflight remains part of the later activation boundary.

#537 owns canonical configuration and the shared numeric decoder. Its frozen
`bb637184682216ea38cbb9eb79dab4841368eca1` is pending independent review and is not included here.
The shared helper, dependency features and consumers must be integrated together after that review.
Root activation, native/model qualification, support promotion and main integration remain pending.
No remote build, GPU stage, new allocation, or new issue lane was started for this catalog slice.
