# Scope contract before universal activation

Status: architecture-review proposal, not an implemented or approved runtime behavior.
The existing working text loader stays in place until this contract and its implementation pass
review. No artifact is filtered, re-exported, or changed to make a census pass.

## Observed boundary

The hash-verified pinned Step FP8 index declares 1,471 names: 804 for the text model and 667 for
vision/projector assets. The Step text-schema correction accounts for all 804 text names. The
current Step plan does not represent the perception encoder. Index-name evidence and the exact
unrepresented-name list are preserved in [the Step schema record](../modelplan-onboarding-step-hf-schema-20260920/RESULTS.md).
Shard headers, shapes, dtypes and payloads remain distinct checks. Native text gates and whole-file
custody hashes do not qualify vision execution or establish a complete artifact census.

Independent review of the 24 text-shard headers found **930 physical text tensors**, including
126 `weight_scale_inv` auxiliaries absent from the index. Those fold into 804 semantic rows/roles;
all 126 grids and all text shapes/storage passed that metadata check. Thus 1,471 is an index-entry
count, not a complete physical inventory. The 667 vision/projector count is still indexed names,
not an assertion that every physical vision tensor or auxiliary has been inventoried.

All opened shard headers are authoritative for physical census. Validate every indexed name's
owning shard, retain header-only auxiliaries with their exact owner/shape/dtype/byte range, and
reject duplicate ownership and undeclared leftovers. Do not impose a fabricated index/header
bijection or silently drop the 126 sidecars. The existing safetensors reader already includes
header-only names; scoped binding must preserve that behavior.

## Correction: existing native Step vision

Step vision already has a native implementation in `vision_step.rs` and historical staged/e2e
qualification in `research/step37-vision-20260830/`. Earlier wording that implied no implementation
was incorrect. The gap is canonical ModelPlan/bound-loader representation. Inventory-only status
here must not disable or downgrade the existing qualified route. Full canonical activation must
represent and preserve that route, or explicitly remain a text-only migration while the existing
vision path remains intact. New artifact/binary/scope qualification remains separate from the
historical receipts. Any scope check below concerns the compiled binding's coverage, not a blanket
claim that the engine lacks vision.

## Proposed compiler products

Keep one immutable compiler-produced bundle containing:

1. The original opened-source artifact identity from #542, without substitution or reopening paths.
2. Captured source interpretation (including layout/precision declarations), normalized config and
   a complete physical catalog. Every physical tensor and auxiliary has exactly one semantic owner.
3. An explicit requested execution surface and its dependency closure, executable ModelPlan and
   capability manifest. Existing text entrypoints request the text surface explicitly in code.
4. Exact catalogs for declared but unselected surfaces, plus their execution status. A declared
   vision surface may have a validated inventory while its execution remains unsupported.

A pack must declare unused-surface names, shapes, encodings, auxiliaries and ownership precisely.
A prefix wildcard accepting arbitrary `vision_model.*` tensors is forbidden. Catalog-only status
must not manufacture a reference executor, support state or tuned capability. For Step, the vision
catalog still needs pinned header/code evidence for shapes and source defaults before it can be
accepted; the index alone is insufficient.

The requested plan is selected through its surface/dependency manifest, never a runtime
architecture-name allowlist. Shared tensors belong to both dependency closures without being
physically claimed twice. Undeclared, duplicate, ambiguous, wrong-shape or wrong-layout tensors
remain errors even when they sit in an unselected component. Missing declared artifacts remain
errors; this is not a way to accept an incomplete checkpoint by requesting less work.

Placement and weight admission consume an explicitly named selected-dependency view and account
for materialized representations and shared allocations. They must not charge unselected vision
weights as if they were uploaded. The complete physical census remains separately available;
`tensor_census` must not silently become a filtered list to achieve this behavior. Request-state
admission and owning-device reclamation retain the already-qualified #544 rules.

## Access and refusal behavior

Only selected dependency IDs may be materialized or uploaded. Attempts to read an unselected
surface fail with its semantic ID and selected-surface context. Scoped raw GGUF helpers resolve
an authorized semantic ID to its bound physical target first; forwarding an unrestricted file
lookup cannot bypass the selection. Native, auxiliary and disk views all retain that rule and
propagate errors without turning them into optional absence.

A vision request against a text-only loaded model returns an unsupported-surface error. It does
not run a partial image program, fabricate tensors or switch numerical programs in an existing
request. The full checkpoint remains untouched and accounted for. Once the existing native vision route has a correct canonical
plan/reference binding and current receipts, that load/bundle can expose it through this boundary.

## Identity and qualification

Retain the three existing positive support states. Record the qualified surfaces alongside the
state rather than adding a new positive state or implying that text receipts cover every surface.
A receipt must bind the requested surface, its dependency closure/plan, complete catalog,
interpretation and exact binary. Changing scope or any owner/transform invalidates that receipt.

Compose identities under separate domains: retain #542's opened-byte artifact hash as a component,
then hash it with the immutable semantic bundle/scope. Do not replace the byte identity with a
binding digest or allow callers to supply unrelated config/plan/catalog pieces. Existing runtime
library/environment guards, private repack backing, mutation revocation and snapshot revocation
remain in force.

## Required controls before activation

- Complete original artifact catalog binds; no changed or stripped checkpoint is used.
- Malformed tensors in selected and unselected surfaces fail before CUDA upload.
- Text-only load has zero reads/uploads from unselected vision IDs; deliberate access refuses.
- Text output on the supported artifact matches the existing native text program on the intended
  rig, including PP/spec/serving transitions already required for that program.
- The bound interface refuses vision selections it cannot represent; the existing qualified native
  vision route is preserved until its canonical integration, without a silent regression.
- Scope/catalog/ownership/transform changes alter identity; unchanged opened artifacts remain
  pinned across pathname replacement. Error paths do not select another representation.
- Full-artifact, selected-surface and unsupported-surface status remain separate in inspection,
  model cards, native receipts and the final report.

Review requested: whether this complete-catalog/selected-execution boundary preserves the owner's
no-substitution rules and current text behavior, and whether the proposed support/identity scope
is sufficiently explicit. Universal activation remains pending that review and implementation.
