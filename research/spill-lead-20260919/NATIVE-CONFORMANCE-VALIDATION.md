# Native conformance receipt admission — proposal to D

Repository: **avifenesh/memra**. Contract: [freeze v1.3](FREEZE-V1.3.md).
Schema: [`native-conformance.schema.json`](native-conformance.schema.json).

**Schema finalized; CLI hook proposed, not implemented by E.** D owns
`tools/tier-battery.py`. Existing capture/CELL/storage validation remains intact.
The schema deliberately uses only the existing offline `validate_schema` subset:
no external dependency, schema fetch, `$ref`, dynamic object properties or hidden
format checker. Use a named schedule list instead of arbitrary JSON keys; each
entry maps `name × trait × backend × route` to an explicit verdict. The validator
must reject duplicate tuples and compare the complete required schedule manifest.

## Proposed CLI

Add `--schema native-conformance` for one JSON receipt (not JSONL). For `auto`,
recognize the exact object kind `native-conformance` or an explicit documented
suffix, never infer native evidence from a green process exit. Invoke:

```text
python3 tools/tier-battery.py --validate <receipt.json> --schema native-conformance
```

This command is a **proposal**, not a currently supported invocation. Load the
checked-in schema relative to the repository. Its schema_version is receipt v1,
contract_revision is 1.3; neither changes persisted tier WIRE_VERSION=1.

## Required shape

Every object is closed (`additionalProperties: false`). SHA-256 fields require
64 lowercase hexadecimal characters; source commits require 40. Evidence
references require bounded relative paths, byte lengths and SHA-256. Mandatory
unknown observations use null with an explanatory reason, not absent fields or
fabricated values. Shape validation permits an honest HELD receipt.

- `source`: exact repository and commit, explicit dirty bit and fragment manifest.
- `binary`: executed binary SHA-256/length/name, all loaded module descriptors,
  compiler, CUDA driver/runtime, target and feature set.
- `artifact`: either immutable model-lock descriptor plus serialized-plan and
  ProgramIdentity hashes, or a **fixture-only** generator/input byte manifest
  with `model_artifact: null`. Full ProgramIdentity bytes live in the hash-bound
  lock/source evidence; a hash is not permission to omit the underlying identity.
- `collector`: CELL run_id, journal/capture/lock descriptors, integrity verdict,
  UTC and monotonic bounds, argv, exit/timeout/failure quote and raw descriptors.
- `hardware`: hardware shape/count, ordinal and capability per card, observed
  `power_limit_w` **and** `power_max_limit_w`, reason for unknown observations,
  250 ms telemetry and inside-lock compute-app snapshot evidence. The corresponding
  nvidia-smi source columns are power.limit and power.max_limit. No machine/provider
  identifiers, addresses, locations or costs belong in this receipt.
- `storage`: explicitly null if unused; otherwise requested/actual backend,
  ancestry, tested-object binding, logical/I/O/physical bytes and fallback reason.
- `required_schedules`: hash-bound manifest of required trait/backend/route tuples
  for this scope, independently selected before execution, not inferred from PASS.
- `schedules`: exact schedule name, trait, concrete backend, actual route, imported
  source descriptor, PASS/FAIL/REFUSED/HELD, execution bit, reason, test filter,
  raw/event trace evidence, actual/expected byte hashes and checked byte count.
- `ignored_tests`, `filtered_tests`: explicit inventories; absence is not success.
- `scope`: CUDA conformance, checkpoint or serving, kept distinct.
- `qualification`: **always false** in this capture envelope. Schema/integrity
  validation cannot issue numerical, model, route or release qualification.

## Required semantic hook, after structural validation

1. Use D's existing evidence resolver to verify every relative path, symlink
   containment, byte count and digest. No absolute paths, escaping symlinks,
   `..`, missing files or digest mismatch. Never retrieve missing evidence online.
2. Run `validate_cell` / `validate_capture`; require the unique complete CELL
   start/end pair to match `collector.cell_id`, argv, timestamps, status, raw,
   lock proof and binary/module/source identities. Integrity PASS must agree
   with replay, not the receipt's assertion. Torn/interrupted runs stay retained
   as HELD/FAIL and never promote. Check monotonic and UTC ordering explicitly.
3. Verify executed binary hash against the collector's executable manifest. If
   only the asserted hash is available, mark identity HELD. For a dirty run,
   require the complete source-fragment manifest, not a single convenient file.
   A clean source must have an empty fragment array. Verify exact imported
   schedule bytes against the pinned repository commit/manifest.
4. Resolve and validate the model lock and full ProgramIdentity for checkpoint
   or serving scope. Fixture artifacts can only support conformance scope.
   Artifacts, plan, stream/numeric class and expected outputs must match the
   declared test program; independently derived expected hashes cannot just be
   copied from actual output.
5. Compare hardware count/unique ordinals and power observations to raw capture.
   Null power data requires an unknown reason and keeps affected qualification
   held. Do not replace max_limit with current limit or infer headroom. Check
   250 ms telemetry coverage and canonical lock/snapshot ordering from raw times.
6. If storage was used, bind the actual command object to captured filesystem and
   mount ancestry, not a caller-selected unrelated root. Require object binding
   for new receipts; legacy captures lacking it remain diagnostics and cannot
   qualify storage through this new envelope. Keep overlay/direct/physical NVMe
   and logical vs physical bytes distinct. Refusal is an explicit raw marker,
   never a generic execution error relabeled as a precondition failure.
7. Expand the externally pinned required schedule manifest; reject duplicate,
   missing or unexpected tuples. Include required ignored/filtered tests as HELD.
   PASS requires executed=true, nonempty hash-bound raw/event trace, zero relevant
   failure/timeout, and all schedule assertions. FAIL requires executed=true plus
   a captured error; REFUSED requires explicit matched raw refusal; HELD requires
   a precise missing-gate reason. Non-PASS requires non-null reason. Do not
   synthesize PASS from aggregate exit status or from structural validity.
8. Byte schedules require positive checked_bytes and independently sourced equal
   actual/expected hashes for PASS; non-byte schedules explicitly use null hashes
   and zero count. Replay event/lifetime evidence: source and destination charges
   do not disappear on logical retirement; unobserved completion never advances
   the state. Enforce the per-schedule facts in the frozen shared schedule, not
   a reduced collector-only version.
9. Return distinct `structural_valid`, `integrity_valid`, and per-tuple results,
   with `qualification: false`. A separate explicit owner gate decides native
   qualification after coverage, identity and numerical checks. This validator
   does not run a GPU, rerun model serving or authorize release/performance claims.

## CPU verification and D acceptance cells

`test_native_conformance_schema.py` runs against the **actual existing** offline
validator with a structural-only fixture, every mandatory field removed at each
nesting depth, bad source/binary hashes, path escape, unknown fields, bool-for-int,
missing power maximum and missing schedule verdict. No native receipt is fabricated.

D's integration must add semantic red cells: swapped CELL id/binary/source, stale
schedule source, dirty-without-fragments, forged artifact hash, mismatched max power,
missing required tuple, duplicated tuple, PASS with executed=false/empty trace,
ignored-required PASS, generic-error-as-refusal, fixture-as-checkpoint, escaping
symlink, wrong storage object/mount and interrupted capture. Existing historical
receipts are immutable; do not rewrite them to pretend this validator ran then.
