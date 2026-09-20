# PR #560 — independent Lane E review

Repository: **avifenesh/memra**. Reviewed PR head
**dcbc1bc089a6f95ad7f5e4bc7e90aa658dadd69d** against base
**b3487a03b0ee3f833c1157e7b7d68f2cb35a3843**. Local review source is merge
**d3d054a8e2a177c10e5e923614945eba96c1d069** (the same reviewed file bytes).
This is a source/CPU review, not native CUDA, serving, or release qualification.
No other lane's implementation was edited. Lead owns dispatch and integration.

## Findings

**0 critical, 0 high, 3 medium, 0 low.** Medium means a reproducible development
runner/evidence defect, not a demonstrated token-numerics defect. Resolve before
relying on the affected runner or collector claim. The review does not approve
hardware qualification or the PR's unrun serving surfaces.

| ID / severity | File:line at reviewed head | Evidence / impact | Owner and proposed fix | Disposition |
| --- | --- | --- | --- | --- |
| E560-1 / medium | `crates/memra-engine/Cargo.toml:35-37`; `research/spill-b-20260919/native-patch-check.py:29-30` | Explicit registration changes the only Cargo target to `kv-tier-gate`, while the newly added native driver still builds/tests `kv_tier_gate`. `cargo build --offline -p memra-engine --bin kv_tier_gate` actually exits **101**, `no bin target named ...`; metadata exposes only the hyphenated name. The driver stops before its gate tests on the integrated tree. | **B + lead manifest owner**: keep one canonical target; update executable build/test consumers and current instructions to `kv-tier-gate`. Preserve historical command/binary receipts unchanged. Add a metadata-to-driver target-name regression. | **HELD — B**. Rechecked `a2a9eecf` (remote lane B); canonical-target correction not yet present. |
| E560-2 / medium | `tools/tier-battery.py:368-373`; `crates/memra-engine/src/bin/kv_tier_gate.rs:307-309` | `explicit_refusal` only accepts terminal `REFUSED:` or `Error:` with exit 2. The actual active raw line starts **`kv-tier-gate: REFUSED:`**, so the current function returns `None`; conversely `Error: CUDA_ERROR_OUT_OF_MEMORY` with exit 2 is labeled refused, though it is an execution failure, not a precondition refusal. Replayed the archived active log through the current function and exercised the generic-error negative arm. Missing-native-seam vs failed execution is therefore misclassified in both directions. | **D**, coordinate diagnostic format with **B**: recognize an explicit refusal marker with the supported executable prefix, not generic `Error:`. Retain terminal/exit/timeout checks and the complete raw quote. Add both regression cases; do not rewrite old captures to pretend the old collector classified them correctly. | **HELD — D**. Rechecked `ff50258b` (remote lane D); refusal correction not yet present. |
| E560-3 / medium | `tools/tier-battery.py:825-830`; `research/spill-a-20260919/storage_capture.py:28-37` | The new storage admission predicate scans individual argv basenames. A canonical `bash -c '/bin/storage-bench roundtrip /different-filesystem/object 264 direct'` bypasses mandatory `--storage-root`, although the storage join explicitly accepts that wrapper. Even direct argv with a root supplied never binds that root to the benchmark object path/mount. Thus the command can execute on a different filesystem from the ancestry in its CELL. Stubbed only subprocess/lock/storage effects and exercised the real CLI control flow: missing-root wrapper reaches the runner, and unrelated-root wrapper reaches it with that unrelated descriptor. | **D**, coordinate exact command parsing with **A**: parse supported direct/canonical wrapper forms consistently, refuse ambiguous storage invocation, and derive/prove ancestry for the actual object directory (existing parent for new roundtrip). Check actual mount identity rather than a string prefix: a nested mount/symlink can change the device. Validate command-to-storage binding in the join. Generic subprocess captures must not claim an arbitrary root describes their I/O. | **HELD — D**. Rechecked `ff50258b` (remote lane D); storage-root binding correction not yet present. |

Reproduction output and API snapshots: [`pr560-review/`](pr560-review/).
`probes.json` records exact command/exit/raw hashes and source; `refusal-probes.json`
uses the actual archived refusal. `storage-root-probes.json` is explicitly
**CPU-stub control flow**: no GPU, storage command, or canonical lock was used.
The storage finding does not invalidate A's observed byte-exact O_DIRECT cells or
claim those historical cells secretly used NVMe. It identifies an admission gap.

## Engine and doctrine inspection

- Read the complete new `ple_rows_tier.rs`, its `Qwen4ExpGpu` constructor/gather/
  qualification hook, and `qwen4exp_gpu_gate` dispatch. OFF remains `None`; the
  existing host gather loop and GPU PLE math stay unchanged. `GatheredRows::legacy`
  adds no charge. The tier path is installed by the explicit gate, reset to OFF
  after its result, and compares output plus convolution state bitwise. No source
  evidence found of changed OFF-arm arithmetic or request-mode switching.
- The host adapter is bounded (rows/output/metadata budget), derives checksums
  from the immutable source borrow, preserves requested order/repeats through
  RowService, copies expanded values before retiring CPU bank leases, and retains
  the output charge through the existing H2D consumer. Its synthetic artifact id
  is explicitly ephemeral; it is **not** checkpoint authentication or persistent
  cache identity. The host-only proof must not be reused as device publication.
- The F32/BF16 conversion is confined to independent tiny gate fixtures. Each
  encoding compares its own identical input bytes OFF/ON; it does not substitute
  BF16 for a requested model program or claim checkpoint-faithful support.
- `banked_residency.rs` changes only scoped dead-code annotations on an uncalled
  native compile probe. `lib.rs` registers it privately. No mixed-layout expert
  dispatch is connected by this PR; this is not expert-residency qualification.
- Read complete `kv_tier_gate.rs`, `cli.rs`, and `capture_contract.rs`. Active and
  prefix refuse before artifact/CUDA initialization. Baseline uses one tokenwise
  trunk program, validates native KV geometry and plan-declared state slots,
  refuses unbound sidecars, and distinguishes unexecuted MTP. No architecture-name
  allowlist or format fallback was added. The three permitted build/lock env
  names are not an architecture capability allowlist. The gate does not prove
  chat/template behavior, MTP, prime, active reload, or cross-request serving.
- **`storage_bench.rs` has no diff in PR #560** against the stated base. Inspected
  its current direct/buffered selection, bounded exact readback, backend labels,
  and null physical/device timing fields as context; did not attribute its
  inherited implementation to this PR. O_DIRECT acceptance on overlay is not
  physical NVMe ancestry or spill speed.
- Read changed collector, lock-proof, bootstrap, and topology code. Canonical
  GPU lock names are retained; bootstrap pidfile is process metadata, not a third
  GPU campaign lock. Inherited-FD ownership proof and unknown/missing telemetry
  stay distinct from qualification. Explicit unproven-storage mode is a labeled
  development diagnostic, not a format/backend fallback delivered as support.
- No new runtime env read, numerical default, kernel/FFI change, or model-family
  allowlist was found in the reviewed engine diff. Gate-only CLI arms are retained
  qualification diagnostics, not performance doors requiring a default promotion.

## Automated reviews

Read-only GitHub queries for both `/pulls/560/comments` and `/pulls/560/reviews`
returned **`[]`** at the UTC recorded in `pr560-review/probes.json`.
Therefore there were **zero comments to classify**, not an automated approval.
No comment was posted or resolved. Later comments require a fresh lead/owner
triage against the exact changed head; absence at this snapshot proves nothing
about their eventual arrival.

| Comment | Real / not real | Owner | Proposed disposition |
| --- | --- | --- | --- |
| None present in saved API snapshot | Not applicable | Lead monitors subsequent reviews | Re-read before integration; do not mark absent comments resolved. |

## Held — not demonstrated defects / not cleared gates

1. **Native completion/retirement semantics:** new C adapter is host-only. Real
   producer events, installed consumer waits, last-use CUDA completion, graph
   pins and uncertain submission teardown are not exercised by this source
   review. A/B/C must bind shared schedules to their real backends (v1.3 draft).
2. **Stateful PLE continuation:** gate histories cover prefill/decode/verify-shaped
   calls but initialize fresh PLE state for each twin. This is valid tiny-slice
   evidence, not persistent-state churn, cancel/reuse, model-scale capacity or
   mid-request program-transition qualification. Do not expand its claim.
3. **KV full-program identity:** artifact/plan/prompt/binary pins and baseline
   snapshots exist; complete native `ProgramIdentity`, allocator-stable operands,
   scheduler demote/reload engagement, graph lifetime and serving-shape equality
   remain held. Active's explicit refusal is correct, not something to suppress.
4. **Native build and exact source:** historical native receipts have their own
   commits, fragments and binary hashes. No nvcc/GPU is available here. Mac tests
   or CI compile-only checks do not establish that PR head ran on CUDA.
5. **Storage/peer claims:** NVMe ancestry, physical-byte counters, io_uring,
   measured active PCIe behavior, directed peer grants, PRO-pair and four-card
   gates remain unrun here. A topology ceiling/idle sample is not bandwidth.
6. **No speculative cleanup finding:** CPU wrapper error paths and budget
   accounting warrant native fault-injection coverage, but no reachable leak or
   token-corruption counterexample was established in this review. Do not file
   that concern as a proven defect without a reproducer.

Check execution and final remote revision are recorded separately after the
required lane battery. All findings are handoffs, not fixes or a merge claim.
