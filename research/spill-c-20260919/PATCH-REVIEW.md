# Hy3 dispatch review — NO-GO, partial unapplied guard patch

Repository `avifenesh/memra`; reviewed source `b1df0e73` on
`lane/spill-c-20260919`. `hybrid.rs`, `hybrid_forward.rs`, `model.rs` and
`moe_cache.rs` remain unchanged from the integration parent. No runtime patch was
applied, built, or tested with CUDA. No engine/server completion claim.

**Task-3 is incomplete.** `HY3-DISPATCH-PATCH.diff` is a conservative first slice
(mask/bounds preconditions), NOT the requested complete BankedResidency dispatch
conversion. Review verdict: retain for the next native-owner round; **do not apply
as a completed migration**. Generating calls to nonexistent native owner/SLRU
methods or treating host leases as GPU-ready would conceal the actual blocker.

## Patch checks actually run

- `python3 research/spill-c-20260919/make-dispatch-patch.py`: generates candidate
  text in memory, runs rustfmt on the candidate, writes only the `.diff`.
- `git apply --check research/spill-c-20260919/HY3-DISPATCH-PATCH.diff`: exit 0.
- `git diff -- crates/memra-engine`: empty. This proves unapplied state, not buildability.
- rustfmt parsed candidate Rust syntax. No type check, native HostExps build,
  numerical gate, GPU byte test, performance run, or independent reviewer ran.

## Hunk-by-hunk source review (original file:line)

| Hunk | Existing code and proposed effect | Review |
|---|---|---|
| `hybrid.rs:2833` | Add `require_active_expert`: all three bank counts and mask length must agree; original index must fit bank and `BlockId` u16; masked original index errors. | Checks precede indexing; no renumbering, byte fabrication, format or top-k change. Error path native-build pending. |
| `hybrid_forward.rs:14641` | Sequential selected-expert loop before slab/staged/cache branches. | Reject bad original ID before layout/byte access. Existing earlier host-page hints are NOT all covered: full source-binding migration still required. |
| `hybrid_forward.rs:19835` | `moe_cached_gemm_q8`, before per-projection lookup. | Retains existing q8 program; does not promote mixed/Q2_K into q8. All caller predicates remain unchanged. |
| `hybrid_forward.rs:19871` | `moe_cached_gemm`, before layout/source/cache access. | Metadata-aware f32 qmatvec, offsets, row_bytes and macro folds unchanged. |
| `hybrid_forward.rs:19911` | `moe_profile_admit_expert`. | Guard only; existing external CPU-expert implementation remains untouched and is not used for this lane's qualification. |
| `hybrid_forward.rs:19939` | `moe_frozen_gemm`, before fixed-cache hit or scratch miss. | Refuses before both paths, preserving frozen assignment and exact existing qmatvec. |
| `hybrid_forward.rs:19992` | `moe_prefetch_expert`. | Masked hints cannot reserve a cache slot through this helper. No demand heat is added by new C predictor code. |
| `hybrid_forward.rs:20016` | `moe_prefetch_disk_expert`. | Same refusal before disk/mmap source resolution and worker submission. |
| `hybrid_forward.rs:21148` | Grouped selected IDs checked immediately after unchanged masked router, before host recording and group indexing. | Prevents invalid `groups[ex]`; no replacement top-k or reordered accumulation. |
| `hybrid_forward.rs:21220` | A mixed bank with a uniform resident slab errors before the later slab branch. | **Refusal, not fallback**: avoids switching a malformed resident request into a different staged numeric program. |

## Full conversion still owed at every native boundary

The missing work is not satisfied by the guard patch or by passing CPU tests:

1. `model.rs:3158-3223` and loader construction: install immutable catalog/source
   bindings, retained file/mmap owners, original mask and all scale planes. Split
   sources use offset zero. Loader/catalog/manifest memory is charged to B.
2. `hybrid.rs:2053-2065,2835-2838`: resident slab build must retain **actual
   UniformLease ownership of the represented bytes**. A metadata boolean or a
   homogeneous selection from a PerRecord bank cannot construct that proof.
3. `hybrid_forward.rs:13676-13677,13776,13796-13837,13929-13937,14131-14136,
   14211-14234,14524-14573,17892,18342,20063,21159-21180`: pairs/dev/pointer/slab/
   grouped-decode/resident-grouped fused entry points must take `&UniformLease`
   through their actual wrappers. Retain numeric/geometry/macro/clamp/local-owner
   predicates. CPU `with_uniform_experts` alone does NOT enforce these old entry points.
4. `hybrid_forward.rs:14831-14846,14917-14965,19826-19900,21212-21550`:
   staged/SLRU/grouped demand must use BankedResidency keys and exact per-record
   offsets/lengths/qtypes/row_bytes. Native owner obtains A ReadyView/consumer
   wait, calls the same qmatvec, then retains backing until its last-use fence.
   Q2_K stays staged f32 dequant. All partial completions fail the logical batch.
5. `moe_cache.rs:1100-1155,1197-1237`: adapt existing SLRU/pending copies rather
   than instantiate another policy. Full BankId identity must replace local
   `(layer,projection,expert)` cache keys at the bridge. Unknown copy completion
   quarantines slots and source keepalives; cancel never frees active DMA.

The available A implementation is `CpuTransfers::drive`, not a CUDA owner backend.
C's new ObjectReader consumes it correctly, including per-item status and `retired`,
but it intentionally cannot produce a ReadyView. B's single Governor is now tested
with C; no replacement production governor exists. Native source installation,
SLRU adapter and owner lifecycle must be implemented together, under an isolated
rig build, before the rest of the patch can be reviewed as an executable migration.

## Exact native commands before AND after eventual full patch

Run from separately pinned baseline/candidate checkouts. Build outside GPU flock
(no inherited sccache daemon lock); run identical artifacts/prompts/numeric class
under the canonical whole-box lock. Save stdout+stderr with `tee` before parsing.
`HY3_ARTIFACT`, `HY3_KERNEL_GGUF`, `OUT`, `TEST_BIN` below are shell path parameters,
not new MEMRA reads. Qualified native Hy3 artifact must stay byte-identical. A
GGUF-only kernel fixture is supplementary, never substituted for model qualification.

```bash
set -euo pipefail
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source.txt"
cargo build --release -p memra-engine --bin kernel-check --bin run-gen --bin run-spec \
  --bin qwen4exp_gpu_gate 2>&1 | tee "$OUT/build.log"
cargo test --release -p memra-engine --lib --no-run --message-format=json \
  2>&1 | tee "$OUT/build-tests.jsonl"
TEST_BIN=$(python3 - "$OUT/build-tests.jsonl" <<'PYBIN'
import json, pathlib, sys
executables = set()
for line in pathlib.Path(sys.argv[1]).read_text().splitlines():
    try:
        row = json.loads(line)
    except ValueError:
        continue
    if (row.get("reason") == "compiler-artifact"
        and row.get("target", {}).get("name") == "memra_engine"
        and row.get("profile", {}).get("test") and row.get("executable")):
        executables.add(row["executable"])
assert len(executables) == 1, executables
binary = executables.pop()
assert pathlib.Path(binary).is_file(), binary
print(binary)
PYBIN
)
sha256sum target/release/{kernel-check,run-gen,run-spec,qwen4exp_gpu_gate} "$TEST_BIN" \
  > "$OUT/binaries.sha256"
flock /tmp/memra-gpu.lock target/release/kernel-check 2>&1 | tee "$OUT/kernel.log"
flock /tmp/memra-gpu.lock target/release/kernel-check "$HY3_KERNEL_GGUF" \
  --require-cell d2-cache-bit-identity 2>&1 | tee "$OUT/cache-identity.log"
flock /tmp/memra-gpu.lock "$TEST_BIN" model::tests::mixed_expert_loader_keeps_each_encoding_and_extent --exact --nocapture 2>&1 | tee "$OUT/layout.log"
flock /tmp/memra-gpu.lock "$TEST_BIN" model::tests::mixed_expert_loader_omits_masked_expert_bytes --exact --nocapture 2>&1 | tee "$OUT/mask.log"
flock /tmp/memra-gpu.lock "$TEST_BIN" model::tests::tiered_expert_source_does_not_double_apply_layout_offset --exact --nocapture 2>&1 | tee "$OUT/split-offset.log"
flock /tmp/memra-gpu.lock target/release/run-gen "$HY3_ARTIFACT" \
  --prompt 'Compute 17 times 23 and explain briefly.' 2>&1 | tee "$OUT/argmax.log"
flock /tmp/memra-gpu.lock env -u MEMRA_SPEC_K target/release/run-spec "$HY3_ARTIFACT" \
  2>&1 | tee "$OUT/spec-k1-8.log"
```

Verify actual test counts are nonzero, argmax MATCH and supported K=1..8 output;
process success alone is insufficient. Pin all numeric flags and both artifact
manifests, include forced nonzero misses, zero/small/full feasible caches, all
Q2_K/Q3_K/NVFP4/macro/scale surfaces and cancellation/late-fence red arms. Do not
compare cache-off f32 with cache-on q8 as if storage alone changed. The new
`banked_residency_gpu`/`table_rows_gpu` engagement gates do not exist yet; native
byte/logit/token equivalence plus serving and balanced N>=5 AB/BA measurements
remain mandatory. On a fitting 5090 fixture use `/tmp/memra-5090.lock`; full Hy3
qualification remains on the non-serving PRO pair.

### Registry mismatch found during re-entry

`docs/TESTING.md` at this source has **no named Hy3 spill-gate command section**.
Its lines 310-317 specify non-vacuous spill engagement (`q35slru`, a different
model), and line 631 notes unavailable Hy3-repack artifacts in an older count.
The exact existing Hy3-adjacent checks above come from `model.rs:3547,3570,3643`
and `kernel_check.rs:8087-8176`, not an invented TESTING entry. `kernel-check`
accepts GGUF, whereas run-gen/run-spec accept native directories. Add the correct
Hy3/native-bank entry to the lead-owned registry only with actual compiled gates.

## Day-4 v2 review — still NO-GO for full conversion

Engine files are unchanged from integration `020d2047`; the v2 patch remains
**UNAPPLIED**. No native compiler or independent reviewer has approved these
hunks. The patch generator rustfmt-parses the candidate and `git apply --check`
checks context only. Default-SLRU CPU transitions and a typed BankSource installer
now exist (SLRU.md, BANK-SOURCE.md); neither is installed into native MoeWeights.

The sole v2 extension is inside `require_active_expert`: after mask/ID checks,
validate all three projection layouts/tiers/macros vector lengths, prove the
uniform stride multiplication cannot overflow, and validate the exact source
extent using the CPU-tested `memra_tier::bank::validate_bank_source_extent`.
Split storage checks its own exact length and source offset zero; unsplit storage
checks the original layout offset against backing length. Failure returns an
error, never another tensor, qtype, fallback program or repaired mask.

`HostExps` fields used are public (`model.rs:1900–1925`), HostBuf::len exists at
1737, and the shared memra-tier dependency is already on the integrated engine
manifest. These are source-review facts, not proof that the engine compiles.
No new dispatch conversion hunk is justified by a CPU SLRU model alone. Existing
v1 callsite coverage gaps, unwrapped fused kernels, scale ownership, loader source
registration and native SLRU/pending bindings remain owed.

Prerequisites: (a) CPU source installer + byte-path fakes/tests present, native
load wiring pending; (b) CPU default-SLRU trace equivalence + charged host
BankedResidency seam present, native tuned list/slot adoption pending; (c) exact
ownership mapping in READYVIEW-OWNERSHIP.md, **not implemented or qualified**.
Full conversion remains **NO-GO until owner/consumer fence implementation lands
and passes real rig gates**, including the remaining native (a)/(b) bindings.
