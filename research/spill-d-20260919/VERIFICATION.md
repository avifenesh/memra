# Day-1 verification receipt

Repository `avifenesh/memra`, implementation commit
`646200bd` (full SHA in `cpu-checks.jsonl`), branch `lane/spill-d-20260919`.
This is a **local committed CPU scaffold**, not a merge, deployment, live verification,
GPU transport implementation, or generic-tier GO. Stop at day-1 milestone pending lead resume.

## Commands actually executed

`cpu-checks.jsonl` records exact argv, UTC start, implementation SHA, exit status and raw-log
path for the final check set. These are CPU-check receipts, not GPU `runs.schema.json` rows.
All output was captured to raw logs before inspection; no parser could hide command status.

| Check | Actually ran / result |
|---|---|
| `cargo fmt --all -- --check` | Yes, exit 0. Existing root workspace only; new crate is deliberately not wired by D. |
| `rustfmt --check --edition 2021` on all five D Rust source/test modules | Yes, exit 0. Covers new modules outside root workspace. |
| `bash research/spill-d-20260919/verify-cpu.sh` | Yes, exit 0; standalone temporary Cargo scaffold, offline `cargo check`, offline `cargo test --tests`: **8 peer + 5 placement tests passed**, zero failures/ignored/filtered in both suites. |
| Python receipt tests inside above | **7 tests passed**. Actual CLI positive fixture accepted; deliberate corruption returned exit 2 and `evidence hash mismatch`; missing/empty/partial evidence, identity mismatch, fake P2P, wrong lock, non-engagement and schema drift rejected. |
| Proposal Rust facade compile inside above | Yes, `rustc --edition=2021 --crate-type=lib --extern memra_tier=…`, exit 0. |
| `tools/check-flags.sh` | Yes, exit 0; **864 runtime literal reads; no uncovered runtime names**. No new env reads. |
| `git diff --check` | Yes, exit 0. Also checked staged additions before implementation commit. |
| `bash -n research/spill-d-20260919/verify-cpu.sh` | Yes, exit 0. |
| Python AST/JSON schema parse | Yes, exit 0; two Python ASTs + schema parsed. |
| `python3 -B tools/tier-battery.py --plan` | Yes, exit 0; emitted **pending-gpu-adapters**, standard gates and case roster. No GPU execution. |

Earlier local standalone-scaffold check/test logs are preserved under `raw/` as well.
`raw/cargo-test.log.gz` losslessly preserves the earlier Cargo output (including its blank
terminal line); compression avoids a new-blank-line-at-EOF `git diff --check` violation.
The first placement test run caught incorrectly hand-entered expected byte integers in the
new test (GPU0/GPU2 off by 320 B, GPU1 by 160 B). Independent Python arithmetic reproduced
Rust's result, and the expected integers were corrected before the implementation commit.
This did not change 07's rounded GB figures or N29/T1,037,290 frontier. Final raw test output
is `raw/verify-cpu-final.log`; the earlier failed output was observed in the session, not
captured into a raw file, so no raw failure receipt is invented.

## Scope / hygiene

- Source and test changes only in D's reserved peer/placement/test paths; tooling only
  `tools/tier-battery.py`; evidence only this research namespace.
- No edits to `pp.rs` or `tp_transport.rs` yet; no GPU/FFI/kernel/default changes.
- Root `Cargo.toml`/`Cargo.lock`, tier `Cargo.toml`/`lib.rs`/`contracts.rs` and all shared
  docs/INDEX/hooks are untouched in commits. Temporary local tier scaffold was removed.
  Reproduction script now creates/removes a standalone scratch crate within this research
  namespace without touching any lead-owned file. Target artifacts and Python caches removed.
- Lead-owned Cargo/test/doc fragments are in `CONTRACTS-PROPOSAL.md`. Shared structs/exports
  need freeze before integration; no production consumer uses these proposals yet.
- No push, PR, tag, live instance access or GPU lock acquired. No SSH from D; lead's launch
  connectivity failure is inherited context, not a D topology measurement.
- No root-worktree changes; dedicated worktree/branch intentionally retained for lead resume,
  not a closed/abandoned lane. Unrelated changes were not staged.

## Open gates

D1 actual native P2P, pool/context grants, transfer cancellation and graph address lifetimes;
D2 official Step-3.7 FP8 PP ladder + Qwen tier consumers; D3 integrated four tiers then all
12 four-card routes/shared-fabric campaign; D4 predicted-versus-allocated bytes are all
pending. ModelPlan/engine/server integrated suites and required `step-pro` are not run by
this CPU-only day-1 slice. G0–G7 remain pending as an integrated program.

## Lead decisions needed

1. Freeze shared handles, per-item acceptance and cancellation/publication semantics; decide
   borrowed-release proposal and source-vs-destination-vs-state epoch separation.
2. Assign authoritative StateBundle/RecordLayout and governor permits, then approve D's
   adapter binding without introducing duplicate contracts or another allocator.
3. Supply pinned official Step/Qwen artifacts and exclusive non-serving PRO pair access;
   four-card topology gate waits for target availability. No 5090 veto for Step-only delivery.
4. Bind the GPU case roster to real engine/serving adapters and full telemetry/failure collector.
   `ppn-gate` is GGUF-only at baseline; official FP8 PP coverage uses the safetensors-aware
   `decode-batch-gate`, with official eager-PP coverage still to be reconciled.
