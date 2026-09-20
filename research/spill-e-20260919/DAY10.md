# WP-E day 10: contracts and docs alignment (2026-09-20)

Lane `lane/spill-e-20260919`, worktree `wt-spill-e`. Docs only: no runtime, kernel, dependency or
environment-read change. Base: lane tip `4d89a2434` (already on `origin/main`), then a `--no-ff` merge of the
local lead branch `lane/spill-integ5-20260920` at `9620f1663` (integ5 replayed onto main `8a1559b48`) as
`6973ac5a5`, no conflicts. `research/spill-e-20260919/` did not exist before this session; E's earlier seals
live under `research/spill-lead-20260919/` (`FREEZE-V1.3.md`, `V13-HANDOFF.md`, `PR560-REVIEW.md`).

## What changed

1. `docs/TESTING.md`, section "Generic spill / tiered KV (memra-tier)" (commit `371ee06bb`). Rewritten
   into: CPU gates (conformance path drift fixed: the schedules are the public module
   `crates/memra-tier/src/conformance/mod.rs`, not the removed `tests/contracts/conformance.rs`; the
   collector pytest line added), native conformance and the HELD label, `kv-tier-gate`, the
   experts-via-tier gate, `h2d-probe --copies`, the collector, and boundaries.
2. `docs/decisions/KV-PHYSICAL-RECLAIM.md` (commit `3a885c28b`). Added the target-card status block
   from lane B day 10 and a "G1 as of 2026-09-20" line; criterion (a) to (d) and decide-by 2026-10-04
   unchanged; em dashes removed (five). Nothing relaxed.
3. `docs/ROUTER.md` (commit `5de492708`). One routing line for tiered-KV / spill gates; 42 lines (cap 60).
4. `research/INDEX.md` (commit `735ec83f9`). Five sub-rows added (`spill-a-20260919/day8`, `/day9`,
   `spill-b-20260919/day10`, `spill-c-20260919/day9`, `spill-d-20260919/day10`) with quotes trimmed by
   `...` and never reworded, per the file header; every existing row kept byte-for-byte.

## What each statement was verified against

| Statement in the docs | Source in this tree (or lane tip named) |
| --- | --- |
| `kv-tier-gate` usage line, cases, contexts 8192/32768 (16384 only diagnostic), `--tiers host|host,nvme`, `--same-program` mandatory, `--kv-allocator pooled|vmm` default pooled, `--reclaim-diagnostic` requires active VMM host | `crates/memra-engine/src/bin/kv_tier_gate/cli.rs` (`USAGE`, `parse`, tests) |
| Prefix / non-host active refusal string; naked-baseline `MEMRA_*` refusal (allowlist `MEMRA_NVCC`, `MEMRA_CUDA_ARCH`, `MEMRA_GPU_LOCK`); `program=native-decode_step_h-tokenwise-trunk-no-mtp`; `REFUSED.txt` written for every error; exit 2 | `crates/memra-engine/src/bin/kv_tier_gate.rs` lines 172-186, 218, 344-362 |
| `BASELINE_CAPTURED`, `ACTIVE_RECLAIM_CAPTURED; ...`, `ACTIVE_COPY_RESTORE_CAPTURED; ...` first lines; binary prints no G1 label | `kv_tier_gate.rs` lines 322-328 |
| `residual_class` values, `g1_reclaim_qualified` and `residual_bytes` `not-applicable-pooled` for pooled runs, `reclaimed` requires classified residual | `crates/memra-engine/src/bin/kv_tier_gate/active.rs` lines 506-540 |
| Criterion (a) to (d) as code | `crates/memra-engine/src/bin/kv_tier_gate/reclaim_contract.rs` |
| G1 label assigned offline by the lane verifier | `research/spill-b-20260919/verify-day9.py` lines 47, 240-241 |
| `cli::diagnostic` unwraps `REFUSED:` and prefixes others with `kv-tier-gate: `; collector regex `^(?:kv-tier-gate: )?REFUSED: .+` on the last line with exit 2 | `cli.rs` `diagnostic`; `tools/tier-battery.py` `explicit_refusal` |
| Collector `--rig` choices and default `pro-pair`; `LOCKS` table (`/tmp/memra-5090.lock`, `/tmp/memra-gpu.lock`); `flock(LOCK_EX|LOCK_NB)`; `--external-lock` token `@COLLECTOR_LOCK_FD@`, not with `--resume`, lock proof device/inode; statuses `executed-not-qualified`/`failed`/`refused`; `failure_quote` regex; `qualification: false`; nonzero exit on failed/refused | `tools/tier-battery.py` lines 15, 180-191, 336-384, 886-951 |
| `REFUSED: [Errno 11] Resource temporarily unavailable` on lock contention | `tier-battery.py` main exception handler (`REFUSED: {error}`, exit 2); observed in `research/spill-c-20260919/DAY9.md` on `lane/spill-c-20260919` |
| 250 ms telemetry (`-lms 250`, `telemetry_interval_ms == 250`, CPU fixture cannot invent GPU telemetry) | `tier-battery.py` lines 82-88, 300 |
| `--validate` outputs (`capture-integrity` JSON, `REFUSED: interrupted/invalid CELL journal; not a completed capture`, `CAPTURE INTEGRITY MATCH; ...`, `BYTE-RECEIPTS MATCH`, `TELEMETRY MATCH`) | `tier-battery.py` lines 588, 969-1023 |
| No collector medians (`performance_medians_allowed: False`; dry-run `dry-run-not-qualification`); five AB + five BA minimum | `tier-battery.py` `first_hour_plan`, `run_dry_campaign`, `paired_orders`; `tools/tier-envelope.py` docstring |
| N and regime on every published median | `research/spill-d-20260919/G2-RESULTS.md` on `lane/spill-d-20260919` (`N/arm`, `Regime` columns) |
| `tools/tier-rig-bootstrap.sh --rig rtx5090|pro-single` and its locks | `tools/tier-rig-bootstrap.sh` lines 34-50 |
| `overlay-unproven` storage class | `tier-battery.py` line 499 |
| `memra_tier::conformance` public module re-exporting v11/v12/v13; CPU bindings | `crates/memra-tier/src/lib.rs` line 12; `src/conformance/mod.rs` lines 107-117; `tests/contracts/{transfer,services,v12_bindings,v13_bindings}.rs` |
| Eight native verdict lines | `crates/memra-engine/src/bin/tier_transfer_gate.rs` lines 88-255; recorded in `research/spill-a-20260919/day8/RESULTS.md` on `lane/spill-a-20260919` |
| Gate does not call `device_hand_back` / `transfer_source_retirement` | `rg` over `tier_transfer_gate.rs`: zero matches; the functions exist in `src/conformance/revision_v13.rs` lines 17, 87 |
| HELD reasons (quoted) | `research/spill-a-20260919/day8/RESULTS.md`, "Why canonical v1.3 is still HELD", on `lane/spill-a-20260919` |
| A day 9 `PASS v1.3 ...` lines, source `1adf2be3d` | `research/spill-a-20260919/day9/RESULTS.md` on `lane/spill-a-20260919` tip `b3dc864ce` |
| v1.3 schedule names, `retire_source` default `Unsupported`, `WIRE_VERSION` 1 | `research/spill-lead-20260919/FREEZE-V1.3.md` sections 2 to 4 |
| `install_expert_bank_gate` refusal strings, approved SHA check, `cache.install_banked`, install line, drop lines `[expert-gpu-slru] ...` and `[experts-via-tier] physical_reads=...` | `crates/memra-engine/src/banked_residency/native.rs` lines 60-75, 85-135, 235-247, 303-309 |
| `host_bank_slots` two refusals, 16-record cap, 256 MiB ceiling; default budget 256 MiB read from process args | `crates/memra-engine/src/banked_residency.rs` lines 125-135; `native.rs` lines 239-247 |
| `run-gen` / `run-spec` directory refusals; error surfaces as `Error: "..."` exit 1 via `main`'s `?` | `run_gen.rs` lines 123-134, 1022-1026; `run_spec.rs` lines 149-158 |
| `MEMRA_MOE_SLOTS` clamp to at least 8 | `crates/memra-engine/src/moe_cache.rs` line 533; `research/spill-c-20260919/DAY8.md` seam note |
| Normalization to `REFUSED: experts-via-tier host bank budget cannot hold one expert record`, exit 2 | `research/spill-c-20260919/pressure-refusal.py` lines 15-16; `DAY8.md` "Refusal and GPU-budget seam note" |
| `MATCH` / `=== SELF-CONSISTENCY PASS ===` with eviction engaged | `research/spill-c-20260919/DAY8.md` lines 78-82 |
| `h2d-probe` options, `--copies 1..100000` default 1, `n` 1, `n1-plumbing-not-qualified`, `sum-per-operation-owner-stream`, RESULT fields, collector wrap 300 s | `crates/memra-engine/src/bin/h2d_probe.rs` lines 1-3, 60-63, 118-134, 278, 410-414 |
| One native N=1 matrix, 32 visits, no medians | `research/spill-f-20260919/H2D-RESULTS.md` |
| Target-card G1 numbers, mapped-VA probe rows, pooled control, direct construction 34/0 planes | `research/spill-b-20260919/DAY10.md` on `lane/spill-b-20260919` tip `7d213551a` |
| RTX 5090 numbers in the decision record | unchanged from the record; `research/spill-b-20260919/DAY9.md` in tree |

## Gate runs on the final tree (verbatim)

```text
docs-registry-census: KERNELS.md file references=122, all resolve
docs-registry-census: MODELS.md support-state tokens=5, all in the three-state vocabulary
docs-registry-census: ROUTER.md lines=42 (cap 60)
docs-registry-census: check-flags --list answered with 864 runtime names (coverage enforced by check-flags.sh itself)
docs-registry-census: flags-table-census: docs/FLAGS.md tables=58 rows=899, every row matches its header
check-flags: runtime literal reads=864
check-flags: no uncovered runtime names
check-flags: every runtime MEMRA_* name resolves against 'docs/FLAGS.md' (no grandfather list)
perf board is up to date
```

`git diff --check`: clean. `python3 -m pytest -q crates/memra-tier/tests/battery/`: `78 passed, 32 subtests passed`.

## Left out because it was not verifiable in the tree, or is lane evidence pending integration

- Lead ruling 2 of day 10 (`--expert-bank-gpu-bytes=N`, budget as an installer parameter) is not in the
  tree (`rg expert-bank-gpu-bytes crates/` finds nothing); TESTING.md documents the argv scan that exists.
- Lane B's `Cache::new_with_allocator` direct construction and mapped-VA probe, lane A's day 9 canonical
  v1.3 bindings, lane C's day 9 target-card cells and lane D's G2 medians are on their lane tips, not in
  this tree. They are named as pending integration where cited; the HELD label and the G1 status in this
  tree are stated from the tree.
- `CAPTURE ARCHIVES MATCH: ...` (INDEX row for D day 9) is not a string in `tools/tier-battery.py`; the
  collector's directory validation prints the `capture-integrity` JSON. TESTING.md documents the JSON.
- INDEX rows 509 and 516 (B, lead) carry em dashes inherited from #568 inside verbatim label quotes; kept
  unchanged under "keep existing rows". New rows avoid the dash by quoting other verbatim sentences.
- The write-up files for the new INDEX rows (`spill-a-20260919/day8/RESULTS.md`, `day9/RESULTS.md`,
  `spill-b-20260919/DAY10.md`, `spill-c-20260919/DAY9.md`, `spill-d-20260919/DAY10-VERIFICATION.md`)
  resolve on the lane tips and will resolve here once integ6 lands.
- The brief cited A day 8 as the HELD source; A's tip is now day 9 (`b3dc864ce`). TESTING.md quotes the
  day 8 reasons and names the day 9 result as pending integration, without softening the tree label.

## Push

`git push origin HEAD` was refused by the pre-push perf-ci gate at `6973ac5a5` (the lead merge puts 22
engine files in the range) and again at every docs commit after it; verbatim:
`pre-push: engine files touched after the last perf-ci battery.` No `MEMRA_SKIP_PERF_CI`, no
`--no-verify`. The lead pushes the branch. Flags, releasability and docs-registry hook arms all passed on
each attempt.

Approximately 1.6 agent-hours.
