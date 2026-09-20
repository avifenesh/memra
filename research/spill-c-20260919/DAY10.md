# Session C day ten: typed bank budgets and the GPU slot refusal (option A)

Scope: lead ruling 2 (`research/spill-lead-20260919/INTEGRATION-DAY10.md`) plus
the lead addendum on the native refusal token. Everything measured here is N=1
`executed-not-qualified` development evidence on one RTX PRO 6000 Blackwell
(600 W envelope) through the canonical collector; nothing is a support state.

## What changed (`79353d53d`, `1de17d41f`)

- `install_expert_bank_gate(model, gguf, budget: ExpertBankBudget)` no longer
  scans `std::env::args()`. `run-gen` and `run-spec` parse the door once with
  `expert_bank_cli`: `--experts-via-tier [--expert-bank-host-bytes=N]
  [--expert-bank-gpu-bytes=N]`. Defaults unchanged: host 256 MiB, GPU budget
  unset means native slot sizing (`MEMRA_MOE_SLOTS` or auto) byte-for-byte. A
  budget flag without the door, a bare flag, junk or a repeat is a usage error
  (`Error:`, exit 1), never a silent ignore. CLI door: no `MEMRA_*` read, no
  `docs/FLAGS.md` row; decide-by 2026-10-04 in `MOE-SLOT-CACHE-DOOR.md`.
- `gpu_bank_slots(bytes, max_record, hard_bytes)` sits next to
  `host_bank_slots` in `banked_residency.rs`: slot cost is the record plus the
  eight-byte tail pad the cache allocates; refuses no record, checked overflow,
  below `8 * (max_record + 8)`, and above `hard_bytes`; returns the exact floor
  count. `hard_bytes` is `moe_cache::hard_slot_bytes(free, max_record)`, the one
  implementation the native constructor also uses (`MEMRA_MOE_HARD_VRAM_FRAC` of
  free VRAM minus two slots), measured by the installer before any allocation.
  It is a parameter rather than a constant because it depends on the card.
- The exact count reaches the cache through `MoeSlotCache::with_exact_slots` and
  `Engine::build_moe_cache_exact`, which refuse an already built cache or a
  count below eight instead of clamping. `MoeSlotCache::new` delegates to the
  same builder with no exact count and is behavior-identical; the
  `MEMRA_MOE_SLOTS` clamp is untouched. A `MEMRA_MOE_SLOTS` value given together
  with `--expert-bank-gpu-bytes` is refused as a conflict.
- Refusal token contract (lead ruling 6, addendum): both budget refusals are the
  typed `ExpertBankRefusal`, raised before any bank, CUDA slot or source read.
  Each gate binary downcasts the installer error once (`refusal_reason`); only
  that type becomes the final stderr line `REFUSED: <reason> (requested N,
  minimum M, ceiling H)` with exit 2. Every other installer error keeps `Error:`
  exit 1. `pressure-refusal.py` (day eight) is now a red arm: it passes only the
  native token through and turns the pre-day-ten `Error:` exit 1 shape into
  `FAIL`; `test-day8.py` asserts that contract, the frozen day-eight receipts are
  unchanged.
- Self-review nit: `TracedDispatch::demand` uses `ok_or(Error::Incomplete)` in
  place of two `slru_policy().unwrap()` calls.
- `fixtures/slru-synthetic.json` re-pinned to the new `moe_cache.rs` hash by
  `slru-trace.py`; every trace row is identical.

CPU tests (`crates/memra-tier/tests/bank/day10.rs`, engine bridge include):
`gpu_bank_slots` cells no record, below one slot, seven slots, minimum minus one,
exact minimum, plus one byte, plus one slot, ceiling, ceiling plus one,
`u64::MAX`, ceiling below minimum, checked overflow; typed messages and the
`refusal_reason` downcast (plain errors map to `None`); argv parse green and red
arms. Push gate on the tree: `cargo fmt --all -- --check` clean,
`cargo test -p memra-tier -p memra-kv --offline` 264 pass (bank 58 to 61),
`DOCS_RS=1 cargo clippy -p memra-engine --offline --all-targets -- -D warnings`
clean, `tools/check-flags.sh` 864 names no uncovered, `git diff --check` clean.

## Target-card cells (`pro-single-day10-budget/`, remote `c-day10/budget/`)

Build: `/root/wt-c` fast-forwarded to `79353d53d`; `cargo build --release -j 16
-p memra-engine --bin run-gen --bin run-spec` into its own
`CARGO_TARGET_DIR=/root/wt-c/target-day10` (3m01s, nvcc 13.2, rustc 1.97.1),
receipt `build.json` + `build.log`. Day-ten binaries: run-gen
`ec513d94dfed9c94ec419134406b0fe8be190242a6970ed8f5972a9795c2ea98`, run-spec
`0685dfa17d4835c0d98143ac75840b598db37ba84cb8e910ed29e4bd22aed1ac`. The frozen
day-nine binaries (`148e7f0e9`, `b1c4090c…` / `ded88cc8…`) were never rebuilt;
`binary-manifest.json` and `binary-postcheck.json` record both sets before and
after the cells.

Every cell: `run-day10-budget.py` through `tools/tier-battery.py --rig
pro-single` only (canonical `/tmp/memra-gpu.lock`, 250 ms telemetry, 600 W
limit recorded), `env MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 run-gen <artifact> 55
88 13 --experts-via-tier <budget flag>`, no `MEMRA_MOE_SLOTS`. Largest expert
record 860,160 bytes, so one slot is 860,168 bytes and the eight-slot minimum is
6,881,344 bytes.

| cell | flag | collector | verbatim |
|---|---|---|---|
| a `budget-refuse-7` | `--expert-bank-gpu-bytes=6021176` (7 slots) | `refused`, exit 2, 79.7 s, attempt 1 | `REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum (requested 6021176, minimum 6881344, ceiling 78022085820)` |
| b `budget-exact-8` | `--expert-bank-gpu-bytes=6881344` (8 slots) | `executed-not-qualified`, exit 0, 113.8 s, attempt 1 | `prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH`; `[expert-gpu-slru] slots=8 allocated_bytes=6881344 evictions=103651` |
| c `budget-host-refuse-1` | `--expert-bank-host-bytes=1` via `pressure-refusal.py` | `refused`, exit 2, 80.4 s, attempt 1 | `REFUSED: experts-via-tier host bank budget cannot hold one expert record (requested 1, minimum 860160, ceiling 268435456)` |

Cell a: the only line before the token is the model-load mirror line; no
`[experts-via-tier] installed`, no `[expert-host-slru]`, no `[expert-gpu-slru]`,
no `loaded`, no tape. The ceiling in the message is the measured hard ceiling on
this card at install time (78,022,085,820 bytes), so the same command refuses
with a different ceiling number on a different card or with other tenants.

Cell b: `[experts-via-tier] gpu_bank_budget bytes=6881344 slots=8
hard_ceiling=78022085820` precedes `[experts-via-tier] installed ... host_slots=16
max_expert_bytes=860160`; the gen tape (32 tokens) is identical to the day-nine
native `default-gen-off` control on the same card, so the eight-slot exact bank
is the same numeric program as the native cache and the day-nine 9,986-slot
bank. Eviction accounting under an eight-slot GPU bank and a sixteen-record host
bank: GPU evictions **103,651**, host evictions **103,502**, physical reads
**103,518** (every host miss is a read, `physical_reads == misses`), re-reads
**91,440**, 103,659 host trace rows, `owner_close=Ok(())`. Day nine's 8 GiB
bank on the same prompt had 12,091 GPU evictions and 73,982 reads; the exact
minimum bank thrashes on purpose and stays correct.

Cell c: the native binary itself emitted the token with exit 2; the wrapper's
last three lines are the native token, `native_exit_code=2`, and the token
again (its pass-through), with no `Error:` and no `FAIL`. This is the same
one-byte host cell day eight banked through the old normalizer, now native.

`verify-day10-budget.py`: PASS (build receipt identity and its own target dir,
frozen binaries unchanged before and after, `git diff --quiet 79353d53d
<worktree head> -- crates/` empty, both refusal cells token-final with the
arithmetic above, exact cell 8 slots / 6,881,344 bytes / MATCH / tape equal to
the day-nine control / evictions and re-reads present, postcheck after every
cell). `test-day10-budget.py` red arms:
6 tests OK (classifier outcomes, cell budgets, green replays, and the tampered-hash, work-after-refusal, refusal-not-final, wrong-arithmetic, legacy-error-shape, wrong-token, slot-count, missing-tape and budget-after-install arms). `VERDICT.json` is the verifier output.

## Boundaries

- N=1 per cell, one card class, development evidence; no medians, no timing
  claim, no cross-box comparison. The exact-eight cell is a correctness and
  eviction-accounting cell under extreme pressure, not a performance number.
- The GPU budget is uniform-layout only and reads free VRAM at install time
  (`MOE-SLOT-CACHE-DOOR.md` pending item 2). `run-spec` carries the same parse
  and mapping but had no budget cell today; its day-nine and day-ten spec
  receipts are on the frozen binary.
- Push: `git push origin HEAD` was refused by the perf-ci pre-push gate at
  `79353d53d` (engine files touched after the last perf-ci battery; the six
  files it lists are this lane's). No `MEMRA_SKIP_PERF_CI`, no `--no-verify`;
  the lead pushes. The tree reached the target card as a git bundle over the
  lead's ssh master and was fast-forwarded in `/root/wt-c`.

Effort: approximately 2.5 agent-hours.
