# Generic-spill pre-work — day-3 integration record (lead: agent-c07799, 2026-09-19)

Branch `lane/spill-integ-20260919`; base `c5a33b14` (main). Merged today, in order, on top of
`914229ae` (engine `memra-tier` dep + `storage-bench` bin — A's fragment, found missing by D):
A `6ef57399` (brings v1.1 `lane/spill-lead2-20260919` @ `ce49cd86`) → B `40f34e47` → C `e7188cd6`
→ D `b8038b36` (directed-grant fix) → C `5784d11f` (v1.1 binding fix).

## Two integration errors made and corrected by the lead (recorded, not hidden)
1. Day-2: scripted union-resolve dropped B's `pub mod tier;` — caught by inspection, restored.
2. Day-3: union-resolve of `crates/memra-tier/tests/bank/main.rs` duplicated four `revision_v11_*`
   binding fns (E0428) — resolved by taking C's file wholesale (C owns it; superset), merge amended.
Lesson: union-resolve only additive `pub mod`/`[[test]]` lines; for code files take the owning
lane's version.

## v1.1 strict schedules: three real findings, all closed on this tip
| Lane | Finding | Fix |
|---|---|---|
| A | quarantined pinned backing freed on unknown last-pool Drop (`revision_v11.rs:215`, `Ok(())` vs `Err(Busy)`) | `b0296f9a`: retain backing + governor pin through unknown shutdown |
| D | fake `PeerCapacity` had one global `grants: bool` → reverse route denied with forward (`revision_v11.rs:567`) | `2bb7524c`: per-direction context/pool/link-health grants; reject before charge |
| C | v1.1 bank bindings assumed synchronous `stage`/`gather`; day 3 is enqueue-then-`progress` | `4b50d980`: bindings drain progress before observing; no production/schedule change |

## Integrated CPU battery (this Mac; no nvcc, no GPU)
memra-kv 51 · memra-tier: bank 35, contracts 44, peer 17, placement 6, storage 37, doctests 4 — all
pass, `--no-fail-fast`, 0 ignored. clippy `-D warnings` OK · fmt OK · `git diff --check` OK · flags
census 864 reads / none uncovered · Linux cross-target `cargo check` OK (compile only) · `cargo
metadata` OK · `py_compile` tools/tier-*.py OK · `bash -n tools/tier-rig-bootstrap.sh` OK.
80 commits over base; 815 files, +36,107.

## NOT run (stated): engine/server native build (nvcc), any GPU cell, any model load, Linux
O_DIRECT execution, io_uring. B's `HOSTPREFIX-PATCH.diff` and C's `HY3-DISPATCH-PATCH.diff`
(partial, NO-GO for full conversion) remain UNAPPLIED reviewed patches. G0–G7 all pending.

## Ready for the first rented 5090 hour (owner-approved Vast/RunPod spot, non-production)
`tools/tier-rig-bootstrap.sh` (two 8 GiB cudaMalloc acceptance runs, power-limit re-read, branch
refusal, canonical lock) → blocking native engine+server compile at this tip → A `storage-bench` →
D1 local bytes → C PLE → B Qwen3.8 fitting cells; PRO-pair-only cells listed in
`research/spill-d-20260919/RIG-DAY1.md`. Runners: `research/spill-{a,b,c}-20260919/rig-cells-*.sh`.
Blocked on: a Vast/RunPod session or API key on the operator Mac (none present; Chrome shows
Vast logged out, RunPod renders blank in background mode).

## Carry-forward for day 4
- D: export the `PeerCapacity` fake as a construction seam for B's combined B→D schedule (B ask).
- A: io_uring dependency proposal → lead decision; async facade follow-up; per-extent catalog/sharding.
- C: native conversion prerequisites (loader source installation, SLRU-backed `BankedResidency`,
  CUDA ready-view / consumer-fence ownership) — real scope for the remaining WP-C days.
- Lead: push `lane/spill-integ-20260919` to origin when the rig is imminent (pre-push hook runs
  the perf-board check + flags census; never `--no-verify`).
