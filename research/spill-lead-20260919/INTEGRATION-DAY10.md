# Integration: day 10 (`lane/spill-integ5-20260920`, replay onto current `main`)

Lead: successor of @agent-c07799 (session moved from the operator Mac to the Linux rig 2026-09-20 ~16:00Z).

## Replay
`origin/main` moved `fdb78136` → `f79b3e57` (#555 GLM TP device ownership, #558 Qwen2 tokenizer, INDEX rows).
`git merge --no-ff origin/main` into integ5 at `5f386c72` → `4f36a0e7`; no conflicts (`git merge-tree`
clean; the only file both sides touched was `crates/memra-engine/src/lib.rs`, auto-merged). INDEX.md rows
are the union.

## CPU battery on the replayed tree (`4f36a0e7`, Linux rig, `--offline`, CPUQuota 1200%)
Raw logs: `integration-day10/cpu-battery/*.log`, summary `SUMMARY.txt`.
`cargo fmt --check` OK · `cargo test -p memra-tier -p memra-kv`: 261 passed / 0 failed (63 kv + 2 + 2 + 58 bank
+ 55 contracts + 18 peer + 6 placement + 53 storage + 4 doctests) · clippy `-D warnings` (engine, server, tier,
kv, gguf; all-targets, `x86_64-unknown-linux-gnu`, `DOCS_RS=1`) clean · `check-flags.sh`: 864 runtime names, no
uncovered · publish census 12/12 · docs-registry census OK · collector Python suite 78 passed (32 subtests) ·
perf board up to date · `git diff --check` clean.

## Review state on #568
CI jobs all green on `5f386c72`. Both `revuto` inline findings (fixed VA field, pooled G1 label) are fixed at
`a799cf5d` / `eb010c87` and verified against the tip (`active.rs`: `not-applicable-pooled` on both fields;
`reclaimed = vmm_granularity != 0 && bounded_no_leak && residual_class != "unclassified"`). `revuto` then hit its
2-round cap ("Revuto did not run a review on this pull request"); Bugbot capped for the day. Per the owner
ruling of 2026-09-16 the author's self-review COMMENT is the review; merged with `gh pr merge 568 --merge`.

## BOX3 state at resync (16:10Z)
One tmux `b-day10-vmm32` (B's original-source 32k VMM cell, collector attempt 3 after three lock refusals)
holds `/tmp/memra-gpu.lock`. B's 8k VMM on the target card: `g1_reclaim_qualified=true residual_bytes=0
residual_class=none vmm_granularity_bytes=2097152` (B pushed `4c295f225`). C's four 8 GiB pressure cells:
`pressure-status.json` `state: complete`. D's smoke exit 0; D's `--validate` over all BOX3 receipts still
`REFUSED: interrupted/invalid CELL journal; not a completed capture` (a peer cell was live). A: storage cells
264/4097/1048576 captured; 4194568 and the pread baseline pending.

## Local RTX 5090 Laptop battery (`tools/local-ci.sh`, correctness stage, tree `5b3d0b08`)
Raw: `integration-day10/local-ci-5090/{local-ci.log,window.txt,local-ci.exit,hitgate-qwen/}`. CPUQuota 1200%, lock
`/tmp/memra-5090.lock` held for the whole run, no co-resident compute app at start. Verbatim verdict lines:
`kernel-check: ALL GREEN (109 cells, 10 skipped)` · `decode-batch-gate config B=8: ALL GREEN (9B NVFP4)` ·
`decode-batch-gate strict B=4 equalized: ALL GREEN (9B NVFP4)` · same two lines `(9B Q8_0)` ·
`ALL GREEN: graph-warmup-stress gate (10 cycles + canary)` · `decode-dc-gate: PASS` · `graph-decode-gate: PASS` ·
`graph-session-gate: ALL GREEN` · serve-smoke plain + `ok: cache-metering accounting exact (per-request + /metrics + economics)`
(spec / gemma4 / Q35 arms SKIP: model files absent on this rig) · **SKIPPED on this rig, models absent:** `prime-gate`, `run-spec K=1..8` (no Qwen3.6-35B), `run-gen/VERIFY-GATE/spec` (no gemma 31B), `12B run-gen/VERIFY-GATE` (no gemma 12B at the battery path); the argmax and K=1..8 coverage for this program's paths comes from the lanes' own target-card cells (C: `MATCH` / `SELF-CONSISTENCY PASS` under banked experts; B: bit-identical continuation across demote/restore) · CPU chain: memra-server suite `513 passed; 0 failed; 24 ignored`, skip-census 280 passed · `serve-stress-gate: ALL GREEN (c=64 …)` ·
`accept-gate: SKIP (no cell artifacts present on this rig: 1 cell(s))` · `SPEC-ON-CACHE-HIT GATE: 2 FAILURE(S) (qwen)`:
`FAIL: r3 spec-on text != spec-off text (identity law)` and `FAIL: g2 …`, r1/r2/g1 identical. `rc=1`, perf stage not
reached. The two failing cells are exactly the ones that fail on pristine `main` on this rig (bisected earlier today
by the exec-lanes session to `cc9d593be`, PR #379, `research/prefill-fairness-20260908`); the engine files this PR
changes are not on that path (KV writer seam, gate binaries, expert bank qualification). This run overwrote
`/tmp/memra-ci-hitgate/qwen` from that session's battery; the bisect evidence lives in `/tmp/memra-lane-logs/`.

## Second replay: `origin/main` moved again (`f79b3e57` → `8a1559b48`: #567 include_usage, #520 must-haves)
`git merge --no-ff origin/main` → `1fbdda97`; `git diff 5b3d0b08 1fbdda97 -- crates/memra-engine crates/memra-kv
crates/memra-tier` is EMPTY (main added one memra-server file), so the 5090 battery above stands for the engine
crates. CPU battery rerun on `1fbdda97`: `integration-day10/cpu-battery-remerge/`, fmt, 261 tier/kv tests, clippy
(engine, server, tier, kv, gguf) clean, flags census 864, publish census 12/12, docs-registry census, 78 collector
tests, perf board, `git diff --check`: all `rc=0`.

## Perf-ci freshness gate: announced, logged override (lead decision, owner may reverse)
The pre-push perf-ci gate compares the `perf-ci.jsonl` mtime with the newest `crates/` commit in the push range; a
merge of main into a lane is itself that commit, so every push of this branch (and of lane C, whose only engine
delta is this same merge) reds on this rig, while the operator Mac never enforced the gate (no models dir).
`tools/local-ci.sh --perf` cannot go green on any tree containing main since #379 (above). Decision: push with
`MEMRA_SKIP_PERF_CI=1` (printed by the hook, rows appended to `.git/memra-gate-skips.log` at 16:42:51Z for lane C
and at the integ5 push time), with this record and the banked 5090 correctness battery as the measurement that
was possible. The engine files the hook lists for these pushes (`hybrid.rs`, `lib.rs`, `model_memory*.rs`,
`pp.rs`) are #555's, already on main; nothing new reached main unmeasured through the skip. The exec-lanes
session deferred the same call on #565 to the owner; the difference here is that this PR's own gates ran on the
5090 (BOX2) and the target card (BOX3), and its engine delta is a door-off seam plus gate binaries.

## Lane status after resync (owner note 16:38Z: two lanes still live on the Mac → A and B by the evidence)
- **A (Mac, live):** pushed `538d9dbb` → `ed4a7d6d` → `1212246a` 19:18–19:30 +0300; `STATE.md`: eight O_DIRECT
  storage + eight N=1 pread rows exact on the block device, `storage-batch.exit` 0, no A process on BOX3;
  canonical v1.3 remains HELD (see ruling 1); io_uring DEFERRED.
- **B (Mac, live):** pushed `5afe9df6` → `b442b6c2` → `2e009eec` 19:24–19:27 +0300; `/root/wt-b` at `61eb02ac`;
  `STATE.md`: 32k VMM on the PRO card `bit-identical, 2097152 B residual, unclassified, not G1 PASS`; pooled 8k
  control `not-applicable-pooled`; diagnostic-32768 captured, injected-8192 pending. The Linux-rig B agent I had
  launched was stopped at 16:38Z when the collision surfaced; its two docs-only commits (`DAY10.md` verdict
  ledger, `ALLOCATOR-INJECTION.md`, a rewritten `STATE.md`) are parked unpushed on local `hold/spill-b-docs-20260920`
  (`6ca65010`) so the live Mac lane's `STATE.md`/`DAY10.md` are not stomped; `wt-spill-b` reset to origin.
- **C (this rig, done):** `90c7e68e` pushed by the lead (override above). Day 9 report finished, `verify-day9.py`
  PASS, `MOE-SLOT-CACHE-DOOR.md`, `BUDGET-REFUSAL.md`, one N=1 8 GiB spec-on repeat cell on BOX3
  (`pressure-spec-on-repeat-attempt6`: `=== SELF-CONSISTENCY PASS ===`, 63,996 GPU evictions / 73,966 host /
  73,982 reads / 53,684 re-reads, identical to day 9; five lock refusals kept). The Mac's C had sealed `066afe91` first.
- **D (Mac, sealed `a3aff2ae`):** G2 exit 0 (200 samples, N=10/arm, ≥497 ms visits, 37–41 °C, 600 W), smoke exit 0,
  `DAY10-VERIFICATION.md`/`G2-RESULTS.md`; "No further D GPU work required"; lead step: `tier-battery.py --validate
  /root/spill-receipts` once every peer collector is closed (refused at 16:02Z and 16:10Z on a live peer journal).
- E `4d89a243`, F `7644c41f` unchanged.

## Lead rulings, day 10
1. **A, canonical v1.3 native bindings (scope, no schedule edits).** Implement in `CudaTransfers`: (a) per-side
   graph pins (source pin, destination pin) replacing the ticket-wide `pin_graph`; `retire_source` refuses only
   while a SOURCE pin or source consumer is live; (b) destination lifetime bound to acknowledgement plus explicit
   destination-consumer retirement, not to the whole-ticket host lease; (c) call the frozen
   `conformance/revision_v13.rs` schedules (`device_hand_back`, `transfer_source_retirement`) unchanged; no fixture
   flags, no early destination drop. Qualify CPU-first with the contract crate's fake backend, then native on BOX3
   through the collector. Until all-v1.3 PASS the label stays HELD. io_uring stays DEFERRED: the N=1 pread rows
   (O_DIRECT 16 MiB cold 3,335 MiB/s vs buffered cold 2,065 MiB/s, virtio block device) are development plumbing,
   not an admission.
2. **C, bank budget refusal: option A.** Keep the `MEMRA_MOE_SLOTS` clamp; add `--expert-bank-gpu-bytes=N` to the
   gate installer as a PARAMETER of `install_expert_bank_gate` (the same commit moves `--expert-bank-host-bytes`
   from `std::env::args()` scanning to a parameter, self-review nit 4); refuse below 8 slots or above the hard
   ceiling before load finishes with a `REFUSED:` line, exit 2. No FLAGS row (CLI door), decide-by 2026-10-04 with
   the MoeSlotCache door. Option C (typed `CacheBudget`) is the right shape only if the door wins its decide-by.
3. **B, 32k residual.** The verdict stays `not G1 PASS` until the mapped-VA diagnostic classifies the one granule
   on the PRO card; the 5090 spare-VA probe is not corresponding evidence (B's own finding). No relaxation.
4. **Concurrency.** Never more than two background agents on this rig (global rule, OOM 2026-09-18); with A and B
   live on the Mac, this rig runs at most two of C/D/E/F and never a letter the Mac holds. Check
   `git fetch` + BOX3 `tmux ls`/`/root/wt-*` HEADs before every dispatch.

## integ6 (`lane/spill-integ6-20260920`, from `main` `847168642` = #568 merged)
Lane tips merged, all clean: D `a3aff2ae0`, A `b3dc864ce`, B `7d213551a`, C `0a974ea8a`, E `483f7d3da` (F `7644c41f` already in).
Merged integ branches integ..integ5 deleted on origin (each an ancestor of main); `lane/spill-c-device-day7` (2 commits
ahead, not this program's) left alone. Code delta vs main: 17 crate files, +1,366/-108 (A `tier_transfer.rs` per-side
retention + `tier_transfer_gate.rs` v1.3 schedules; B `KvAllocator` + `Cache::new_with_allocator` + mapped-VA probe;
C typed expert-bank budgets, native refusal token, `MoeSlotCache::with_exact_slots`, library-build Send/Sync assertion).
Self-review: `PR-INTEG6-SELF-REVIEW.md`.

### Headlines
- **A: canonical v1.3 native PASS on the target card**: `PASS v1.3 device_hand_back native CUDA`,
  `PASS v1.3 transfer_source_retirement native CUDA` (frozen schedules invoked unchanged, real pending CUDA producer via
  `cuLaunchHostFunc` latch, per-side graph pins, charged destination lifetime past acknowledgement); six exact roundtrips
  4 KiB–256 MiB and governor drain green; 86 remote hashes matched (`research/spill-a-20260919/day9/`).
- **B**: `ACTIVE-8K G1 PASS` on the PRO card twice (empty-plane swap and direct `KvAllocator::Vmm` construction);
  `ACTIVE-32K physical reclaim/restore bit-identical, residual 2097152 B, class unclassified, not G1 PASS`; mapped-VA
  probe: freeing and re-reserving the demoted planes' VA does not return the granule; pooled control
  `not-applicable-pooled`. The gate now grants the label only at residual 0 (stricter than criterion (d); see ruling 5).
- **C**: `--expert-bank-gpu-bytes=6021176` (7 slots) → `REFUSED: experts-via-tier GPU bank budget cannot hold the
  eight-slot minimum (requested 6021176, minimum 6881344, ceiling 78022085820)`, collector `refused`, exit 2;
  `--expert-bank-gpu-bytes=6881344` (8 slots) → `prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH`,
  `[expert-gpu-slru] slots=8 allocated_bytes=6881344 evictions=103651`, tape identical to the day-9 control; host
  refusal now native too (`REFUSED: ... (requested 1, minimum 860160, ceiling 268435456)`).
- **D**: G2 N=10/arm scored envelope + smoke sealed; **global BOX3 receipt validation now runs**: `tier-battery.py
  --validate /root/spill-receipts` → `kind: capture-integrity, cells: 32, failed_commands: 2, refused_commands: 2,
  qualification: false`, exit 0 (`integration-day10/box3-validate/`). No BOX3 process was live.
- **E**: `docs/TESTING.md` spill section rewritten from source (verdict strings quoted), `KV-PHYSICAL-RECLAIM.md`
  PRO-card status, ROUTER line, INDEX rows for A day 8/9, B day 10, C day 9, D day 10.

### Batteries on integ6 `78d2ef25`
CPU battery (`integ6-cpu-battery/`): fmt, tier/kv tests (all `ok`, incl. new `bank/day10.rs`), clippy engine/server/tier/kv/
gguf `-D warnings`, flags census, publish census, docs-registry census, collector pytest, perf board, diff-check: all `rc=0`.
Local 5090 `tools/local-ci.sh` (`integ6-local-ci-5090/`): `kernel-check: ALL GREEN (109 cells, 10 skipped)`, decode-batch
config/strict ALL GREEN (NVFP4, Q8_0), `ALL GREEN: graph-warmup-stress gate (10 cycles + canary)`, decode-dc PASS,
graph-decode PASS, graph-session ALL GREEN, serve-smoke plain + metering exact, `serve-stress-gate: ALL GREEN`; CPU chain
memra-server `513 passed; 0 failed`. SKIPPED (models absent on this rig): prime-gate, run-spec K=1..8, run-gen/VERIFY-GATE
31B and 12B, accept-gate, serve-smoke spec/gemma4/Q35 arms. `SPEC-ON-CACHE-HIT GATE: 2 FAILURE(S) (qwen)` r3/g2 = the
known main #379 regression; perf stage not reached. Push used the logged `MEMRA_SKIP_PERF_CI=1` (record above); lane C's
own engine files went to origin the same way at 17:4xZ after this battery covered them.

### Lead ruling 5 (B day 11): classify the 32k granule by repetition, do not relax the label
Run N≥5 demote/restore cycles in ONE process on the PRO card (`--kv-allocator vmm --context 32768`), recording free VRAM
after each restore. A residual that is exactly one granule after cycle 1 and does not grow through cycle N is
`one-time-driver-mapping-metadata` (non-leak); a growing one is a leak and the door is deleted at decide-by. The same
cycle cell runs on a 5090 (the local laptop card if the Qwen3.8-27B NVFP4 artifact is staged there; otherwise a rental)
so the two-card rule can be met. Only with the class on both cards does the gate line `reclaimed = … && residual == 0`
get revisited (criterion (d) accepts a classified residual); until then `not G1 PASS` stands. Decide-by 2026-10-04.

## PR #573 text from the old Mac session (their integ6 from `main` `dbf88d46`, verbatim apart from heading level and dash style)

Their commit replaced this file with the text below (101 lines of the day-10 record above were dropped). The record is restored here and their text kept as a section, so both survive on `main`.

## Integration: days 9-10 on the target card (`lane/spill-integ6-20260920`)

Lead: @agent-c07799 (old session, under the owner's "if something can merge, merge it"; the new-rig session was
told via HANDOVER-20260920 §LIVE COORDINATION). Base: `origin/main` `dbf88d46` (#568 merged). Lane tips merged:
A `b3dc864c`, B `7d213551`, C `90c7e68e`, D `a3aff2ae`, E `483f7d3d`. All merges clean.

### What this integrates (rented RTX PRO 6000 Blackwell: the target card class: 600/600 W, collector-locked,
### `executed-not-qualified`)
- **A day 8-9**: native conformance ALL PASS on the target card; six exact roundtrips 4 KiB-256 MiB; eight O_DIRECT
  storage cells byte-exact on a real block device (label: `block-device ext4 (virtio; NVMe ancestry provider-claimed,
  not proven)`); pread baseline N=1 (io_uring screening targets in `IO-BASELINE.md`, io_uring still deferred); and the
  **canonical v1.3 schedules bound natively**: `PASS v1.3 device_hand_back native CUDA`,
  `PASS v1.3 transfer_source_retirement native CUDA`: per-side retention graphs + destination charge past
  acknowledgement; frozen schedules unchanged (`V13-BINDING.md`).
- **B day 10**: BOX3 frozen baselines 8k/32k (`BOX3-BASELINES.json`); **`ACTIVE-8K G1 PASS` twice** on the target
  card (empty-plane swap, and native direct construction with 34 VMM planes); pooled control
  `not-applicable-pooled`; **32k**: bit-identical, one 2 MiB residual on this card too, mapped-VA release measured
  at 0 B → class stays `unclassified`, **not G1 PASS**. Allocator injection (`Cache` constructs VMM planes) landed
  behind the same door.
- **C day 9/10**: target-card baselines + banked experts (default and 8 GiB, gen + spec, ON/OFF) all `MATCH` /
  `SELF-CONSISTENCY PASS`; eviction counts identical to the 5090 run (deterministic SLRU trace); the existing
  `MoeSlotCache` owner-thread door audited (`Engine: Send + Sync` assertion); `BUDGET-REFUSAL.md` with three options
  for the lead decision on `MEMRA_MOE_SLOTS` (clamps to 8, cannot express refusal).
- **D day 10**: `pro-single` profile reviewed (schema enum fixed, 7 tests); **G2 scored** N=10/arm (5 AB + 5 BA),
  5 sizes × 2 directions, 200 visits ≥497 ms, 250 ms telemetry, 37-41 °C: pinned wins from 64 KiB up (1 MiB
  32.7 vs 17.5 GiB/s; 256 MiB ≈53 vs ≈37), 4 KiB within noise; `pp-transport-smoke PASS` (single-card loopback
  only); D archive `--validate` clean; bootstrap receipt: 30 exit-0 steps + one allowed prerequisite exit (not
  "all green").
- **E**: docs/ROUTER + INDEX alignment for v1.3.

### Battery (this tree, Mac, offline)
262 tier/kv tests · clippy `-D warnings` (engine, server, tier, kv, gguf all-targets, Linux target, `DOCS_RS=1`)
clean · fmt · flags census · publish census 12/12 · docs registry census · perf board current · collector Python
suite 85 passed · `git diff --check` clean.

### Gates
G1: 8k PASS on both card classes; 32k held by the unclassified one-granule residual (same on both cards; not VA
reservation). Suggested next probe: repeat demote/restore cycles in one process: a non-growing residual is
one-time driver metadata (non-leak). G2: first scored envelope exists (development evidence, one card class).
G0, G3-G7 unchanged.

### Review round on PR #573 (`55895c18`)
CI: every job pass. Automated review: six inline findings, all documentation drift against the merged tree, all
valid, fixed in the follow-up commit: `docs/TESTING.md` tier-transfer-gate section (canonical v1.3 is now bound
and passing; eleven-line verdict block; per-side pins), `--kv-allocator vmm` mechanism (direct construction;
containment is a call-site policy since `Cache::new_with_allocator` / `KvDev::alloc_vmm_u8` are public), the
mapped-VA probe receipt surface, the three probe-derived residual classes, the day-10 **zero-residual tightening
(e)** and the `verify-day10.py` pointer; `docs/decisions/KV-PHYSICAL-RECLAIM.md` scope (direct construction
landed; the surface the decide-by promotes or deletes) and criterion (e) recorded as an explicit tightening.
