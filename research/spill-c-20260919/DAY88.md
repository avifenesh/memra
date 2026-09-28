# WP-C day 88 (2026-09-27): OWED C1(c) ruled PROMOTE; C2, the promotion work, phase 1 registered before any code

Owner rulings, 2026-09-27, verbatim through the lead: "accept all clear ones. A. promote." The lead's reading for this
lane: (1) C1(c): promote the MoE slot cache door; C2 is now the promotion work; the door becomes the naked MoE spill
program; the legacy SLRU slot cache stays as the rollback seam (one flag with a FLAGS row and a decide-by 14 days
out, deleted after two unused weeks). (2) `DAY80.md` section 4a accepted: the registered pool (private anonymous
memory registered with `cuMemHostRegister`, portable) is the door's default host pool; today's `cuMemHostAlloc` pool
goes behind `--expert-bank-pool-allocated` with its decide-by. (3) C10 accepted: `MEMRA_MOE_PREFETCH=1` becomes the
naked default on both cards, `=0` its seam, with a decide-by; reconciled with the promotion, one numeric program per
request. Tree at start: `6328a15ea` (the lane with `main` `80f734c77` merged in).

## 0. The measurements the rulings rest on (the decision record copies these)

- **C1(c).** The deciding cell, `DAY51.md`: `DAY51 VERDICT rig=pro-single integrity=ok -> door_wins` and `DAY51 VERDICT
  rig=rtx5090 integrity=ok -> door_flat`. The door against REF (the legacy with its prefetch), tuned since:
  `DAY85.md` section 5, `i22=improves door=i22 vs_ref=matches` on the 9950X class (BOX43) and `i22=flat door=i22
  vs_ref=matches` on the 285K class (BOX41), the door 1 to 2 ms behind REF over 32 tokens gen-only by the medians
  (9 to 10 at I15).
- **The pool.** `DAY80.md`: `DAY80 REGPOOL VERDICT rig=box37-285k integrity=ok -> registered_clears`; `DAY80 REGTIME
  VERDICT ... dr=flat` on the 285K (BOX37) and on the 9950X (BOX38). `DAY86.md`: whether the recent cuts raise the
  compaction state's rate on a long-running 9950X stays open (`not_reproduced` on BOX43).
- **C10.** `DAY59.md`: G1, G2 and G3 PASS on both cards; `DAY59 VERDICT rig=pro-single shape=pftime integrity=ok ->
  pf_wins`, `... shape=pfnaked ... -> pf_flat`, and the same two verdicts on `rig=rtx5090`.

## 1. The scope, and the rule it answers to (surfaced to the lead)

Every one of those receipts is on one artifact: Qwen3.6-35B-A3B-UD-IQ4_XS, SHA-256 `df27a780...7adf`, the installer's
lock. The standing rule "no generic-model support claims" forbids inferring the door's or the prefetch's default for
another artifact or family from these. So phase 1 makes the door, the registered pool and the prefetch the naked program
**on both cards for a qualified artifact**, identified by its digest (a qualified-artifact list in the installer, one
entry today), and every other artifact keeps today's program unchanged. An artifact joins the list only with its own
census, gates and receipts (C2 item 3). The list is an identity the tensor contract already binds, not an
architecture-name allowlist.

Also outside phase 1, as the door document's pending items state them (C2's remaining work, each registered on its
own): `memra-server` has no installer (item 6: a serving installer and its serving-shape bit-identity gate, banked
against native, solo against batched); a PP stage split refuses the owner registry (item 1); mixed layouts and scale
planes are refused (items 2, 3, 5). Under phase 1 each of those keeps today's program, and says so on one printed line.

## 2. Phase 1: the flip

**2.1 Qualification, before the model loads** (`run-gen` and `run-spec`). The gate binary opens the GGUF, then, unless
`MEMRA_EXPERTS_VIA_TIER=0`: when `MEMRA_MOE_CACHE` is on, the artifact is one shard and no PP split is configured (`MEMRA_PP_STAGES` below 2),
and the file's byte length equals a qualified entry's, it hashes the already-open inode (the installer's own SHA-256)
and compares it with the entry's digest. Only a file of a qualified length is hashed, so no other artifact pays for it.
A match yields a typed `QualifiedArtifact` (the digest and the opened file's device and inode) that the installer
accepts instead of hashing the file again; the result is one printed line, `[experts-via-tier] qualified <sha>` or
`[experts-via-tier] off: <reason>`.

**2.2 The door by default.** For a qualified artifact the expert banks load as views of the artifact's mapping (day 44's
load option). After load, if the MoE experts went to the SLRU slot cache (not resident), the door installs with the
default budget: the host tier holds the whole bank (every record, the shape every card cell measured) when that fits
under the host ceiling (three quarters of `MemAvailable`, day 43's); if it does not, the door is off with a printed
reason and the legacy program runs from pinned copies (decided before load, from the bank's size in the tensor table
and `MemAvailable`, so the legacy never stages from mapped views; the installer then plans against that same ceiling
reading, so the decision and the plan cannot disagree). If the experts are resident, the door does not
install (`[experts-via-tier] off: experts resident`); the resident upload reads the same bytes from the mapped views
instead of a pinned copy (cell `promo-res` reads it). `--experts-via-tier` stays as the gate's assertion: with it, a
door that does not install refuses (`REFUSED`, exit 2) instead of running the legacy. The budget flags stay machine
config and apply to the default door; with the door off they refuse as a usage error, never a silent no-op.

**2.3 The rollback seam.** `MEMRA_EXPERTS_VIA_TIER=0` (a `docs/FLAGS.md` row, decide-by 2026-10-11): the legacy SLRU
slot cache from pinned copies, byte for byte today's legacy program, with the prefetch as 2.5 sets it (REF).

**2.4 The registered pool by default.** The door's host pool is the registered kind; `--expert-bank-pool-allocated`
selects today's `cuMemHostAlloc` pool (decide-by 2026-10-11 in the door document, a CLI door). What retires, with
"Removed doors" rows: `--expert-bank-pool-registered` (its kind is the default), `--expert-bank-pool-pageable` (its
census answered, `DAY78.md`) and `--expert-bank-pool-chunk-bytes` (`chunk_does_not`, `DAY76.md`).

**2.5 The prefetch, reconciled.** One meaning on both paths: `MEMRA_MOE_PREFETCH` governs the in-token expert prefetch
of the slot cache, whichever program serves it. Default: on for a qualified artifact (the door's owner-routed grouped
prefetch under the door; the legacy's copy-stream prefetch under `MEMRA_EXPERTS_VIA_TIER=0`), off for every other
artifact, as today. `=0` turns it off on both paths (under the door, day 50's demand-only program); `=1` turns it on
for any artifact, as today. Its other readers keep their meaning (the CPU experts' predictor depth `=2..8`, the worker
spill backend's disk lookahead). `docs/FLAGS.md` row updated, decide-by 2026-10-11.

**2.6 One numeric program per request.** Every choice above is made once, before the first token, and latched for the
process: no request can cross from one residency or prefetch program to another. Neither choice changes a kernel or a
byte a kernel reads (the door and the legacy stage the same record bytes into the same slots; the prefetch only moves a
copy earlier), so every arm of every cell below must read one tape, as every door cell has.

**2.7 The records.** `docs/decisions/MOE-SPILL-DOOR-DEFAULT.md` (chosen, rejected, the measurements of section 0, the
scope of section 1); `docs/FLAGS.md` (the two rows, the removed doors); `MOE-SLOT-CACHE-DOOR.md` (status: promoted on
qualified artifacts in the gate binaries, the pending items renumbered as C2's phases); `docs/models/qwen36-35b-a3b.md`
and the two rig cards the default touches (`docs/rigs/rtx-5090.md`, `docs/rigs/rtx-pro-6000-blackwell.md`). No `.cu`
file changes.

## 3. CPU gates before any card (`day88-cpu/`)

The qualification's unit tests (a length mismatch never hashes; a digest mismatch is `off`; the token refuses a
different inode); the CLI's new arms (the budget flags without an installing door, `MEMRA_EXPERTS_VIA_TIER=0` with a
budget flag, `--experts-via-tier` with the door off, each a typed refusal); the prefetch default's decision table (unset,
`0`, `1`, `3`; qualified and not); the pool kinds with the retired flags refused as unknown; the census tests that pin the
forward's prefetch condition and the installer's `set_expert_bank_prefetch` updated to the new condition; the tier
suites, the engine library, clippy (`-D warnings`, all targets), fmt, `tools/check-flags.sh`, `rc-scan.py --live`.
Builds under `nice -n 19`, a 600% CPU quota and `MemoryMax=12G` (four lanes share the 60 GB rig).

## 4. The local RTX 5090 check

The promoted binary and I22 (`4b378a064`, the door as the cells qualified it: `--experts-via-tier
--expert-bank-host-bytes=17179869184`), the cells' spill argv, both orders: every run `MATCH`, one tape, one host
demand sequence; the naked run prints `qualified`, `installed` and the registered pool; `MEMRA_EXPERTS_VIA_TIER=0` prints
`off` and runs the legacy with its prefetch.

## 5. The cells on both host classes (registered here, before their scripts)

**Cell `promo`, the spill shape** (the cells' argv: `MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32 MEMRA_MOE_SLOTS=9986`, prompt
`55 88 13`). Binaries: `p88` (the promotion) and `i22=4b378a064`. Arms:
- `naked`: `p88`, no flag (the door, the registered pool, the prefetch, all by default);
- `q22`: `i22` with `--experts-via-tier --expert-bank-host-bytes=17179869184` (the door as qualified; the same program);
- `legacy`: `p88` with `MEMRA_EXPERTS_VIA_TIER=0` (the rollback: the legacy with its prefetch, REF);
- `alloc`: `p88` with `--expert-bank-pool-allocated`;
- `nopf`: `p88` with `MEMRA_MOE_PREFETCH=0` (the door without the prefetch);
- `legnopf`: `p88` with `MEMRA_EXPERTS_VIA_TIER=0 MEMRA_MOE_PREFETCH=0` (the legacy naked before C10).

Order 1 (naked, q22, legacy, alloc, nopf, legnopf) x 5, order 2 reversed x 5: 60 runs, one collector hold.
Integrity: every run exits 0 with `MATCH`, one tape across all 60, the door arms (naked, q22, alloc, nopf) one host
demand sequence with the fill complete and `physical_reads=0`, the qualification and installation lines where they
belong and the `off` line in the legacy arms, each arm's pool line as its flags say. Admissibility: DAY64's clause (every
arm's gen-only and window IQR at most 0.005 s). Readings, DAY61 section 2's rule (gen-only primary, the window beside):
- `naked` against `q22`: the promotion changes no program; `regresses` blocks the promotion;
- `naked` against `legacy`: the default against its rollback (`DAY85.md`'s question; `loses` is recorded and blocks);
- `alloc` against `naked`: the pool's rollback (DAY80 read it flat);
- `nopf` against `naked` and `legnopf` against `legacy`: the prefetch seam on each path (C10's pf_wins predicts
  `improves` for the prefetch; recorded, deciding nothing further).
Also per run, beside and deciding nothing: the wall from process start to the first generated token (the door's
startup, its SHA lock included, against the legacy's pinned copies).

**Cell `promo-res`, the resident shape** (the same argv without `MEMRA_MOE_RESIDENT=0` and `MEMRA_MOE_SLOTS`, where the
experts fit the card): arms `naked` and `legacy`, order 1 x 5 and order 2 x 5. Integrity: `MATCH`, one tape, the
`off: experts resident` line in `naked`. Reading: decode `naked` against `legacy` (the mapped views change no resident
decode; `regresses` blocks), and the load and first-token walls beside it.

**Cell `promo-spec`**: `run-spec` with `p88`, no flag, the spill shape, K=1..8: `=== SELF-CONSISTENCY PASS ===` with
the door installed by default, and the same with `MEMRA_EXPERTS_VIA_TIER=0`.

**Where.** The target card on the 285K class, then a 9950X (a fresh host; a long-running one that shows C12's signature
runs `slow86` first, `DAY86.md` section 2); the local RTX 5090 runs the same three cells as the development check.
**What they decide.** All three pass and `promo`'s `naked` neither regresses against `q22` nor loses to `legacy`: phase
1 lands as the naked program on both cards. Otherwise phase 1 does not land, and the reason is its receipt.

## 6. Phases 2 to 4 (C2's remaining items), each registered on its own before its code

Phase 2, the serving installer (item 6) with its serving-shape bit-identity gate. Phase 3, the owner under a PP stage
split (item 1). Phase 4, installer generality (item 3: a second qualified artifact with its census and receipts, and
scale admission for the Hy3 and Step ladders), mixed-layout budgets (item 2) and the refused arms (item 5).

## 3a. Phase 1 on the CPU, before any card

Phase 1 landed as `9c20f327d` and `0155bc69f` (the second makes the door's plan print the prefetch's effective state
beside its default, log only), on the lane with `main` `80f734c77` merged in (`6328a15ea`). As registered, with four
details the source gave:
- The qualification is a free function over the opened file (`qualify_file`) behind `Engine::qualify_expert_door`, so
  its decision table is a CPU test; the gate binaries call `Engine::plan_expert_door` before load and
  `Engine::install_expert_door` after, one path for `run-gen` and `run-spec`.
- The qualified entry carries the whole bank's bytes per catalog (`run-gen`'s trunk: 15,219,032,064 bytes over 30,720
  records; `run-spec` with the MTP head: 15,600,713,728 over 31,488, as the installer's `host_bank_plan` lines read
  them on BOX39 and in DAY52's spec cell); the installer refuses if its catalog's bank differs from the entry.
- The door's plan sets the prefetch default from the artifact's identity even when the door is off (the rollback, or
  a bank over the ceiling), so the legacy then runs with its prefetch (REF, the qualified program).
- `--experts-via-tier` with a door that does not apply keeps the installer's old refusals: an artifact that is not
  qualified reads `experts-via-tier artifact SHA256 mismatch` (exit 1), a bank over the ceiling is the typed `REFUSED`
  (exit 2), resident experts the installer's resident refusal.

**CPU gates** (`day88-cpu/gates.log`, under `nice 19` in a 600% scope with `MemoryMax=12G`): the tier suites (the bank
suite 96 passed, `day10`'s CLI tests rewritten for the default door: the budgets without `--experts-via-tier`, the
whole-bank default, the pool rollback flag, the retired flags as unknown, the rollback seam's two usage errors), the
engine library (599 passed, with four `day88_qualify` cells: identity then fit, each process refusal with its reason and
no read, the entry is the approved artifact, the prefetch's decision table; the day-44 and day-50 censuses moved to the
door's plan and to the prefetch's one condition), clippy (`-D warnings`, all targets) and fmt clean, the flags census
(the new `MEMRA_EXPERTS_VIA_TIER=0` row, the `MEMRA_MOE_PREFETCH=0` row), `git diff --check`, `rc-scan.py --live` 0.

**The local RTX 5090 check** is queued (queue v20, `rtx5090-queue-v20-20260927.sh`, `day88-cpu/gpu-check.sh` and its
reader), behind other lanes' work on the card: `run-gen-p88` (`0155bc69f`) by default against `run-gen-i22` as
qualified, both orders, then the rollback.

## 5a. The sitting, prepared before any cell

`day88-cell.sh` (the three cells), `day88-read.py` (all three and the phase's verdict), `day88-box-build.sh` (run-gen
per label, run-spec for `p88`) and `day88-box.sh` were written after section 5. Dry checks (`day88-cpu/`): the reader on
a synthetic root from DAY85's BOX41 receipts relabelled with the promoted binary's lines (`make-synthetic.py`;
meaningless) prints every line and `phase1_lands` (`dry-check-reader.log`); its red arm, the same root with one naked
run's pool kind and one nopf run's prefetch line removed, reads `integrity=FAIL` and `void` (`dry-check-reader-red.log`);
the cells under stubs run 60, 20 and 2 invocations in the registered order with the registered flags and environment per
arm (`dry-check-cell.log`); the driver under stubs names the builds, the three cells with their validations and the
reader, and a rerun skips the cells (`dry-check-driver.log`).

Beside the dry check, deciding nothing: on those BOX41 receipts the door's process start to its prefill `MATCH` line
read 9.79 s against REF's 4.37 (the SHA-256 lock over 18.2 GB and the host fill before the first token); the cell
`promo` reads the same wall for the promoted binary, beside its registered readings.

Run as `D88_BUILDS="i22=4b378a064 p88=0155bc69f" bash /root/wt-c/research/spill-c-20260919/day88-box.sh` on a Core Ultra
9 285K host with one RTX PRO 6000 Blackwell Workstation Edition, then the same on a fresh 9950X (at least 48 GB
MemAvailable and 17 GB for the door's pinned bank beside it; about 45 minutes: two builds, 82 runs).

## 6. The cells, read as registered (run by the lead, tree `60166b161`; `pro-single-day88/`, `pro-single-day88-9950x/`)

Both hosts: `D88_BUILDS="i22=4b378a064 p88=0155bc69f" bash .../day88-box.sh`; the builds name `tree=4b378a064...` and
`tree=0155bc69f...`, each `rc=0`; 389 receipts per host `OK` against the lead's box manifests (re-checked here), ELFs by
hash, the power brake not active on either.

**285K class** (BOX44, a Core Ultra 9 285K, one RTX PRO 6000 WS, 249 GB, driver 580.173.02, after lane A's sitting on the
same card; 18:49Z to 19:09Z; `promo` 38 to 47 C, SM median 2692 MHz, N=800 busy samples). Verbatim (`reading.log`):
- `DAY88 PROMO CHECKS rig=pro-single runs=60 integrity=ok` (host demand sequences: the door with its prefetch
  `4bdc2610c3534e42`, `nopf` `0e220d04f52d13e9`)
- `DAY88 PROMO ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0033 max_iqr_window=0.0022 failing=[] -> admissible`
- `DAY88 PROMO gen-only decode medians (N=10 each): naked=0.314 q22=0.314 legacy=0.311 alloc=0.314 nopf=0.380 legnopf=0.379`
- `DAY88 PROMO naked_vs_q22 gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0010 noise=0.0013 -> flat`
- `DAY88 PROMO naked_vs_legacy gen-only decode: pooled=+0.0025 o1=+0.0030 o2=+0.0030 noise=0.0013 -> loses`
- `DAY88 PROMO naked_vs_legacy steady window: pooled=+0.0005 o1=+0.0010 o2=+0.0000 noise=0.0010 -> matches`
- `DAY88 PROMO-RES ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0003 max_iqr_window=nan failing=['naked:window_s=nan', 'legacy:window_s=nan'] -> inadmissible`
- `DAY88 PROMO-SPEC spec-naked rc=0 self_consistency=PASS installed=True off_rollback=False` and `spec-legacy ... PASS`
- `DAY88 VERDICT rig=pro-single -> void (naked loses to legacy; promo-res inadmissible)`

**9950X class** (BOX45, a fresh Ryzen 9 9950X, one RTX PRO 6000 WS, 186 GB, driver 595.91.07; 18:36Z to 18:59Z; `promo`
46 to 59 C, SM median 2677 MHz, N=599). Verbatim:
- `DAY88 PROMO CHECKS rig=pro-single runs=60 integrity=ok` (the same two host demand sequences)
- `DAY88 PROMO ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0010 max_iqr_window=0.0010 failing=[] -> admissible`
- `DAY88 PROMO gen-only decode medians (N=10 each): naked=0.245 q22=0.245 legacy=0.246 alloc=0.245 nopf=0.324 legnopf=0.326`
- `DAY88 PROMO naked_vs_q22 gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0000 -> flat`
- `DAY88 PROMO naked_vs_legacy gen-only decode: pooled=-0.0010 o1=-0.0010 o2=-0.0010 noise=0.0003 -> beats`
- `DAY88 PROMO naked_vs_legacy steady window: pooled=-0.0020 o1=-0.0020 o2=-0.0020 noise=0.0000 -> beats`
- `DAY88 PROMO-RES ADMISSIBILITY ... failing=['naked:window_s=nan', 'legacy:window_s=nan'] -> inadmissible`
- `DAY88 PROMO-SPEC spec-naked rc=0 self_consistency=PASS installed=True ...` and `spec-legacy ... PASS`
- `DAY88 VERDICT rig=pro-single -> void (promo-res inadmissible)`

**Read as registered.** On the 285K class phase 1 does not land: `naked` loses to `legacy` gen-only (+2.5 ms over 32
tokens, 0.314 against 0.311 s, both orders), matching it on the window. The clause stands as registered. On the 9950X
class `promo` passes (the door beats its rollback by 1 ms gen-only and 2 ms on the window) and `promo-spec` passes;
the phase's verdict is void there only because `promo-res` is. On both hosts the promotion changes no program (`naked`
against `q22` flat) and `promo-spec` passes both arms.

**Beside it, deciding nothing.**
- The prefetch: every arm with it is 65 to 80 ms faster over 32 tokens gen-only than without it, on both programs and
  both classes (`naked_vs_nopf` -0.0655 and -0.0790, `legacy_vs_legnopf` -0.0675 and -0.0800).
- The pool: `alloc_vs_naked` flat on both.
- Startup in the spill shape (process start to the prefill `MATCH` line, medians): the door 10.39 s against the
  legacy's 8.72 on the 285K, 12.39 against 9.70 on the 9950X (the SHA-256 lock over 18.2 GB and the host fill). In the
  resident shape the promoted binary starts faster than the legacy (6.19 against 8.84 s; 8.54 against 9.81), the
  mapped banks sparing the pinned copy.
- The resident shape's gen-only, read by the rule's arithmetic though the cell was not admissible: flat on the 9950X
  (0.112 and 0.112 s), `+0.0010 ... noise=0.0003` on the 285K (0.153 against 0.152).

## 6a. Addendum, registered before any rerun: the resident shape's admissibility

The harness gap: without a slot cache, `run-gen` measures and prints no steady-state window (the window lives inside
the slot cache's report, `run_gen.rs`), so the resident shape has no window by construction, and DAY64's clause, applied
to both keys, read the two absent IQRs as failing. Section 5 registered `promo-res`'s reading on decode alone. From now
on, for `promo-res` only, admissibility is the gen-only IQR (at most 0.005 s per arm) and the reading is gen-only; the
spill cells keep both keys. `day88-read.py --only promo-res` reads a rerun on its own. Nothing else changes: the rerun
is `promo-res` on both classes, the same binaries and tree, into its own receipts (`day88b-box.sh`; the cell now hashes
only the binaries it runs, so `promo-res` records `run-gen-p88` alone), and `regresses` still blocks. If it
reads `regresses` on a class, the resident shape's load (mapped banks for a resident qualified artifact) is the change
to fix, registered on its own (pinned copies whenever the experts will be resident), before its code.

## 6b. What the 285K class's loss means for the promotion (for the owner)

**Where the 2.5 ms is.** The door and the legacy run the same kernels on the same bytes (one tape, and `naked` equals
`q22`); the difference is the door's host-hit protocol on the CPU, between two launches of a launch-bound decode
(`DAY72.md`, `DAY83.md`). At I22 the door-only work in the generate phase is 223 us per generated token on the 285K
class (BOX41's clocked arm in `DAY85.md`: the lease 164, the residency check 4.5, the retire 54), 7.1 ms over 32
tokens, and 190 us on the 9950X class (BOX43). About a third of it reaches the 285K's wall (the 2.5 ms); the 9950X hides
it and the door beats there. So the loss is real, host-dependent, and a known quantity.

**Proposal.**
1. **Close it with further cuts, then rerun `promo` on both classes; no per-host default.** A per-host default would
   have to key on a CPU model, one measured model per class, which carries one host's evidence to hosts nobody measured.
   The remaining door-only leaves, measured in situ (`DAY85.md` section 3b), are the budget governor's per-ticket
   reserve and release (about 58 us per token on the development host), the retire walk with its event query (36 to 57),
   the owner proxy's registry, identity and pending bookkeeping (47), the adapter's `validated` (42) and the host cache
   (40). Given how much of a cut the 285K's wall has returned, the wall needs on the order of 100 to 150 us per token
   removed: two or three cuts, the governor and the retire side first, each registered and CPU-gated as before, then one
   `promo` sitting on both classes (with the `promo-res` rerun of section 6a).
2. **In the meantime, what does not depend on the door can land.** C10's prefetch wins 65 to 80 ms over 32 tokens on
   the legacy program as on the door, on both classes (this cell), with G1 to G3 and `pf_wins` from `DAY59.md`; the
   registered pool matters only under the door. The owner can take the prefetch default now (for the qualified
   artifact, on the legacy slot cache: REF, the program this cell's `legacy` arm ran), with the door kept behind
   `--experts-via-tier` until the cuts land; or hold all three for one landing.
3. **Or accept the loss.** The 285K class's gen-only loss is +2.5 ms over 32 tokens (0.8 percent) with the window equal,
   and the 9950X class beats; promoting on both classes now is inside the owner's power and outside the rule this lane
   registered, so it is the owner's call and not this lane's.

This lane recommends 1, with 2's prefetch landing now if the owner wants the 65 to 80 ms in the meantime.

**The local RTX 5090 check landed** (queue v20, `day88-cpu/gpu-check.log`, raw logs in `day88-cpu/gpu-check/`): `DAY88
GPU CHECK PASS`. Every run `MATCH` with one tape; the promoted binary by default and I22 as qualified read one host
demand sequence (`4bdc2610c3534e42`, 22077 lines); the default runs print `qualified`, `installed`, the registered
pool and the prefetch on; the rollback prints `off: MEMRA_EXPERTS_VIA_TIER=0`, the prefetch on and no host demand line.

## 6c. DAY88b, `promo-res` under section 6a, read as registered (run by the lead, tree `64326c89c`)

**9950X class** (BOX45, the first sitting's host and card, one RTX PRO 6000 WS at 600 W, driver 595.91.07; 19:37Z to
19:40Z; `pro-single-day88b-9950x/`, 106 files, the lead's manifest re-checked here OK):
- **The build.** The cell ran the first sitting's `run-gen-p88`, `00bad053...` from `0155bc69f`. The build found it
  cached (0.04 s, the same hash, `builds.log`).
- **The card.** 43 to 56 C over 781 samples; SM median 2625 MHz over the 201 samples with GPU utilization above zero
  (a short cell of mostly load).
- **The reading, verbatim** (`reading.log`; re-read here with `day88-read.py --rig pro-single --only promo-res`, and the
  output is identical):
  - `DAY88 PROMO-RES CHECKS rig=pro-single runs=20 integrity=ok`
  - `DAY88 PROMO-RES ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0000 failing=[] -> admissible`
  - `DAY88 PROMO-RES gen-only decode medians (N=10 each): naked=0.112 legacy=0.112`
  - `DAY88 PROMO-RES naked_vs_legacy gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0000 -> flat`
  - `DAY88 PROMO-RES to_prefill_s medians (beside): naked=8.48 legacy=9.75`
  - `DAY88 PROMO-RES VERDICT rig=pro-single -> passes`

Read as registered: `promo-res` passes on the 9950X class. The resident decode is equal at the printed millisecond
(IQR 0 in both arms), and the promoted binary reaches prefill 1.3 s sooner, as in the first sitting (8.5 against
9.8 s). On this class all three cells now pass (`promo` beats, `promo-spec` passes, `promo-res` passes). Section 5
decides on both classes, so phase 1 still does not land while the 285K class's `promo` loses. That half of DAY88b
(BOX44, after integ73's GPU battery) records whether the resident shape needs its own fix there (section 6a's last
clause).

**285K class** (BOX44, the first sitting's host and card, a Core Ultra 9 285K, one RTX PRO 6000 WS at 600 W, driver
580.173.02; tree `1cb4e8a93`, after integ73's GPU run; 20:11Z to 20:14Z; `pro-single-day88b/`, 106 files, the lead's
manifest re-checked here OK):
- **The build.** The cell ran the first sitting's `run-gen-p88`, `5683c63f...` from `0155bc69f`, found cached (0.02 s,
  the same hash).
- **The card.** 43 to 51 C over 662 samples; SM median 2617 MHz over the 273 samples with GPU utilization above zero.
- **The reading, verbatim** (re-read here with the same command; the output is identical):
  - `DAY88 PROMO-RES CHECKS rig=pro-single runs=20 integrity=ok`
  - `DAY88 PROMO-RES ADMISSIBILITY rig=pro-single ceiling=0.005 max_iqr_gen=0.0010 failing=[] -> admissible`
  - `DAY88 PROMO-RES gen-only decode medians (N=10 each): naked=0.153 legacy=0.153`
  - `DAY88 PROMO-RES naked_vs_legacy gen-only decode: pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0010 -> flat`
  - `DAY88 PROMO-RES to_prefill_s medians (beside): naked=6.23 legacy=8.96`
  - `DAY88 PROMO-RES VERDICT rig=pro-single -> passes`

Read as registered: `promo-res` passes on the 285K class too.
- The first sitting's inadmissible arithmetic had put the resident decode at +1 ms (0.153 against 0.152 s, noise 0.3
  ms). The admissible rerun reads it `flat` (0.153 and 0.153 s, noise 1 ms), and the rerun is the reading of record.
- The promoted binary reaches prefill 2.7 s sooner (6.23 against 8.96 s, as in the first sitting's 6.19 against 8.84).
- Section 6a's last clause does not fire on either class: the resident shape needs no fix.
- Phase 1 now waits on one cell only: `promo` on the 285K class, where `naked` loses to `legacy` by 2.5 ms gen-only.
  That is the owner's question in section 6b.
