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
