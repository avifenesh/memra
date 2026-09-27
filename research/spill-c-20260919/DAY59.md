# WP-C day 59 (2026-09-25): `MEMRA_MOE_PREFETCH=1`'s deciding cell, pre-registered (lead owed item 1)

Lead, integ60 resume: `MEMRA_MOE_PREFETCH=1` beat both the door and the legacy on the target card (`DAY51.md`
section 4: REF gen-only decode 0.255 s against the door's 0.277 and the legacy's 0.311; steady window 0.226, 0.240,
0.251). Its `docs/FLAGS.md` row reads "experimental; target-rig gate required" and names no decide-by date, which the
door-hygiene rule forbids. This registers its deciding cell before any code or cell; the row gains its date in the same
commit. Tree at start: `2dc6cfe50` (integ60: main `5d653e851` with this lane's tip).

## 0. What the variable does, from source

- `hybrid_forward.rs` `moe_prefetch_enabled()`: `MEMRA_MOE_PREFETCH=1` (or the opt-in spill worker) turns on the
  in-token prefetch of the legacy MoE slot cache: in the per-token expert loop, before the current expert's kernels,
  the next routed expert's three blocks are staged on the copy stream (`MoeSlotCache::prefetch_source`, a copy
  ordered after the slot's earlier readers), and `dispatch_source` consumes a pending block after a compute-stream
  wait on its copy event. Same bytes, same kernels; only when the H2D happens moves.
- `cpu_experts.rs` `prefetch_depth_from_env()`: the same variable parsed as a depth `1..=8` for the CPU-experts
  companion's prediction worker (`start_moe_prefetch_predictor`, started by `run-gen`/`run-spec` only after a residency
  freeze warm-up, sigmoid-router models, `MEMRA_CPU_EXPERT_LIB`). Not reached on the cells below (no freeze warm-up,
  a softmax-router model, no companion). This cell decides the in-token prefetch only; the companion's reading keeps
  its own rows (`MEMRA_MOE_PREFETCH_TOP`, `MEMRA_MOE_PREFETCH_MIN_LAYER`) and is not changed by any decision here.

## 1. Pre-registration

**The binary.** One `run-gen`, `run-spec` and `memra-server` from the tree at the cells' commit (named in the sitting
before it runs), used by both arms: OFF (the variable unset, the naked default) and PF (`MEMRA_MOE_PREFETCH=1`). The
artifact is the MoE door's approved one (`Qwen3.6-35B-A3B-UD-IQ4_XS.gguf`, `df27a780...7adf`).

**Correctness gates, per card (pass/fail; any failure voids that card's timing verdict).**
- G1, the tape: `run-gen` OFF and PF at two shapes, the pressure shape (`MEMRA_MOE_RESIDENT=0 MEMRA_NGEN=32
  MEMRA_MOE_SLOTS=9986`, prompt `55 88 13`, day 18's) and a heavy one (`MEMRA_MOE_SLOTS=512`, the same otherwise):
  every run `MATCH`, one `tokens:` tape per shape across both arms.
- G2, speculative self-consistency: `run-spec` PF at the pressure shape, K=1..8: `=== SELF-CONSISTENCY PASS ===` and
  every K `self-consistency: PASS (identical to plain target)`.
- G3, the serving shape: `memra-server` booted OFF then PF on the pressure MoE environment, the same greedy requests
  in each boot (three sequential, then four concurrent, 48 tokens each, `/v1/completions`): every response of the PF
  boot byte-equal to the same request's in the OFF boot, no request error (`day59-serve-cell.sh`).

**The timing cells, per card.** Two shapes, each its own collector hold, 20 runs, order 1 (OFF, PF) x 5 and order 2
(PF, OFF) x 5, N=5 per arm per order, 250 ms telemetry, the runner under the 1200% CPU cap (12 pinned cores on a box
without systemd):
- `pftime`, the pressure shape (where misses exist);
- `pfnaked`, the naked shape (`MEMRA_NGEN=32`, prompt `55 88 13`, no MoE variable: the default residency on that card).

**Readings and the rule** (`day59-pf.py`). Gen-only decode seconds (the day-18 observation), the steady window beside
it. `noise` the larger IQR of the two arms. Per shape: `pf_wins` iff `median(PF) < median(OFF) - noise` in order 1, in
order 2 and pooled; `pf_loses` iff `median(PF) > median(OFF) + noise` in both orders; `pf_flat` otherwise.

**What follows (the flags doctrine, per card).** PF becomes that card's naked default iff G1, G2 and G3 pass there,
`pftime` reads `pf_wins` and `pfnaked` does not read `pf_loses`; the promotion itself is the owner's call, and a
default flip is its own change (the variable keeps `=0` as the rollback seam). `pf_loses` or `pf_flat` at the pressure
shape on a card: the in-token prefetch is not that card's default, and if it holds on both cards the door is deleted
by the lane that measured it (the flags doctrine: losing or flat arms are deleted), unless the MoE slot cache door
(which carries its own prefetch, `DAY50.md`) is deleted first, in which case the decision stands for the legacy alone.

**Decide-by** 2026-10-04 (the MoE slot cache door's date, the two are read together), in the row.

**What each card can decide.** Each card its own default; the RTX 5090 (which needs its reset first) and the target
card are read separately.

## 1a. The binaries, named before any cell

One build label, `c60`, serves this day and `DAY60.md`: `run-gen-c60`, `run-spec-c60` and `memra-server-c60`, all
from one tree, the lane's tip at the sitting (engine of `fec3c582f`, which adds `DAY60.md`'s log-only
`--moe-dispatch-clock`; no arm here sets it, and without it the slot cache keeps no clock). `day59-cell.sh` names
those three binaries (it named `-final` before this section; the day-51 `-final` binaries in the local binary
directory are a different tree and stay untouched for `DAY51.md`'s queued cells). The exact commit is recorded in
each cell's `tree.sha` and `binary.sha256` and in the sitting's build log.

Pinned (`DAY61.md` section 2b, before any cell): the lane's tip now carries DAY61's I11 and I12, so the label `c60`
is built from `da649107c`, the commit whose engine is `fec3c582f`'s as this section names.

## 2. The target card (BOX12, one RTX PRO 6000 Blackwell Workstation Edition at 600 W; `pro-single-day61/`)

Sitting `day61-box.sh` on the lane tip `f0e77b8be`, 2026-09-25 04:35Z to 05:00Z; binaries built on the box from
`da649107c` (`run-gen-c60` `57995815...`, `run-spec-c60` `cf0cc23d...`, `memra-server-c60` `7ddb8187...`); the runner
pinned to 12 cores with `taskset` (no systemd on the box); one collector hold per cell. Receipts mirrored file for
file (`MIRROR-CHECK.txt`: 635 of 635 `OK` against the box's `MANIFEST.box.sha256`). Regime per cell in
`pro-single-day61/regime.txt` (the four cells 31 to 51 C, SM median 2617 to 2827 MHz). Verbatim (`*/reading.log`):

- `DAY59 G1 rig=pro-single slots=9986 exits={'off': 0, 'pf': 0} match=True one_tape=True -> PASS`
- `DAY59 G1 rig=pro-single slots=512 exits={'off': 0, 'pf': 0} match=True one_tape=True -> PASS`
- `DAY59 G2 rig=pro-single exit=0 k_lines=8 k_pass=8 -> PASS`
- `DAY59 G3 rig=pro-single requests=7 equal=7 errors=[] -> PASS` (48 tokens each, both boots)
- `DAY59 TIME rig=pro-single shape=pftime gen-only decode: off=0.310 pf=0.255 (N=10 each) pf_minus_off pooled=-0.0550 o1=-0.0550 o2=-0.0550 noise=0.0000 -> pf_wins`
  (steady window `off=0.251 pf=0.226 ... noise=0.0003 -> pf_wins`); `DAY59 VERDICT rig=pro-single shape=pftime integrity=ok -> pf_wins`
- `DAY59 TIME rig=pro-single shape=pfnaked gen-only decode: off=0.113 pf=0.113 (N=10 each) pf_minus_off pooled=+0.0000 o1=+0.0000 o2=+0.0000 noise=0.0003 -> pf_flat`;
  `DAY59 VERDICT rig=pro-single shape=pfnaked integrity=ok -> pf_flat` (the naked shape keeps every expert resident
  on this card, so there is no miss to prefetch)

A note on `noise=0.0000`: `run-gen` prints the gen-only seconds to three decimals and each arm's ten runs read the same
value, so the IQR is 0 at the printed resolution; the rule reads it as registered.

**What follows, as section 1 registered.** On the target card G1, G2 and G3 pass, `pftime` reads `pf_wins` and
`pfnaked` does not read `pf_loses`: `MEMRA_MOE_PREFETCH=1` qualifies as this card's naked default. The promotion is
the owner's call, and the flip is its own change (`=0` kept as the rollback seam). The RTX 5090's cells wait on its
reset (queue v6).

## 3. The RTX 5090 (queue v9, 2026-09-26 01:02Z to 01:09Z; `rtx5090-day59/`)

The RTX 5090 queue v9 (`rtx5090-queue-v9-20260926.sh`) ran these after the rig's reboot wiped the queued binaries in `/tmp`: every binary was rebuilt from its named commit by `c-local-build.sh` in a build worktree under the lane's `target/` (CUDA 13.1, sm_120a; build logs in each cell's `builds/`), behind `/tmp/memra-5090.lock` with the card idle (no compute app) before each hold. A rebuilt binary's hash differs from the one named before the first attempt (the build path is part of the binary); its source tree is the named one. Here `run-gen-c60` `809132ce...`, `run-spec-c60` `b4779a91...`, `memra-server-c60` `14d6760f...`, tree
`da649107c`. Regime over the `pftime` hold: 61 to 71 C, SM median 1627 MHz, N=608. Verbatim:

- `DAY59 GATES rig=rtx5090 -> PASS` (`G1 ... slots=9986 ... -> PASS`, `G1 ... slots=512 ... -> PASS`, `G2 ... k_pass=8 -> PASS`)
- `DAY59 G3 rig=rtx5090 requests=7 equal=7 errors=[] -> PASS`
- `DAY59 TIME rig=rtx5090 shape=pftime gen-only decode: off=0.412 pf=0.364 (N=10 each) pf_minus_off pooled=-0.0480 o1=-0.0490 o2=-0.0480 noise=0.0020 -> pf_wins`
- `DAY59 TIME rig=rtx5090 shape=pftime steady window: off=0.408 pf=0.397 (N=10 each) pf_minus_off pooled=-0.0115 o1=-0.0120 o2=-0.0120 noise=0.0033 -> pf_wins`
- `DAY59 VERDICT rig=rtx5090 shape=pftime integrity=ok -> pf_wins`
- `DAY59 TIME rig=rtx5090 shape=pfnaked gen-only decode: off=0.216 pf=0.214 (N=10 each) pf_minus_off pooled=-0.0015 o1=+0.0020 o2=-0.0050 noise=0.0075 -> pf_flat`
- `DAY59 VERDICT rig=rtx5090 shape=pfnaked integrity=ok -> pf_flat`

**Read as registered: G1, G2 and G3 PASS, `pf_wins` under pressure, `pf_flat` naked, the same as the target card.**
`pfnaked` does not read `pf_loses`, so `MEMRA_MOE_PREFETCH=1` qualifies as this card's naked default too. Both cards
now qualify it; the promotion is the owner's call and the flip its own change.
