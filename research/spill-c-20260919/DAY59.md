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
