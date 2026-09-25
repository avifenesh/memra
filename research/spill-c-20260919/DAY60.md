# WP-C day 60 (2026-09-25): the door's gap to REF, attributed stage by stage (lead owed item 2), registered before code

Lead, integ60 resume: "Attribute the door's gap to REF stage by stage and tune the door to match or beat it". On the
target card (`DAY51.md` section 4) the tuned door read 0.277 s gen-only and 0.240 s steady window; REF (the legacy slot
cache with `MEMRA_MOE_PREFETCH=1`) read 0.255 and 0.226: 0.69 ms per gen token and 0.44 ms per window token. Written
before any code; tree at start: `97bd6d889`.

## 0. The two programs, from source (`moe_cache.rs`, `hybrid_forward.rs`)

Both prefetch the same block at the same point: in the per-token expert loop, before expert `j`'s kernels, the three
blocks of `sel[j+1]` (`moe_prefetch_expert` -> `prefetch_source`), each staged on the copy stream after an event
ordering the slot's earlier readers, consumed by `dispatch_source` after a compute-stream wait. Neither prefetches
`sel[0]`, so the first expert of every layer is a demand miss in both. REF is not a prediction table: the CPU-experts
predictor (`MEMRA_MOE_PREFETCH=<depth>`, `cpu_experts.rs`) is not started on this model or shape (`DAY59.md` section 0).
What differs per block:

- **Legacy (REF).** `dispatch_source`: `reap_copy_sources`, a frequency update (a SipHash map), the table lookup; a
  pending block's compute wait; a miss's synchronous `stage_expert` from the loaded `HostExps` (write-combined
  pinned). `prefetch_source`: three map lookups, `reserve_prefetch_slot`, `stage_on_copy_stream`, a pending insert.
- **Door (ON).** `admit_banked`: the pending guard, `retire_banked` (the in-flight queue's front event, `finish`
  through the owner proxy), the dense validate memo, the table lookup; a pending block's compute wait and its lease
  moved to the in-flight queue; a miss's `demand` and `with_bytes` through the proxy and `stage_banked`.
  `prefetch_banked`: the guards, `retire_banked`, the memo, `host_resident` through the proxy, the slot, `demand`
  through the proxy (the tier's lease, the governor, the trace line), `with_bytes` and `stage_on_copy_stream` from the
  cached pinned pool, the pending insert with its lease. Every prefetched block is also `finish`ed later through the
  proxy. So per prefetched block the door makes three or four owner-proxy calls and a tier demand that REF does not.

The hypothesis this points to, stated before any measurement: the gap is the door's per-block owner-proxy and tier
work on its prefetch path (about 88.6 prefetched blocks per window token, `DAY50.md`), not the GPU side.

## 1. Pre-registration

**The instrument, `--moe-dispatch-clock`** (a `run-gen` flag, log only, both programs; its decide-by is 2026-10-04 in
`MOE-SLOT-CACHE-DOOR.md`, the door's). When set, the slot cache keeps a clock and `run-gen` prints, at the phase
points of the door's stage lines (gate, generate, warm, window), `[moe-cache] dispatch-clock phase=<p> ...` with, both
programs:
- `dispatch_calls`, `dispatch_ns` (the whole `dispatch_source`), and its outcomes `dispatch_hits`, `dispatch_pending`
  (a prefetched block consumed), `dispatch_sync` (a demand miss);
- `prefetch_calls`, `prefetch_ns` (the whole `prefetch_source`), `prefetch_issued`;
- inside the prefetch, `pf_reserve_ns` (the slot) and `pf_stage_ns` (the copy-stream enqueue), both programs; and on
  the door only `pf_retire_ns` (`retire_banked`), `pf_resident_ns` (`host_resident`), `pf_demand_ns` (`demand`), the
  staging inside `with_bytes` counted in `pf_stage_ns`.
Nothing else changes; the clock reads `Instant` and adds counters. A CPU census pins that the clock's only writes are
its own counters.

**The cell `gap`** (both cards; `day60-cell.sh`, reader `day60-gap.py`). Day 18's pressure shape, the final tree's
binary for every arm. Arms: REF (`MEMRA_MOE_PREFETCH=1`), REFC (REF with `--moe-dispatch-clock`), ON
(`--experts-via-tier --expert-bank-host-bytes=17179869184`), ONC (ON with `--moe-dispatch-clock`). Order 1 (REF, REFC,
ON, ONC) x 5, order 2 reversed x 5, 40 runs, one collector hold, 250 ms telemetry, the 1200% cap.

**Readings** (medians over each arm's 10 runs, per token over the 32 generated and the 32 window tokens):
- R1, the gaps: `gen_gap` and `window_gap` = ON minus REF (ms per token), per order and pooled.
- R2, the instrument's cost: REFC minus REF and ONC minus ON on the window; `within_bound` if each is at most the
  smaller of 10% of `window_gap` and 0.1 ms per token.
- R3, the CPU brackets: ONC minus REFC per window token for `dispatch_ns`, `prefetch_ns` and each `pf_*`, with the
  outcome counts of both.
- R4, the split: `cpu_gap` = (ONC's `dispatch_ns` + `prefetch_ns`) minus REFC's, per window token; `residual` =
  `window_gap` minus `cpu_gap` (the GPU side and whatever runs outside the brackets).
- The verdict line names the largest term of R3 (`top=`) and says whether `cpu_gap` covers at least 75% of
  `window_gap` (`cpu_side`) or not (`not_cpu_side`). The same terms for the generate phase beside it.

**What it decides.** Nothing about the door by itself: it names where the gap is, which the improvements that follow
(each its own registration, starting `DAY61.md`) target. The hypothesis above is either the `top` term or refuted.

**What each card can decide.** Each card its own attribution; the target card first (the RTX 5090 needs its reset).
