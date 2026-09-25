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

## 1a. The binary, named before any cell

"The final tree's binary" in section 1 is `run-gen-c60`: the build label `c60` of `DAY59.md` section 1a, the lane's
tip at the sitting, whose engine is `fec3c582f` (the instrument). `day60-cell.sh` runs it for all four arms. The reader
`day60-gap.py` is written before any cell: integrity (40 runs, every exit 0 and `MATCH`, one tape, 32 generated and
32 window tokens per run, the four clock lines in every clocked run and none in the others), then R1 to R4 and the
verdict line as registered. `top=` is the largest ONC minus REFC term among `dispatch_ns`, `prefetch_ns` and the
five `pf_*` terms, read literally (`prefetch_ns` contains the `pf_*` terms). The generate phase's clock delta spans
`gate` to `generate`, which includes the three-token prime; its per-token terms divide by the 32 generated tokens,
as registered, and the prime's share is stated when read.

Existing evidence, read before this cell and moving no bound: the target card's `DAY58.md` tip arm (the tuned door,
`--expert-bank-stages`, N=10, `pro-single-day52/smallfix/`) spends, per window token, `inner_demand_ns` 0.423 ms on
92.3 host-hit demands (88.6 of them prefetches), of which the bank's `stage()` is 0.333 ms (3.6 us per demand), with
`wait_ns` 0 (the in-flight bound never waited). That is the same size as the 0.44 ms window gap, which is the
hypothesis of section 0; this cell measures both programs with one clock.

Pinned (`DAY61.md` section 2b, before any cell): `run-gen-c60` is built from `da649107c`, whose engine is
`fec3c582f`'s; the lane's tip now also carries DAY61's I11 and I12, which this cell does not measure.

## 2. The target card (BOX12; `pro-single-day61/gap/`)

In the days 59 to 61 sitting (`DAY59.md` section 2 for the box, the binary and the mirror check), one hold 04:46Z to
04:53Z, 41 to 46 C, SM median 2617 MHz, N=10 per arm. Verbatim (`gap/reading.log`):

- `DAY60 GAP CHECKS rig=pro-single runs=40 integrity=ok`
- `DAY60 R1 gen_gap_ms_per_token on_minus_ref pooled=+0.688 o1=+0.688 o2=+0.688 | medians ref=0.255 refc=0.255 on=0.277 onc=0.279 (N=10 each)`
- `DAY60 R1 window_gap_ms_per_token on_minus_ref pooled=+0.437 o1=+0.437 o2=+0.437 | medians ref=0.226 refc=0.226 on=0.240 onc=0.241 (N=10 each)`
- `DAY60 R2 instrument_ms_per_token refc_minus_ref=+0.000 onc_minus_on=+0.047 bound=0.044 -> over_bound`
- `DAY60 R3 window per token: dispatch_ns=+0.092 prefetch_ns=+0.607 pf_reserve_ns=+0.001 pf_stage_ns=+0.034 pf_retire_ns=+0.009 pf_resident_ns=+0.054 pf_demand_ns=+0.464 | refc dispatch_calls=471.0 dispatch_hits=378.7 dispatch_pending=88.6 dispatch_sync=3.8 prefetch_calls=412.1 prefetch_issued=88.6 | onc dispatch_calls=471.0 dispatch_hits=378.7 dispatch_pending=88.6 dispatch_sync=3.8 prefetch_calls=412.1 prefetch_issued=88.6`
- `DAY60 R4 window wall_gap=+0.437 cpu_gap=+0.699 residual=-0.262 top=prefetch_ns (+0.607) -> cpu_side`
- `DAY60 R4 gen wall_gap=+0.688 cpu_gap=+1.478 residual=-0.790 top=prefetch_ns (+1.265) -> cpu_side`
- `DAY60 GAP rig=pro-single integrity=ok window: wall_gap=+0.437 cpu_gap=+0.699 top=prefetch_ns cpu_side; gen: wall_gap=+0.688 cpu_gap=+1.478 top=prefetch_ns cpu_side`

**Read as registered.** R2 reads `over_bound`: the door's clocked arm is 0.047 ms per window token slower than its
unclocked arm against a bound of 0.044. That is one half of a printed millisecond over 32 tokens (the ONC median
0.2415 s against ON's 0.240 at three printed decimals); REF's arm reads +0.000. The instrument's own cost on the door
sits at the resolution floor, so the R3 terms carry about 0.05 ms per token of the clock itself on the door side;
recorded, not corrected. R3 and R4: the two programs make the same dispatches (471.0 per window token, the same hit,
pending and sync counts) and the same prefetches (412.1 calls, 88.6 issued); the door's CPU brackets exceed REF's by
0.699 ms per window token against a 0.437 wall gap (`cpu_side`), and the largest part is the prefetch path (`top=
prefetch_ns`, +0.607), within it the owner demand (`pf_demand_ns` +0.464, the host-hit lease, 5.2 us per issued
prefetch). The CPU side exceeds the wall gap because part of the CPU time overlaps GPU work already queued. The
hypothesis of section 0 is the reading. The improvement it points to is `DAY61.md`'s I11, measured in the same sitting
(`DAY61.md` section 3).
