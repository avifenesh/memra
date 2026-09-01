# Two-programs inventory — fence-or-prove sweep of the numeric-program pairs

Date: 2026-08-13. Method: 64-agent fan-out, mine → dedup → adversarial verify → rank.
59 candidate pairs mined, 59 deduped, **10 survived adversarial verification**.
Raw: `WORKLIST-raw.md` (full ranked worklist), `journal.jsonl` (per-agent return values).

Motivated by CLAUDE.md §Correctness discipline, added after two production defects with the same
root cause: **any code path that can produce tokens for the SAME request under two different
numerical programs is a correctness bug unless the transition is forbidden or proven bit-identical.**

## FIXED IN THIS COMMIT — W1, the sub-floor prompt tail

`prefill_tick`'s sub-floor tail merge was a **provable no-op**:

```rust
let mut take = if eager_mono { q } else { q.min(budget) };
if q - take > 0 && q - take < PRIME_MIN_T {
    take = if q <= budget { q } else { take };   // <-- always take = take
}
```

Reaching the guard implies `take < q`, which in the non-eager branch implies `budget < q`, so
`q <= budget` is always false and `take` was reassigned to itself. The eager branch never entered
the block at all (`take == q` ⟹ `q - take == 0`). Verified by reading the code, not inferred.

**Both sibling sites already had the working form** — step_session's prefill phase (`take = q;`) and
`interactive_prefill_budget` (`widened = queued;`). This site was the only broken one of three, which
is what a typo-class regression looks like.

### Why it is a correctness bug and not a rounding detail

A remainder of 1..15 prompt tokens falls through `prefill_tick`'s `else` arm, which feeds **prompt**
tokens through `decode_step` **one at a time** instead of `prime_cache`. Those are two different
numeric programs for identical bytes, and the repo already documents the consequence — `run_gen`'s
prime gate, gap #46:

> the batched prime flips a near-tie first token (Qwen3.6-35B pp512 probe: 365 -> 198 "\n" then
> EOS at 2 tokens)

A 2-token completion sold as an answer. Reachability was the worst on the whole worklist: **no flag,
no cache needed, arch-independent, and Q27 dense is the money model.** Any prompt with
`len > budget` and `len mod budget ∈ 1..15`.

It is also **load-shaped**, which is the signature of this defect family: the same prompt arriving
*solo* takes `interactive_prefill_budget`, which widens and *does* merge the tail, so it primes whole.
Identical bytes, different program, decided by whether a peer happened to be unfinished.

### The fix

`take = q`, matching the siblings. Overshoot is bounded to `PRIME_MIN_T - 1 = 15` tokens past the tick
budget, because the guard only fires when `q < take + PRIME_MIN_T` — it cannot move an SLO.

Extracted the whole computation into a pure `prefill_tick_take(q, budget, eager_mono, bound_rem)`,
**because the root cause of the no-op surviving was that nothing tested it.** Three tests added, and
each was verified to FAIL against the old form before being kept:

- `prefill_tick_take_never_leaves_a_sub_floor_tail` — the invariant, swept over budgets
  {16,17,64,256,1024,4096} × all q × both eager arms. Old form fails at q=17/budget=16.
- `prefill_tick_take_overshoot_is_bounded_by_the_prime_floor` — plus the concrete regression
  (1030 tokens @ 1024 budget must take all 1030, not 1024 leaving 6), and that a tail already at the
  floor is NOT merged.
- `prefill_tick_take_stops_on_boundary_without_sub_floor_chunks` — capture boundaries still land
  exactly, and no boundary distance yields a sub-floor chunk.

`cargo test -p memra-server`: 255 passed, 0 failed. Workspace: green.

## STILL OPEN — recorded, not fixed here

**W1 part 2, the boundary door.** `bound_rem.is_none_or(|r| r >= PRIME_MIN_T)` vetoes the *entire*
prime branch when an LCP-split or affinity checkpoint sits <16 tokens ahead, sending the remaining
prompt tokens tokenwise. Correct remedy is to **drop the sub-floor capture** — lose a cache seed, keep
one program — but clearing `snapshot_at`/`ckpt_at` interacts with the post-prime capture logic, so it
needs its own lane. A stale in-code note calling this residual "unreachable at the current
PREFILL_TICK_T" was corrected: that claim only ever covered the tick-budget tail, never this door.

**The single highest-value item on the worklist is a gate, not a fence.** W1–W5 and W9 are all the same
shape — *batch/tick composition, chosen by peer arrival, selects the numeric program*. One serving-shape
gate (peer-arrival delay grid × tick-segmentation grid, asserting **one text hash per (prompt, seed)
cell**) would have caught **eosclass** (delay grid) *and* **splitiso** (cold vs restored cell) *and*
every pair in W1–W5. That harness already exists and is unwired:
`research/eosclass-20260813/repro_width_flip.py` + `run-q27-cache-repro.sh`. Leaving it in `research/`
violates the H100 lane's LAW 3 — *anything guarding a live lane belongs INSIDE the battery.* Promote to
`tools/prime-program-gate.sh`, register in `tools/fast-gate/models.tsv`.

**Also: `tools/tick-invariance-gate.sh` encodes the false assumption.** Its probe replica applies the
tail merge `prefill_tick` lacked (`concat_prime_probe.rs:1091,1264`), so it constructed no sub-floor
tail and never called `decode_step` — which is precisely why this defect was invisible to it. The probe
must mirror `prefill_tick` now that `prefill_tick` is correct, and gain sub-floor cells (tails
0/8/15/16/200 across `PREFILL_TICK_T`).

## PER-HARDWARE OPPORTUNITIES (owner directive 2026-08-13)

Pairs where forbidding the *crossing* keeps a measured win instead of dropping it globally. Detection
seam exists: `fa_sm_count() >= 128` (5090 = 82 SM, PRO 6000 = 188 SM), `Engine::sm_count()`.

| | win at stake | forbidding the crossing instead |
|---|---|---|
| H1 | concat prime batch +30.1% @T=320, +12.6% @512, +5.9% @937 | whole-prompt-only batch membership keeps batching; then key `pb_max`/`pb_maxt` per rig — both are 5090-measured numbers shipped as globals |
| H2 | `MEMRA_SPEC_BURST=128`: +7% c=1, +6.9% c=8 @82 SM; +7.5% @188 SM, greedy byte-identical BOTH rigs | currently fenced to 32 by a **5090-only** contended-first-text tie-break. One PRO cell decides it. **Zero exactness cost for greedy — free win.** |
| H3 | `MEMRA_SERVE_B1FAST`: +8.33% q9 / +5.19% q27 at c=1, PP-N B=1 up to +15% | solo-lease admission guard (refuse admitting a peer while a lease is held) returns the c=1 win at zero corruption risk, and re-opens `MEMRA_SERVE_GS` the same way |

H3 is the exact case the owner doctrine describes: a remote-simplicity default ate a local win that a
crossing-forbid would preserve. H2 is the cheapest — one measurement cell on the PRO pair.
