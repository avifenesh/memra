# No tail fold widens a prime call past the 4,096-row cap (memra#1258, part 1)

Verdict: fixed on the single-device ring-off schedule for every prompt of k x 4,096 + 1 to 15
tokens (k >= 1; 945 lengths up to 262,144 tokens). Each of them primed its last call over
`PRIME_CHUNK_MAX_TOKENS`, at up to 4,111 rows. In the measured sweep (4,060 to 8,168 tokens on an
otherwise idle card), 4,106 and 4,108 tokens were the only lengths whose cold DFlash prime drew more
than admission charged. The prime's pool growth there now drops from 1,980 MB to that of its
neighbours. Greedy outputs are byte-equal to main.
The rest of memra#1258 (the first chunk's pool draw varies from run to run, and a later chunk
cannot reuse the earlier chunk's freed blocks) is separate work and is not fixed here.

Card: one RTX 5090 Laptop GPU (24 GB). The enforced power limit at the start of each run is in its
`guard.log` (150 to 157 W) and was not changed. Model: Qwen3.8-27B NVFP4 + Q5_K with the DFlash2
drafter, ctx 32,768, prefix cache and reuse pool off, cold distinct prompts one at a time.

## What happened

`fixed_prime_chunk_ranges` cuts a prompt of `t` rows into calls of `chunk` rows. With the ring
off, a remainder shorter than `PRIME_MIN_T` (16) folded into the previous call. On the default
schedule (`chunk` = `PRIME_CHUNK_MAX_TOKENS` = 4,096), a prompt of k x 4,096 + r tokens
(k >= 1, r = 1 to 15) therefore primed its last call at 4,096 + r rows; at k = 1 that is the whole
prompt in one call. The ring schedule already moved the boundary back instead.
`docs/decisions/PRIME-TAIL-FOLD.md` rejected the same widening for its own fold: "a call wider
than 4,096 rows would re-size the resident prime slab and its admission charge".

Measured before the change (`before/`, main `11b368f7d7` with the `[diag1255]` logging, two boots
with the lengths in opposite orders, the pool drained before each prime, `MEMRA_ALLOC_TRACE=1`):

| prompt tokens | prime pool growth (MB) | prime + KV + plane against cost + floor (MB) |
|---|---|---|
| 4,060 to 4,088 | 1,611 to 1,644 | -213 to -232 |
| 4,106 | 1,980 | +114 |
| 4,108 | 1,980 | +105 |
| 4,153 to 8,168 | 1,644 to 1,879 | -47 to -273 |

`prime growth` is the pool's reserved growth from the drafter plane's allocation to the end of the
drafter ingest. The last column adds the KV and plane bytes and the pool's unused bytes at the
prime's start, against the gate's `cost + reserve` (`harness/probe-table.py`).

## Change

`fixed_prime_chunk_ranges` (`crates/memra-engine/src/hybrid_forward.rs`) folds a short tail only
while the merged call stays within `PRIME_CHUNK_MAX_TOKENS`. Otherwise the last boundary moves back
to `t - PRIME_MIN_T`, as the ring schedule's does, and the GDN grid alignment then places it
(4,106 tokens: calls of 4,064 and 42 rows). An explicit `MEMRA_PRIME_CHUNK` wider than the cap
(monolithic) keeps the legacy fold. The explicit-chunk schedule is not aligned to the GDN grid
afterwards, so on a GDN trunk the split itself is placed on the grid
(`fixed_prime_chunk_ranges_on_grid`). An explicit chunk equal to the cap then primes in the
default schedule's calls.

## Evidence

- **Footprint** (`after/`, the same harness on the change): 4,106 tokens grew the pool by 1,644 MB
  (-221 MB against the charge) and 4,108 by 1,611 MB (-264). The other lengths read as before.
  The `[dflash-taps]` lines show the widest call at 4,064 rows for both (`after/*-taps.txt`).
- **Exactness** (`gates/`, `harness/b-exact.sh`). Token ids come from `run-gen` on a prose text,
  and texts of exactly 4,100, 4,106 and 4,111 tokens are made from them with the GGUF vocabulary
  (`harness/gguf-detok.py`). Each binary pair is built from the same tree, with only
  `hybrid_forward.rs` from main or from the change:
  - `run-gen` at 4,106 tokens: prefill against tokenwise decode `MATCH`, batched prime against
    tokenwise `MATCH`, the same argmax and logit maxdiff (2.185e-1, 2.329e-1) on both arms, and the
    same 32 generated tokens.
  - `run-spec` K=1..8 at 4,106 tokens with `MEMRA_SPEC_TEMP=0`: `STEP-IDENTITY ... MATCH`, eight
    `self-consistency: PASS` lines identical between the arms, `=== SELF-CONSISTENCY PASS ===`.
  - `dspark_q38_gate` at 4,100 / 4,106 / 4,111 tokens: `ALL EXACT` on both arms, with the same
    `spec_sha256` and `plain_sha256` per prompt on main and on the change.
  - Bit for bit (`gates-review/`, `harness/b-exact3.sh`, balanced profile). Two cases:
    - Qwen3.8-27B at 8,200 tokens: calls of 4,096 + 4,104 rows on main, 4,096 + 4,064 + 40 on the
      change.
    - The MoE trunk Qwen3.6-35B-A3B (IQ4_XS) at 4,106 tokens: one 4,106-row call on main,
      4,064 + 42 on the change.

    On each, the serving prime's last-row logits (`MEMRA_PP_PRIME`, `MEMRA_PP_LOGITS`, 248,320 f32)
    are byte-equal between main and the change (logit maxdiff 0; `logits.sha256`). The 16 greedy
    tokens are equal, and `run-gen` reads the same argmax and maxdiff lines on both arms:
    - 27B: `MATCH` / `MATCH`, 2.090e-1 / 1.973e-1;
    - 35B-A3B: `MATCH` / `MATCH`, 9.374e-1 / 9.894e-1.
- **CPU** (`cpu/`, `cpu-review/`):
  - `prime_chunk_schedule_tests`. `no_default_prime_call_exceeds_the_chunk_cap` sweeps t = 1 to
    4 x 4,096 + 64 with the ring on and off, with no GDN grid and with grids of 32, 64 and 128
    rows. Every call stays within the cap and covers 0..t with calls of at least 16 rows, and every
    schedule equals the old one wherever the old one fit (240 do not: the ring-off tails).
  - `an_explicit_cap_chunk_keeps_the_default_schedule`: an explicit 4,096 chunk equals the default
    schedule on each grid.
  - With the fold rule reverted (`cpu/red-fold.patch`), `fixed_schedule_retains_measured_geometry`
    and `no_default_prime_call_exceeds_the_chunk_cap` fail.
  - The GLM host test `the_mhc_prime_carries_a_bounded_number_of_rows_at_every_context` now bounds
    a call by the cap itself. It fails on main's schedule (`cpu-review/red-glm5cap-base.log`).
  - The DFlash tap bookkeeping test reads production's schedule.
  - `cargo fmt --check`, `cargo clippy -p memra-engine --all-targets -D warnings` and the engine
    library suite are clean.

## Cost

The change adds a call of 42 rows (4,106 tokens) to the affected prompts. In the
`dspark_q38_gate` runs, `prime_s` read 0.28-0.41 s higher on the change, but those were single runs
in sequence, main first on a cooler card. So TTFT was compared on its own (`ttft/`,
`harness/b-ttft.py`). The comparison ran 05:12-05:24 on 2026-11-13 on the balanced platform profile,
with no throttle log, before the rig's thermal rule for timed cells existed. It is an indication,
not a measured cost. Cold raw-text prompts of 4,090 (one call on every arm), 4,100, 4,106 and
4,111 tokens were streamed, 3 reps per boot, boots interleaved main, change, variant 2, variant 2,
change, main (`memra-server` built from the same tree with only the engine change).

The card heats through each boot and the run: the 4,090-token control moved from 2.96 s (main's
boots) to 3.01 s (the change's) and 3.10 s (the variant's, which ran in the middle). So each prompt
is read against the control in the same rep:

| prompt | main, s over the 4,090 control | the change | variant 2 |
|---|---|---|---|
| 4,100 | +0.082 (sd 0.037) | +0.134 (sd 0.041) | +0.159 (sd 0.032) |
| 4,106 | +0.120 (sd 0.070) | +0.189 (sd 0.076) | +0.197 (sd 0.068) |
| 4,111 | +0.194 (sd 0.079) | +0.230 (sd 0.101) | +0.277 (sd 0.047) |

The indication is at most about 0.05 to 0.07 s over main (about 2% of TTFT): one card, 6 runs per
arm, within one standard deviation. Variant 2 runs main's single call and still reads 0.08 s over
main; that follows the boot order, since its boots ran in the middle of the run while the card was
hottest. The +0.3 s of the gate runs was the card warming.

Variant 2 (`variant2/variant2.patch`, not adopted) keeps the fold and sizes the prime slab and the
DFlash tap sink for `PRIME_CHUNK_MAX_TOKENS + PRIME_MIN_T - 1` rows from their first allocation.
The before run's allocation trace shows what the folded 4,106-row call did beyond the cap:
- it re-allocated the whole prime slab at 4,106 rows (`prime_slabs_get`, about 1.4 GB beside the
  old one);
- it allocated a 420 MB tap sink next to the cached 419 MB one;
- every per-layer buffer of the call was 15 rows larger than the 4,096-row blocks the pool had
  cached, so none of them could be reused.

The variant removes the first two but not the third. The split keeps every call within the cap and
costs nothing measurable, so the split was kept.

## Files

- `before/`, `after/`: per-boot server logs of `harness/b-probe.py` (`up.log`, `down.log`), the
  footprint tables (`*-table.md`), the first-call allocation census (`*-chunk1.md`,
  `harness/cmp-chunk1.py`) and the `[dflash-taps]` lines.
- `gates/`: `guard.log` (binary hashes, clock and power at the start), the gate logs per arm
  (`1258base-*` main's schedule, `1258-*` the change), the prompt texts and their ids.
- `gates-review/`: the bit-for-bit runs of the review (8,200 tokens on the 27B, 4,106 on the
  35B-A3B): `guard.log`, the `run-gen` logs per arm (`*-pp.log` the logits dump), the prompt texts
  and ids, and the logits digests.
- `cpu/`: the schedule tests, green and red.
- `cpu-review/`: the checks of the review fixes. Its `summary.txt` covers the schedule and DFlash
  tests, the GLM host tests, fmt, clippy, the engine suite and the GLM bound's red arm on main.
- `ttft/`: the TTFT A/B, one directory per boot (`rows.jsonl`, server log) and its `guard.log`
  (binary hashes, and the card's clock and power after each boot), and the prompt texts.
- `variant2/`: the variant's patch.
- `harness/`: the scripts.
