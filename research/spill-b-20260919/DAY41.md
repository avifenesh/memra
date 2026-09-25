# WP-B day 41: O11, the grid-checkpoint rewind arm for the verbatim-extension resume (priced for the owner)

OWED.md O11. The lead's order (2026-09-25): pre-register and price the grid-checkpoint rewind arm, so the owner decides
the near-tie residual question on receipts; do not change the serving default.

The question. A continuation that exactly extends a parked session's committed tokens resumes today on the parked
state, which holds the previous turn's DECODED rows; a cold prime of the same prompt primes those rows. `docs/SERVING.md`
names this a near-tie residual ("verbatim-extension continuation resumes keep decode-computed rows whose arithmetic a
cold prefill never reproduces"); `CLAUDE.md`'s one-numeric-program rule reads such a crossing as a bug unless forbidden
or proven bit-identical. Measured on the 27B on the target card: 1 flip in 19 exact-extension resumes (DAY38 2.1), 3 in
30 on both arms and both orders (DAY38 2.3). A resume that rewinds to the entry's grid checkpoint (a prime-produced state
on the GDN prime grid) and re-primes from there is cold-identical by the grid law (`grid_align_boundary`: a prime split
on the grid is bit-identical to the monolithic prime). This day builds that arm behind a default-OFF door and measures
what it costs against keeping the decoded rows. Every cell is `executed-not-qualified`; no default moves; the contract
is the owner's.

## 1. Pre-registration

Committed and pushed before any day-41 code and before any day-41 cell. Nothing in section 1 changes after a number is
seen; a failed clause is recorded as it reads and a revision is a new, dated addendum pushed before its code.

### 1.1 The arm (`MEMRA_RESUME_GRID_REWIND`, default unset; decide-by 2026-10-09)

Unset: today's program, byte for byte. `1`:

- **Arming.** Every plain session arms its affinity checkpoint at `plain_checkpoint_boundary(prompt)` (grid-aligned,
  before the last turn marker, or `PLAIN_CKPT_RAW_GUARD` tokens before a markerless prompt's end), not only a
  nominatable prompt or a named conversation; every cold spec session arms its turn checkpoint by the same boundary
  (resumed spec sessions already arm one).
- **Plain pool.** An exact-extension hit (`continuation_reuse_index`) rewinds the entry to its checkpoint before the
  suffix primes, after the park-compact regrow when there is one, through the affinity path's own restore
  (`restore_cache_checkpoint`, `fed` truncated to the checkpoint, the checkpoint's logits). The primed suffix is then the
  parked prompt's tail past the checkpoint, the parked turn's generated tokens, and the new tokens. An entry with no
  checkpoint, or a lapped ring, is dropped and the request primes cold.
- **Spec pool.** An Exact or Text hit rewinds the session to its turn checkpoint (`spec_rewind_to_checkpoint`) and primes
  the token suffix from there (the text remainder is not used after a rewind). No checkpoint: cold.
- **Receipts.** One line per resume: `[kv-reuse] grid-rewind: <plain|spec> extension of <F> committed rows rewound to
  <P> (re-priming <R> rows; model <m>)`, or `[kv-reuse] grid-rewind: <plain|spec> declined (<why>); cold`.
- **Outside this arm, stated:** the DSpark pool's resume and the prefix cache's prompt-end seed extension hits (the
  other residual `docs/SERVING.md` names). Each is its own arm if the owner wants the same question answered there.

The arm's costs, by construction: (a) the re-primed rows per resume (the previous turn's generated tokens plus the
prompt tail past the checkpoint, at most `PLAIN_CKPT_RAW_GUARD` + 31 tokens on a raw prompt); (b) one checkpoint
snapshot per session on a recurrent model (156.9 MB on the 27B, DAY39 2.1), booked by the admission door's term when
armed; (c) an armed checkpoint takes a raw-prompt session out of the in-batch fanout and prime-batch paths
(`plain_ckpt_nominatable`'s documented cost).

### 1.2 Engine-level mechanism check (`concat-prime-probe primepath --hist K --rewind`)

A new `rewind` arm beside `hist`: turn 1 primed with a stop and a snapshot at the grid boundary `b` (the largest
multiple of the GDN chunk at or below the prompt end that leaves at least `PRIME_MIN_T` rows after it), primed on to its
end, the same K greedy tokens decoded, a rollback to the snapshot, then `seq[b..]` primed. Against the monolithic
reference the arm is expected `EXACT`; `hist` (the kept rows) prints its own verdict. The line also prints each arm's
suffix-prime wall time. Runs: the 27B on the target card, the 9B on the 5090, prompts of 6,144 and 30,720 tokens from
`docs/SERVING.md`, K in {32, 256}, suffix 64 tokens.

### 1.3 Serving cells (`day41-client.py`, one boot = one arm)

- **Shape RX (a raw agent loop).** Per conversation, `/v1/completions` with `prompt_ids`, greedy, streamed: turn 1 is
  L ids (on the grid) with `max_ctx = L + 2048` and `max_tokens = G`; turn 2 is turn 1's ids, turn 1's completion
  tokenized through `/v1/tokenize`, and 64 stream ids; turn 3 likewise from turn 2. Each turn's cold twin: the same ids
  and `max_ctx` in a fresh namespace. G in {32, 256} (both in every boot, as separate conversations). Lengths: 6,144 and
  30,720 on both cards, plus 122,880 on the target card. N = 5 conversations per (L, G).
- **Shape FX (the fanout cost).** 8 raw requests sharing a 4,096-token prefix, released together, `max_tokens = 16`:
  the in-batch fanout's reach per arm.
- **Routes.** The plain route (`MEMRA_SERVE_SPEC=0`) and the spec route (the default), each its own boots.
- **Arms and orders.** `keep` (door unset) and `rewind` (`=1`), one binary, both orders per route (O1 keep then rewind,
  O2 rewind then keep): 8 boots per card. Plus `off-prev`: the previous tip (`a803d3080`, no door) on the plain route
  with the RX shape at 6,144 only, for R3.
- Per row: tag, shape, route, arm, L, G, turn, cold, TTFT (first streamed token), E2E, `cached_tokens`, completion
  digest, prompt sha256, the pool-hit delta.

### 1.4 Clauses (the arm's correctness; the decision is not a rule)

- **R1 exactness.** On `rewind`, every resumed RX turn (`cached_tokens > 0`) equals its cold twin, on both routes, every
  length and G, both cards; every such resume prints a `grid-rewind:` line whose R equals the committed rows minus P.
- **R2 the resumes happened.** On both arms at least 80% of RX turns 2 and 3 resume (a tokenization miss misses on both
  arms and is counted).
- **R3 door OFF.** Every `keep` RX row at 6,144 on the plain route equals `off-prev`'s.
- **R4 health.** No `CUDA_ERROR_OUT_OF_MEMORY` line, no 503, no crash line on any boot.

Readings, no bound (the price the owner reads): per card, route, L and G: re-primed rows per resume on `rewind` against
the suffix rows `keep` primes; resumed-turn TTFT p50 and p95 per arm and their ratio; generated tokens over boot wall
time; resumed-against-cold flips per arm; idle driver free and pool-cached bytes; FX's prefix hits and TTFT per arm.
Every median states N and the 250 ms regime.

### 1.5 CPU before the cards

Census tests: the arm is read at the two pools' exact and text hits and at the two arming sites, and nowhere else;
unarmed, the probe and admission text are today's. The client and reader dry-run against a local stub.

### 1.6 Price

Code: about 0.5 agent-day (the two pools' rewind, arming, receipts, census, the probe arm, the client and reader).
Cells: the 5090 about 3 h once its reset is done (it is down since 01:25Z, Xid 119 then 154); the target card about
4.5 h (the probe 20 min, 8 serving boots, `off-prev`).

### 1.7 Addendum A (2026-09-25, while writing the cells, before any cell)

1.3 did not say what the prefix cache does in these boots. It decides which program serves a turn: with the prefix
cache on, a turn-2 prompt can restore from turn 1's prompt-end seed (an on-grid entry) ahead of the pool probe, so the
pool's residual would not be the thing measured. So:

- **RX boots run with `MEMRA_PREFIX_CACHE_MB=0`**, as DAY38's cell ran, so every resume is the pool's (keep or rewind).
- **FX runs in its own boots with the prefix cache at its default**, because the in-batch fanout is a prefix-cache
  mechanism: per route, `fx-keep` and `fx-rewind`, one each.
- Boots per card: per route RX `keep`/`rewind` in both orders (4) and FX `keep`/`rewind` (2), 12 in all, plus
  `off-prev` (plain route, RX at 6,144 only, prefix cache off). The target card's time is about 6 h (the 122,880-token
  cold twins dominate), not the 4.5 h 1.6 priced.
- No clause, bound or reading of 1.4 changes.
- **`off-prev` is the tip with the door's commit reverted** (`day41-nodoor.patch`, the reverse of `424b6756e`'s
  `worker.rs` diff), not `a803d3080`: `origin/main` was merged into the lane (`7bcb6364d`) after 1.3 was written, so
  `a803d3080` would differ from the tip by main's changes too. The patch leaves the probe's new arm, which the server
  does not contain.

## 2. Results

### 2.1 The target card (the third sitting, one RTX PRO 6000 Blackwell Workstation Edition at 600 W, 2026-09-25 12:14 to 19:06Z)

Binaries built on the box from `99812fe38`: `tip` sha256 `d82565d6...3159c192`, `offprev` (the tip with
`day41-nodoor.patch`) `85a8493d...eebb8c21`, `concat-prime-probe` `bfb9ad40...60d5d27`; receipts at
`pro-single-day41/box/` (the box manifest checked, binaries by hash only). Section 1 is unchanged.

**The engine-level check (1.2), verbatim from `probe/SUMMARY.txt`:**

```
primepath: T=31040 (prompt 30720 + hist 256 + suffix 64) chat=false splits=[] steps=48 structured-row=0.5 structured-margin=0.5
verdict mono2: EXACT
verdict hist: NEAR-TIE-CLASS
cost hist: suffix_rows 65 wall_ms 744.4 (prime + 48 greedy steps)
verdict rewind: EXACT
cost rewind: checkpoint 30688 suffix_rows 352 (re-primed 288 over hist) wall_ms 776.8 (prime + 48 greedy steps)
primepath: T=30816 (prompt 30720 + hist 32 + suffix 64) chat=false splits=[] steps=48 structured-row=0.5 structured-margin=0.5
verdict mono2: EXACT
verdict hist: NEAR-TIE-CLASS
cost hist: suffix_rows 65 wall_ms 743.3 (prime + 48 greedy steps)
verdict rewind: EXACT
cost rewind: checkpoint 30688 suffix_rows 128 (re-primed 64 over hist) wall_ms 731.5 (prime + 48 greedy steps)
primepath: T=6464 (prompt 6144 + hist 256 + suffix 64) chat=false splits=[] steps=48 structured-row=0.5 structured-margin=0.5
verdict mono2: EXACT
verdict hist: NEAR-TIE-CLASS
cost hist: suffix_rows 65 wall_ms 666.2 (prime + 48 greedy steps)
verdict rewind: EXACT
cost rewind: checkpoint 6112 suffix_rows 352 (re-primed 288 over hist) wall_ms 697.8 (prime + 48 greedy steps)
primepath: T=6240 (prompt 6144 + hist 32 + suffix 64) chat=false splits=[] steps=48 structured-row=0.5 structured-margin=0.5
verdict mono2: EXACT
verdict hist: NEAR-TIE-CLASS
cost hist: suffix_rows 65 wall_ms 665.7 (prime + 48 greedy steps)
verdict rewind: EXACT
cost rewind: checkpoint 6112 suffix_rows 128 (re-primed 64 over hist) wall_ms 653.1 (prime + 48 greedy steps)
```

`rewind` reads `EXACT` against the monolithic prime at both lengths and both K; `hist` (the kept decoded rows) reads
`NEAR-TIE-CLASS` at all four. The rewind's own cost at the engine level is the re-primed G+32 rows: 731.5 against
743.3 ms at K=32 and 776.8 against 744.4 ms at K=256 for 30,720 tokens (prime of the suffix plus 48 greedy steps).

**The serving cells (1.3 to 1.4).** The box's `read.log` crashed in the FX section (`re.error: bad character range x-c
at position 8`, a raw-string regex inside an f-string); every line before the crash is in `read.log` as the box
printed it. The reader's two FX patterns were moved out of the f-string (no reading changes) and the reader re-ran
locally on the mirror (`read-local.log`, `SUMMARY-local.txt`); its non-FX lines equal the box's line for line. Verbatim:

```
DAY41 R4 card=pro6000 boot=fx-plain-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=fx-plain-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=fx-spec-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=fx-spec-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=offprev oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-plain-O1-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-plain-O1-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-plain-O2-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-plain-O2-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-spec-O1-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-spec-O1-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-spec-O2-keep oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R4 card=pro6000 boot=rx-spec-O2-rewind oom_lines=0 crash_lines=0 r503=0 -> PASS
DAY41 R2 card=pro6000 boot=rx-plain-O1-keep turns=60 resumed=60 frac=1.00 -> PASS
DAY41 FLIPS card=pro6000 boot=rx-plain-O1-keep resumed=60 flips_vs_cold=24 tags=['RX-6144-g32-r0-t2', 'RX-6144-g32-r0-t3', 'RX-6144-g32-r1-t2', 'RX-6144-g32-r2-t2', 'RX-6144-g32-r2-t3', 'RX-6144-g256-r0-t3'] (reading; keep carries the near-tie residual)
DAY41 R2 card=pro6000 boot=rx-plain-O1-rewind turns=60 resumed=40 frac=0.67 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-plain-O1-rewind resumed=40 flips_vs_cold=0 tags=[] (reading; keep carries the near-tie residual)
DAY41 R1 card=pro6000 boot=rx-plain-O1-rewind resumed=40 differ_vs_cold=[] grid_rewind_lines=40 r_equals_f_minus_p=True -> PASS
DAY41 PRICE card=pro6000 route=plain order=O1 L=6144 G=32 N=10 reprimed_rows p50=64 max=64 ttft_ms keep p50=69.1 p95=69.2 rewind p50=138.9 p95=139.2 ratio_p50=2.010 e2e_ms keep p50=453.6 rewind p50=523.2
DAY41 PRICE card=pro6000 route=plain order=O1 L=6144 G=256 N=10 reprimed_rows p50=288 max=288 ttft_ms keep p50=69.5 p95=70.0 rewind p50=182.5 p95=183.4 ratio_p50=2.626 e2e_ms keep p50=3153.0 rewind p50=3265.6
DAY41 PRICE card=pro6000 route=plain order=O1 L=30720 G=32 N=5 reprimed_rows p50=22656 max=24672 ttft_ms keep p50=96.0 p95=96.3 rewind p50=6873.2 p95=7370.6 ratio_p50=71.628 e2e_ms keep p50=529.5 rewind p50=7306.6
DAY41 PRICE card=pro6000 route=plain order=O1 L=30720 G=256 N=5 reprimed_rows p50=26880 max=28864 ttft_ms keep p50=96.6 p95=96.6 rewind p50=7973.3 p95=8433.4 ratio_p50=82.526 e2e_ms keep p50=3565.0 rewind p50=11441.7
DAY41 PRICE card=pro6000 route=plain order=O1 L=122880 G=32 N=5 reprimed_rows p50=5696 max=7680 ttft_ms keep p50=196.1 p95=196.5 rewind p50=3400.0 p95=4448.5 ratio_p50=17.338 e2e_ms keep p50=844.7 rewind p50=4048.4
DAY41 PRICE card=pro6000 route=plain order=O1 L=122880 G=256 N=5 reprimed_rows p50=10912 max=12896 ttft_ms keep p50=196.2 p95=196.6 rewind p50=6313.2 p95=7333.8 ratio_p50=32.179 e2e_ms keep p50=5380.3 rewind p50=11495.3
DAY41 THROUGHPUT card=pro6000 boot=rx-plain-O1-keep generated=25847 wall_s=2879.6 tok_per_s=8.98 idle driver_free=6960709632 pool_cached=6680322880
DAY41 THROUGHPUT card=pro6000 boot=rx-plain-O1-rewind generated=25920 wall_s=3603.7 tok_per_s=7.19 idle driver_free=6121848832 pool_cached=6891610944
DAY41 R2 card=pro6000 boot=rx-plain-O2-keep turns=60 resumed=60 frac=1.00 -> PASS
DAY41 FLIPS card=pro6000 boot=rx-plain-O2-keep resumed=60 flips_vs_cold=24 tags=['RX-6144-g32-r0-t2', 'RX-6144-g32-r0-t3', 'RX-6144-g32-r1-t2', 'RX-6144-g32-r2-t2', 'RX-6144-g32-r2-t3', 'RX-6144-g256-r0-t3'] (reading; keep carries the near-tie residual)
DAY41 R2 card=pro6000 boot=rx-plain-O2-rewind turns=60 resumed=40 frac=0.67 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-plain-O2-rewind resumed=40 flips_vs_cold=0 tags=[] (reading; keep carries the near-tie residual)
DAY41 R1 card=pro6000 boot=rx-plain-O2-rewind resumed=40 differ_vs_cold=[] grid_rewind_lines=40 r_equals_f_minus_p=True -> PASS
DAY41 PRICE card=pro6000 route=plain order=O2 L=6144 G=32 N=10 reprimed_rows p50=64 max=64 ttft_ms keep p50=69.8 p95=70.5 rewind p50=137.8 p95=138.2 ratio_p50=1.973 e2e_ms keep p50=454.3 rewind p50=522.4
DAY41 PRICE card=pro6000 route=plain order=O2 L=6144 G=256 N=10 reprimed_rows p50=288 max=288 ttft_ms keep p50=70.7 p95=71.5 rewind p50=181.6 p95=183.2 ratio_p50=2.569 e2e_ms keep p50=3153.5 rewind p50=3265.9
DAY41 PRICE card=pro6000 route=plain order=O2 L=30720 G=32 N=5 reprimed_rows p50=22656 max=24672 ttft_ms keep p50=96.0 p95=96.3 rewind p50=6885.0 p95=7380.0 ratio_p50=71.717 e2e_ms keep p50=529.6 rewind p50=7318.4
DAY41 PRICE card=pro6000 route=plain order=O2 L=30720 G=256 N=5 reprimed_rows p50=26880 max=28864 ttft_ms keep p50=96.2 p95=96.3 rewind p50=7975.3 p95=8432.9 ratio_p50=82.890 e2e_ms keep p50=3565.1 rewind p50=11444.7
DAY41 PRICE card=pro6000 route=plain order=O2 L=122880 G=32 N=5 reprimed_rows p50=5696 max=7680 ttft_ms keep p50=194.0 p95=194.4 rewind p50=3402.6 p95=4451.3 ratio_p50=17.537 e2e_ms keep p50=842.8 rewind p50=4051.0
DAY41 PRICE card=pro6000 route=plain order=O2 L=122880 G=256 N=5 reprimed_rows p50=10912 max=12896 ttft_ms keep p50=194.4 p95=194.5 rewind p50=6315.8 p95=7341.7 ratio_p50=32.490 e2e_ms keep p50=5379.8 rewind p50=11499.0
DAY41 THROUGHPUT card=pro6000 boot=rx-plain-O2-keep generated=25847 wall_s=2883.8 tok_per_s=8.96 idle driver_free=6960709632 pool_cached=6680322880
DAY41 THROUGHPUT card=pro6000 boot=rx-plain-O2-rewind generated=25920 wall_s=3604.9 tok_per_s=7.19 idle driver_free=6121848832 pool_cached=6891610944
DAY41 R2 card=pro6000 boot=rx-spec-O1-keep turns=60 resumed=34 frac=0.57 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-spec-O1-keep resumed=34 flips_vs_cold=6 tags=['RX-6144-g32-r2-t2', 'RX-6144-g256-r3-t2', 'RX-6144-g256-r4-t2', 'RX-30720-g256-r0-t3', 'RX-30720-g256-r2-t2', 'RX-30720-g256-r4-t2'] (reading; keep carries the near-tie residual)
DAY41 R2 card=pro6000 boot=rx-spec-O1-rewind turns=60 resumed=36 frac=0.60 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-spec-O1-rewind resumed=36 flips_vs_cold=0 tags=[] (reading; keep carries the near-tie residual)
DAY41 R1 card=pro6000 boot=rx-spec-O1-rewind resumed=36 differ_vs_cold=[] grid_rewind_lines=26 r_equals_f_minus_p=True -> FAIL
DAY41 PRICE card=pro6000 route=spec order=O1 L=6144 G=32 N=5 reprimed_rows p50=64 max=64 ttft_ms keep p50=158.1 p95=161.2 rewind p50=164.4 p95=164.6 ratio_p50=1.040 e2e_ms keep p50=418.8 rewind p50=425.1
DAY41 PRICE card=pro6000 route=spec order=O1 L=6144 G=256 N=3 reprimed_rows p50=288 max=288 ttft_ms keep p50=160.8 p95=162.5 rewind p50=207.3 p95=208.0 ratio_p50=1.289 e2e_ms keep p50=1767.7 rewind p50=1852.4
DAY41 PRICE card=pro6000 route=spec order=O1 L=30720 G=32 N=2 reprimed_rows p50=23216 max=23776 ttft_ms keep p50=216.4 p95=216.7 rewind p50=7073.3 p95=7194.5 ratio_p50=32.687 e2e_ms keep p50=469.1 rewind p50=7341.3
DAY41 PRICE card=pro6000 route=spec order=O1 L=30720 G=256 N=4 reprimed_rows p50=27040 max=29184 ttft_ms keep p50=215.4 p95=216.0 rewind p50=8123.4 p95=8698.2 ratio_p50=37.711 e2e_ms keep p50=2039.5 rewind p50=9994.0
DAY41 PRICE card=pro6000 route=spec order=O1 L=122880 G=32 N=10 reprimed_rows p50=0 max=7680 ttft_ms keep p50=412.8 p95=3492.7 rewind p50=1296.8 p95=3994.0 ratio_p50=3.141 e2e_ms keep p50=800.0 rewind p50=1702.0
DAY41 PRICE card=pro6000 route=spec order=O1 L=122880 G=256 N=10 reprimed_rows p50=144 max=10912 ttft_ms keep p50=436.5 p95=6627.6 rewind p50=2667.7 p95=6648.0 ratio_p50=6.112 e2e_ms keep p50=3528.3 rewind p50=5753.5
DAY41 THROUGHPUT card=pro6000 boot=rx-spec-O1-keep generated=25920 wall_s=2763.5 tok_per_s=9.38 idle driver_free=3368288256 pool_cached=5683220928
DAY41 THROUGHPUT card=pro6000 boot=rx-spec-O1-rewind generated=25920 wall_s=2816.6 tok_per_s=9.20 idle driver_free=4207149056 pool_cached=2157736364
DAY41 R2 card=pro6000 boot=rx-spec-O2-keep turns=60 resumed=34 frac=0.57 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-spec-O2-keep resumed=34 flips_vs_cold=6 tags=['RX-6144-g32-r2-t2', 'RX-6144-g256-r3-t2', 'RX-6144-g256-r4-t2', 'RX-30720-g256-r0-t3', 'RX-30720-g256-r2-t2', 'RX-30720-g256-r4-t2'] (reading; keep carries the near-tie residual)
DAY41 R2 card=pro6000 boot=rx-spec-O2-rewind turns=60 resumed=36 frac=0.60 -> FAIL
DAY41 FLIPS card=pro6000 boot=rx-spec-O2-rewind resumed=36 flips_vs_cold=0 tags=[] (reading; keep carries the near-tie residual)
DAY41 R1 card=pro6000 boot=rx-spec-O2-rewind resumed=36 differ_vs_cold=[] grid_rewind_lines=26 r_equals_f_minus_p=True -> FAIL
DAY41 PRICE card=pro6000 route=spec order=O2 L=6144 G=32 N=5 reprimed_rows p50=64 max=64 ttft_ms keep p50=161.9 p95=162.3 rewind p50=160.7 p95=161.1 ratio_p50=0.993 e2e_ms keep p50=422.7 rewind p50=421.4
DAY41 PRICE card=pro6000 route=spec order=O2 L=6144 G=256 N=3 reprimed_rows p50=288 max=288 ttft_ms keep p50=161.0 p95=161.6 rewind p50=204.5 p95=204.7 ratio_p50=1.271 e2e_ms keep p50=1766.5 rewind p50=1850.7
DAY41 PRICE card=pro6000 route=spec order=O2 L=30720 G=32 N=2 reprimed_rows p50=23216 max=23776 ttft_ms keep p50=216.2 p95=216.4 rewind p50=7094.6 p95=7223.1 ratio_p50=32.820 e2e_ms keep p50=469.6 rewind p50=7362.3
DAY41 PRICE card=pro6000 route=spec order=O2 L=30720 G=256 N=4 reprimed_rows p50=27040 max=29184 ttft_ms keep p50=215.6 p95=215.9 rewind p50=8162.5 p95=8735.4 ratio_p50=37.867 e2e_ms keep p50=2040.2 rewind p50=10033.2
DAY41 PRICE card=pro6000 route=spec order=O2 L=122880 G=32 N=10 reprimed_rows p50=0 max=7680 ttft_ms keep p50=419.0 p95=3513.2 rewind p50=1298.8 p95=4006.6 ratio_p50=3.100 e2e_ms keep p50=806.4 rewind p50=1704.1
DAY41 PRICE card=pro6000 route=spec order=O2 L=122880 G=256 N=10 reprimed_rows p50=144 max=10912 ttft_ms keep p50=439.5 p95=6665.9 rewind p50=2676.0 p95=6658.9 ratio_p50=6.088 e2e_ms keep p50=3532.7 rewind p50=5762.9
DAY41 THROUGHPUT card=pro6000 boot=rx-spec-O2-keep generated=25920 wall_s=2777.8 tok_per_s=9.33 idle driver_free=3770941440 pool_cached=2593943980
DAY41 THROUGHPUT card=pro6000 boot=rx-spec-O2-rewind generated=25920 wall_s=2824.8 tok_per_s=9.18 idle driver_free=4207149056 pool_cached=2157736364
DAY41 R3 card=pro6000 keep=rx-plain-O1-keep rows=60 differ=[] -> PASS
DAY41 R3 card=pro6000 keep=rx-plain-O2-keep rows=60 differ=[] -> PASS
DAY41 FX card=pro6000 boot=fx-plain-keep ok=8 ttft_ms p50=6999.2 p95=7002.0 cached_tokens_sum=7168 prefix_hit_lines=5 fanout_lines=1
DAY41 FX card=pro6000 boot=fx-plain-rewind ok=8 ttft_ms p50=9070.2 p95=9071.2 cached_tokens_sum=0 prefix_hit_lines=0 fanout_lines=0
DAY41 FX card=pro6000 boot=fx-spec-keep ok=8 ttft_ms p50=7324.6 p95=7325.8 cached_tokens_sum=6144 prefix_hit_lines=0 fanout_lines=1
DAY41 FX card=pro6000 boot=fx-spec-rewind ok=8 ttft_ms p50=9076.6 p95=9079.8 cached_tokens_sum=0 prefix_hit_lines=0 fanout_lines=0
```

**Clauses, as they read:**

- **R4 PASS on all 13 boots. R3 PASS in both orders** (60 of 60 `keep` rows equal `offprev`'s).
- **R1 on the plain route: PASS in both orders** (40 resumes, none differs from its cold twin, 40 `grid-rewind:` lines,
  every R equal to the committed rows minus P).
- **R1 on the spec route: FAIL in both orders** on the line half (`resumed=36 differ_vs_cold=[] grid_rewind_lines=26`).
  No resume differs from its cold twin. The ten resumes without a `grid-rewind:` line are the spec tier's existing
  affinity rewind (`spec-affinity: rewound to 119200 of 122976 prompt tokens (fingerprint; priming 3776 suffix)`), all
  at 122,880 tokens: the exact probe missed (the next item), the fingerprint nominated, and the session rewound to its
  turn checkpoint by the affinity path, not the arm's.
- **R2 on `rewind`, plain route: FAIL in both orders** (40 of 60). All 20 misses are turn 3 at 30,720 and 122,880:
  `grid-rewind: plain declined (no checkpoint); cold`. Placed in the code: turn 2 resumed at the checkpoint P, and the
  arming rule `plain_checkpoint_boundary(prompt).filter(|&b| b > seed_fed.len())` then found the same boundary P for
  turn 2's prompt, which is not ahead of the resumed depth, so turn 2 armed no checkpoint.
- **Why P sits where it does.** The workload is `docs/SERVING.md` tokenized, and that file contains the literal text
  `<|im_start|>` and `<|im_end|>`, which tokenize to control tokens. The 6,144-token prompts hold none; the 30,720 and
  122,880-token windows hold one, and `plain_checkpoint_boundary` puts the checkpoint at the grid floor of the last
  control token (10,080 down to 2,112 at 30,720 as the window slides 997 tokens per conversation; 119,200 down to
  110,240 at 122,880), not at the prompt end. This sets the re-primed rows at those lengths (p50 22,656 and 26,880 at
  30,720), so the PRICE lines there read the boundary rule on this text, not the arm's floor, which is the probe's
  G+32 rows.
- **R2 on the spec route: FAIL on both arms in both orders** (`keep` 34 of 60, `rewind` 36 of 60). Placed in the code:
  a spec session commits the final burst's accepted drafts past `max_tokens` (`SpecSession::committed`, "INCLUDING
  overshoot"), so its parked `committed` is longer than the public stream the client sends back, and the exact probe
  (`prompt.starts_with(&e.sess.committed)`) misses whenever the last burst overshot; a `prompt_ids` request has no text
  probe. This is today's program on the default route, not the arm.

**Readings, target card (N per line; the 250 ms regime in each boot's `samples.csv`).**

- Flips against cold on `keep`: 24 of 60 resumed turns on the plain route in both orders, 6 of 34 on the spec route;
  on `rewind`: 0 of 40 (plain) and 0 of 36 (spec).
- The price where the checkpoint sits at the prompt end (6,144, plain): TTFT p50 69.1 against 138.9 ms at G=32 (x2.01)
  and 69.5 against 182.5 ms at G=256 (x2.63), E2E +70 and +113 ms; spec route x1.04 and x1.29.
- Where the checkpoint sits at an interior control token (30,720 and 122,880): TTFT p50 x17 to x83 on the plain route
  and x3 to x38 on the spec route, the whole distance from the control token re-primed.
- Throughput over the boot wall (plain, both orders): 8.97 against 7.19 tokens/s; spec 9.36 against 9.19.
- FX: the armed checkpoint takes the fanout away, as 1.1(c) stated. Plain TTFT p50 7.0 s on `keep` (5 prefix-cache
  hits, one fanout line) against 9.07 s on `rewind` (no hits, no fanout); spec 7.3 against 9.08 s.

The arm is revised by addendum B (1.8) before any code of the revision.
