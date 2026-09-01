# B1FAST/GraphSession demotion cost — box1 result

Date: 2026-08-13
Status: **COMPLETE — the solo loss is real; the sold cache-hit envelope is FLAT; EAGER remains unsafe.**

## Verdict

The expected shape is confirmed.

1. **c=1 solo cost:** the repaired defaults lose **12.183% on Q27** and **7.234% on
   Q35** in request-visible end-to-end output rate. Looking only at sustained decode after
   first token, the losses are **13.753%** and **10.524%**, respectively.
2. **c>=4 revenue cost:** **0% detectable in the mixed 90%-cache-hit sold shape** under the
   required min/max-overlap rule. Its median repair costs range from a 0.205% win to a
   0.273% loss; all four arm ranges overlap and are therefore **FLAT**.

The full-hit long-decode ladder also shows no repaired-arm loss at c>=4: four cells are
FLAT and the other two are small repaired-arm wins. There is no evidence here for doing
sticky-or-refused graph promotion to recover sold-envelope throughput. Re-earning the graph
win remains a solo-latency optimization, not a revenue-throughput blocker.

Positive `repair cost` below means REPAIRED is slower than EAGER:
`(1 - repaired_median / eager_median) * 100`. A cell is FLAT whenever the two observed
N=5 min..max intervals overlap. Every throughput median below is N=5 independent fresh-server
launches **per arm, per cell** under the same uninterrupted no-artificial-cooldown regime:
GPU0 26..60 C, 600 W, active SM clocks 1,282..2,422 MHz; GPU1 reserved idle.

## Exactness headline — seventh early-EOS class sighting

Before the valid attempt-3 ladder, attempt 2 reproduced the corruption independently: Q27 EAGER
at c=16 selected EOS at token 11 on request 11 of the frozen sellgate workload. The exact raw
receipt is `completion_tokens: 11, finish_reason: 'stop'` in
[`driver.log`](raw/box1-attempt2/driver.log). The driver raised immediately and refused to publish
that cell; an 11-token EAGER response is not comparable with a 512-token REPAIRED response.

That reproduction is the load-bearing repair confirmation recorded at `6aba8b2e5`. The separate
pre-repair `cx-cachesize` campaign later stopped at 11/60 tokens too, making the class **seven
independent triggers** and this exact token-11 signature its second cross-lane match. No throughput
from the bad gscost cell appears anywhere below.

## Request-visible full-cache ladder

This is aggregate completion tokens divided by the complete scored-wave wall time. It
includes first-token latency and GraphSession's one-time capture cost. Each request used
the same 172-token fully restored chat prompt and generated exactly 512 tokens with
`finish_reason=length`.

| model | c | REPAIRED N=5 median [min..max], tok/s | EAGER N=5 median [min..max], tok/s | repair cost | verdict |
|---|---:|---:|---:|---:|---|
| Q27 | 1 | 68.677 [68.417..68.691] | 78.204 [78.100..78.276] | **+12.183%** | **REPAIR LOSS** |
| Q27 | 4 | 241.968 [241.651..242.190] | 238.932 [238.542..242.159] | -1.270% | FLAT |
| Q27 | 16 | 401.131 [400.280..401.487] | 398.619 [398.045..401.428] | -0.630% | FLAT |
| Q27 | 40 | 394.796 [394.664..394.851] | 393.626 [393.536..393.650] | -0.297% | REPAIR WIN |
| Q35 | 1 | 228.710 [225.368..229.385] | 246.544 [246.242..246.602] | **+7.234%** | **REPAIR LOSS** |
| Q35 | 4 | 681.727 [677.997..682.872] | 678.499 [656.737..680.547] | -0.476% | FLAT |
| Q35 | 16 | 975.660 [972.210..989.722] | 963.703 [960.579..963.997] | -1.241% | REPAIR WIN |
| Q35 | 40 | 990.371 [989.542..992.595] | 986.011 [978.097..990.679] | -0.442% | FLAT |

## Sustained full-cache decode

This removes each request's first-token interval and divides the remaining generated tokens
by the shared decode window. It isolates the decode-program difference while retaining the
same N=5 waves and fixed work.

| model | c | REPAIRED N=5 median [min..max], tok/s | EAGER N=5 median [min..max], tok/s | repair cost | verdict |
|---|---:|---:|---:|---:|---|
| Q27 | 1 | 68.564 [68.306..68.577] | 79.497 [79.396..79.531] | **+13.753%** | **REPAIR LOSS** |
| Q27 | 4 | 241.687 [241.295..241.786] | 241.662 [241.376..241.801] | -0.010% | FLAT |
| Q27 | 16 | 400.416 [399.575..400.769] | 400.081 [399.531..400.745] | -0.084% | FLAT |
| Q27 | 40 | 394.066 [393.948..394.136] | 393.701 [393.627..393.726] | -0.093% | REPAIR WIN |
| Q35 | 1 | 228.457 [225.110..229.141] | 255.327 [255.319..255.515] | **+10.524%** | **REPAIR LOSS** |
| Q35 | 4 | 681.208 [677.163..682.301] | 677.956 [672.827..679.982] | -0.480% | FLAT |
| Q35 | 16 | 974.122 [970.666..988.174] | 970.097 [967.160..970.603] | -0.415% | REPAIR WIN |
| Q35 | 40 | 988.662 [987.946..990.926] | 987.531 [979.578..992.101] | -0.115% | FLAT |

## Mixed 90%-cache-hit money shape

These cells use the frozen 4,860+60 cx-requal replay: 90% cache-hit tokens and 10% real
misses, with the complete cell wall time in the denominator. Q27 was measured at c=4 and
its c=16 knee; Q35 at c=4 and its c=40 knee.

| model | c | REPAIRED N=5 median [min..max], tok/s | EAGER N=5 median [min..max], tok/s | repair cost | verdict |
|---|---:|---:|---:|---:|---|
| Q27 | 4 | 145.276 [144.248..148.802] | 145.419 [144.696..148.595] | +0.098% | **FLAT** |
| Q27 | 16 | 189.317 [189.097..189.903] | 189.364 [189.278..189.508] | +0.025% | **FLAT** |
| Q35 | 4 | 411.550 [410.407..414.605] | 410.707 [408.386..416.915] | -0.205% | **FLAT** |
| Q35 | 40 | 522.006 [521.726..523.933] | 523.435 [522.517..524.504] | +0.273% | **FLAT** |

All 40 mixed cells were clean: 1,000/1,000 requests succeeded, the measured cache-hit token
ratio was exactly 0.9, and cached-token/prefix-cache counter drift was zero.

## What GraphSession actually covered

The EAGER activation probe captured a GraphSession on both models for a fully restored
prefix-cache hit. Q35 is ineligible for B1FAST, so its c=1 delta is a clean GraphSession
measurement; Q27 c=1 enables both requested eager doors.

A genuinely cold request does **not** currently promote. The worker checks promotion before
prefill and requires `prefill_done && generated.is_empty()`; cold prefill and token 1 complete
later in the same tick, and the next tick fails `generated.is_empty()`. Attempt 1 stopped
before scoring with the exact message `FAIL: q27 EAGER activation probe did not capture a
GraphSession`. Therefore the c=1 tables quantify the eligible solo **full-cache-hit** request,
not a cold miss. At c>=4 GraphSession cannot remain active once peers are present, and the
mixed 60-token budget is also below its 384-token gate.

## Protocol and provenance

- Runtime base: `904a5d5f32a1b9170bc8628f2392cb0287572dbe`. The campaign checkout was
  `9db1310ddf8a8584d28d71e32a51220daf972c7a`; the binary was built at research checkpoint
  `71682fdbdf4a3616b3ab1794bf4b75607d35dcfd`. Every delta from the base through both
  checkpoints was under this research lane, and the runtime/build tree remained unchanged.
- One fresh sm_120a release binary served both arms:
  `d314dfc211918523d93e14454a56a323dfa8544c974254a0e7e236e822848846`.
  REPAIRED left both variables unset; EAGER set only
  `MEMRA_SERVE_B1FAST=1 MEMRA_SERVE_GS=1`.
- Rig: box1, 2x RTX PRO 6000 Blackwell Server Edition, 600 W limits. GPU0 alone ran the
  campaign; GPU1 was reserved idle because the pair shares a PIX path.
- One uninterrupted `/tmp/memra-gpu.lock` hold covered activation, qualification, all scored
  cells, and the pre-release battery: 06:30:17Z through 07:25:46Z. Each point used a fresh
  server. N=5; odd repetitions ran REPAIRED-first and even repetitions EAGER-first.
- Main receipt: 80/80 points, 1,220/1,220 scored requests, zero errors. Mixed receipt: 40/40
  points, 1,000/1,000 scored requests, zero errors.
- Thermal regime: no artificial cooldown; GPU0 ranged 26..60 C at a 600 W limit, with active
  SM clocks 1,282..2,422 MHz. The continuous 250 ms stream has 13,075 samples per GPU.
  GPU1 remained P8 at 26..27 C, 0% utilization, and zero active samples.
- The 15 one-MiB GPU1 rows are **not** a startup cluster: they are isolated from 06:31:01Z through
  07:21:34Z, 15/13,075 samples (0.1147%), with a maximum run of one sample. Every row is
  P8/180 MHz/0% and coincides with a same-time one-MiB GPU0 reading; adjacent GPU1 samples
  are 0 MiB. The corrected guard tolerates only that exact one-quantum idle signature. It rejects
  any allocation above 1 MiB, any utilization, any non-P8 state, or any clock above 200 MHz; it
  also rejects a one-MiB run over two seconds or a campaign share over 0.5%. Every refusal prints
  the reasons, summary, and timestamped signal rows. This preserves the shared-PIX fail-closed
  doctrine while removing the MiB-accounting false positive.
- Frozen mixed harness SHA-256:
  `91eac7250e0d268ac6be8cfd1ee64e346d405dc412824dab45f224e9563e1e5b`;
  workload lock SHA-256:
  `85597a0a28ed874f440b4a966c0b43fd3e31b94fe868266de9e299decc208c34`.

The stale board comparison is not used in any result. The deterministic reduction is
[`raw/box1-attempt3/summary.json`](raw/box1-attempt3/summary.json); all raw artifacts are
covered by [`raw/SHA256SUMS`](raw/SHA256SUMS), whose SHA-256 is
`4d9897574e2bc86f05577b733e2a539d772fe0eeee292d720cbab4da8c46f59e`.

## PRO-class recommendation

**Keep the REPAIRED defaults on the 188-SM RTX PRO 6000 class.** The c=1 cost is real, but the
c>=4 money shape is FLAT and the EAGER arm independently reproduced the corruption. This is only
the PRO half of the owner decision: one-rig evidence sets at most a one-rig default. The separate
`cx-armrig5090` result must decide the 82-SM RTX 5090 half before any combined runtime policy.

Any future EAGER promotion needs an explicit hardware crossing guard, not a bare environment
default: use `Engine::sm_count()` to distinguish the qualified 82-SM and 188-SM paths, default
unknown devices to REPAIRED, and keep `MEMRA_SERVE_B1FAST` / `MEMRA_SERVE_GS` as forced-off
rollback seams. On the evidence here, the 188-SM branch stays REPAIRED.

## Pre-release battery — PASS on the same lock hold

**PRE-RELEASE BATTERY: PASS.** All five `.exit` receipts are `0`; no failure text exists to quote.
For this exact binary on the PRO 6000 verification box, the correctness gate required before a tag
is satisfied. This lane does not create that tag.

| gate | receipt |
|---|---|
| `kernel-check` | **ALL GREEN** — 100 cells, 5 skipped; exit 0 |
| `run-gen` Q27 | prefill/decode argmax MATCH; batched-prime/tokenwise MATCH; exit 0 |
| `run-gen` Q35 | prefill/decode argmax MATCH; batched-prime/tokenwise MATCH; exit 0 |
| `run-spec` Q27 | K=1..8, **8/8 self-consistency PASS**; overall PASS; exit 0 |
| `run-spec` Q35 | K=1..8, **8/8 self-consistency PASS**; overall PASS; exit 0 |

The first reducer invocation exited after the battery because its initial GPU1 rule treated
any nonzero rounded memory sample as work. The exact raw failure is retained in
[`reduce.log`](raw/box1-attempt3/reduce.log). Correlation established the 1 MiB driver-accounting
pattern above; the corrected deterministic reducer re-read all 2,680 main rows, 3,621 mixed rows,
and 13,075 thermal samples per GPU without changing any scored row. Post-cleanup live proof records
both GPUs at 0 MiB/P8, no compute applications, the shared lock free, and port 18468 clear in
[`post-cleanup-live.log`](raw/box1-attempt3/post-cleanup-live.log).

## Discarded attempts

- Attempt 1 was pre-score activation development only; it stopped because a cold request did
  not instantiate GraphSession.
- Attempt 2 is the fail-closed exactness sighting reported above; no attempt-2 throughput enters
  these tables.

No performance board, merge, tag, push, or release was made.
