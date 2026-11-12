# An exact re-send restores its whole prompt; a restored hit's suffix primes first (memra#1233)

Verdict: fixed. memra#1233 reported a prefix-cache hit's first token at about 175 ms on the frozen
mixed hit/cold c = 4 cell against 18.5 ms in an old receipt. The cause is not the card or the
checkpoint. Since memra#602 a prompt-end seed publishes on the 32-token GDN grid, so an exact
re-send of a 4,860-token prompt restores 4,832 tokens and primes a 28-token remainder before its
first token (115 to 136 ms for that call alone), and that call shared its scheduler tick with
unrelated cold 1,024-token prime chunks. Two changes:

- **(a) the prompt-end tail.** The grid entry carries the state its publishing prime ended in. An
  exact re-send restores the whole prompt and decodes from the restored logits; a longer prompt
  still restores the grid entry, so memra#602's law is unchanged.
- **(b) the hit-suffix fence.** A resumed row whose remaining suffix completes in one prime call
  primes ahead of every cold chunk. The fence is bounded.

On GB10 at the frozen 4,096 MiB prefix budget, hit TTFT p50 / p95 goes from 322 / 1,257 ms on
`main` to 88 / 218 ms: 78 of 90 hits restore through a tail and all 90 are restored. On one RTX
5090 Laptop GPU the budget that fits beside the model (2,800 MB) is full of entries, so 7 of 90
hits find a tail (31 ms p50) and the fence is the gain: restored grid hits go from 293 to 391 /
1,216 to 1,251 ms on `main` to 158 to 175 / 410 to 424 ms. Every text equals its cold text. Qwen3.8-27B
NVFP4 + Q5_K with its MTP head loaded, plain path (`MEMRA_SERVE_SPEC=0`).

The numbers above are the PR's code (`The PR's code` below). Commit ids in this file name the
lane's branch before it was squashed for the PR: f1ccab0dc7 is the PR's crates, 6af34243a2 the
round before the review, and the arms' ids are in `Setup`. Most of this file is the round
before it, on a build that differs by the review's changes; it carries the comparisons that chose
the design.

Why this design and what was rejected: [`docs/decisions/PREFIX-PROMPT-END-TAIL.md`](../../docs/decisions/PREFIX-PROMPT-END-TAIL.md).
Not built here: batching concurrent hit suffixes into one prime call (memra#1239).

## Setup

- The cell is the frozen sellgate shape of `research/sellgate-20260812/` and
  `research/coldfix-20260812/`: 4,860-token prompts, exactly 60 completion tokens, temperature 0,
  mixed90 (nine hits to one miss), four closed-loop clients, 20 requests per repetition, five
  repetitions per boot after the hot set is seeded: 90 hit-role and 10 miss-role requests per
  boot. The harness is the sellgate replay itself, imported by `tools/hitcold-cell.py`.
- Server: `MEMRA_SERVE_SPEC=0 MEMRA_CTX=8192 MEMRA_REUSE_POOL=0 MEMRA_AFFINITY=0
  MEMRA_MAX_SESSIONS=96`, tick and TTFT traces on. `MEMRA_PREFIX_CACHE_MB=4096` on GB10 (the
  frozen budget) and 2800 on the RTX 5090, where 4,096 MiB does not fit beside the model in 24 GB.
- A hit-role request that finds no entry (the budget evicted it) restores nothing and is a cold
  prime; the tables count it as not restored and its TTFT is in the all-90 columns. "Whole" is a
  restore of all 4,860 tokens through a tail, "grid" a restore of 4,832 and a primed remainder.
- Devices: one RTX 5090 Laptop GPU (24 GB) and one DGX Spark (GB10). One boot per lease. The RTX
  5090's SM clock moves inside a window (1,590 to 1,654 MHz boot means), so arms are compared
  inside one window, in a symmetric order, and read against the clock.
- Binaries, release builds of this repo, the RTX 5090 ones built with nvcc 13.1 for `sm_120a` on a
  build server, the GB10 ones on the unit (`sm_121a`):

  | arm | source | RTX 5090 sha256[:12] | GB10 sha256[:12] |
  |---|---|---|---|
  | main | d1ed663256 | ad438a3319ce | 545dda6c6301 |
  | b-only | 936ee4552e (the fence commit alone, no bound) | fb4189e8612c | 11118dff535e |
  | fix | 6af34243a2 | c3427b22ffdd | d65351ac4ed9 |
  | tails-first | 4201b67ffd (the fix with every tail evicted before any entry) | 241e9e0c6cbb | 002d4dfdb9a5 |
  | LRU-item | 70a2d0183b (a proven tail is an LRU item beside the entries) | | 3b3a0d7ea742 |

  These are the binaries of the round before the review; the PR's code and its binaries are in
  `The PR's code`.

## The PR's code

The independent review of the PR changed three things in the scheduler and the cache, and the
gates and one window per device were run again on that code (f1ccab0dc7 on `main` 1d3be5b0df;
RTX 5090 binaries fix b9596c4979fb and main f546ecf5227c, GB10 fix 857e7d31fa6a and main
af051d8edd79):

- A whole-prompt hit decodes before the fenced suffixes. On a tick where one waits for its first
  token the fence stands down and the suffixes prime on the next tick. Before, 11 of 17
  whole-prompt hits on the RTX 5090 waited 126 to 278 ms behind one or two suffix calls
  (`rtx5090/abba-telemetry/`); now they take 31 ms p50 there and their p95 on GB10 is 101 to 106
  ms (was 210 to 215).
- The fence's allowance is the time of one tick-budget chunk at the measured prefill rate, and the
  fence stands down while the deadline-aware prefill order is on. The longest prefill tick the
  worker ever ran, the yardstick of the round before, is set by a solo-widened prime (3.9 s on the
  RTX 5090, 3.1 s on GB10 in the banked boots) and allowed 25 to 30 fenced ticks in a row. No
  measured cell queues enough hits to reach either bound: the bound is proven by its unit test
  and its red arms, not on a GPU.
- A fanout entry counts as having served a hit; admission books a pending tail for plain sessions
  only; the serving collector's tail witness reads its own cell's log lines.

- A reclaim that tails alone cover is settled. `evict_to_bytes` strips tails before entries and
  can free bytes with no entry evicted, and the kv-flex shed, the transient reserve and the
  reclaim on defer keyed on the entry count: such a pass skipped the settle, so its bytes stayed
  in the pool. A review bot found it on the PR. All three key on the bytes freed now
  (`a_reclaim_that_tails_alone_cover_is_not_a_failure`, red in `cpu/red-fence/red-reclaim.log`).
  The cells below predate this change; after it the three reclaim-touching gates passed again on
  the RTX 5090 (`rtx5090/reclaim/gates/`: exact re-send 9/9 and covered 4/4, day-16
  newest-turn-fits 12/12 and 31/31, restore-identity 5/5) and one GB10 pair read main 321 / 1,262 ms and the fix 87 / 318 ms, 78 of 90 through a tail, 90 of 90 restored
  (`gb10/reclaim/`). CPU: `cpu/checks-reclaim.log` (1,210 server tests).

GB10, 4,096 MiB, main / fix / fix / main:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| main-035509 | main | 90 (0 + 90) | 322 / 1,257 | - / - | 322 / 1,257 | 17 / 1,011 | 5,154 / 9,630 | 32.7 | - |
| fix-040043 | fix | 90 (78 + 12) | 88 / 218 | 88 / 106 | 203 / 375 | 6 / 166 | 5,016 / 9,540 | 34.3 | 8 / 78 / 22 / 2 |
| fix-040624 | fix | 90 (78 + 12) | 89 / 256 | 88 / 101 | 203 / 318 | 6 / 108 | 5,013 / 9,444 | 34.3 | 8 / 78 / 22 / 2 |
| main-041133 | main | 90 (0 + 90) | 322 / 1,212 | - / - | 322 / 1,212 | 12 / 969 | 5,240 / 9,421 | 32.6 | - |

RTX 5090, 2,800 MB, main / fix / fix / main, 1 Hz telemetry. Other lanes' leases ran between
these boots, so the clock moved more than inside the earlier windows:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| main-041526 | main | 82 (0 + 82) | 391 / 4,453 | - / - | 391 / 1,216 | 118 / 964 | 4,586 / 8,132 | 44.5 | - |
| fix-042419 | fix | 78 (7 + 71) | 260 / 7,206 | 31 / 123 | 175 / 424 | 2 / 253 | 4,872 / 9,080 | 40.6 | 18 / 7 / 83 / 17 |
| fix-043112 | fix | 78 (7 + 71) | 158 / 7,000 | 31 / 119 | 158 / 410 | 2 / 192 | 4,677 / 8,481 | 43.0 | 18 / 7 / 83 / 17 |
| main-043742 | main | 81 (0 + 81) | 296 / 4,758 | - / - | 293 / 1,251 | 6 / 1,009 | 4,920 / 8,599 | 41.4 | - |

| window | boot | hit-role requests not restored | cold-chunk ticks / tokens / s | ms per 1,000 cold tokens | short prime ticks / s | decode s | cold TTFT p50 ms | output tok/s | SM clock MHz / power W, mean |
|---|---|---|---|---|---|---|---|---|---|
| abba | main-041526 | 8 | 88 / 126,080 / 99.2 | 787.1 | 61 / 11.1 | 59.3 | 4,586 | 44.5 | 1667 / 123 |
| abba | fix-042419 | 12 | 98 / 145,044 / 122.1 | 841.8 | 75 / 12.6 | 61.1 | 4,872 | 40.6 | 1540 / 110 |
| abba | fix-043112 | 12 | 98 / 145,044 / 115.0 | 792.9 | 80 / 11.8 | 60.0 | 4,677 | 43.0 | 1632 / 120 |
| abba | main-043742 | 9 | 90 / 131,052 / 108.2 | 826.0 | 66 / 11.2 | 60.8 | 4,920 | 41.4 | 1573 / 108 |

**The slow tail of the all-90 column.** The fix reads 7,206 and 7,000 ms at p95 over all 90
hit-role requests where `main` reads 4,453 and 4,758. That is a rank, not a wait. A hit-role
request that restores nothing is a cold prime, and the p95 sits at rank 84.55 of 89: with M such
requests it is the (M - 5.45)-th slowest of them. Their TTFTs fall in two groups, about 4.3 to 5.6
s (a prime of their own) and 8.3 to 9.2 s (two cold primes sharing their ticks), so the p95 reads
4.5 to 5.2 s while M is 8 to 10 and 6.8 to 7.6 s once M is 12, on any arm (`main-021813`: M = 12, p95
6,887 ms; `fix-020256`: M = 10, 4,886 ms). `tools/slow-tail.py`, the PR's code first, then the
telemetered window of the round before:

| boot | not restored (M) | all-90 p95 ms | not-restored TTFTs, ms, ascending | cold-role TTFT p50 / p95 / max | ms per 1,000 cold tokens | cold-role p50 / p95 as multiples of an unshared prime | longest fenced run, ticks / ms |
|---|---|---|---|---|---|---|---|
| main-041526 | 8 | 4,453 | 4,301 4,342 4,398 4,498 4,575 4,615 9,191 9,192 | 4,586 / 8,132 / 8,132 | 787 | 1.20 / 2.13 | 1 / 246 |
| fix-042419 | 12 | 7,206 | 4,586 4,809 4,841 4,862 4,872 5,454 5,586 8,531 8,532 8,570 8,600 8,989 | 4,872 / 9,080 / 9,080 | 842 | 1.19 / 2.22 | 1 / 372 |
| fix-043112 | 12 | 7,000 | 4,417 4,543 4,571 4,700 4,752 5,085 5,251 8,431 8,461 8,461 8,461 8,685 | 4,677 / 8,481 / 8,685 | 793 | 1.21 / 2.20 | 2 / 356 |
| main-043742 | 9 | 4,758 | 4,269 4,552 4,585 4,619 4,871 4,915 8,942 8,942 8,982 | 4,920 / 8,599 / 8,803 | 826 | 1.23 / 2.14 | 1 / 135 |
| main-015856 | 11 | 4,904 | 4,220 4,241 4,297 4,402 4,615 4,619 5,137 5,265 8,305 8,514 8,515 | 4,648 / 8,224 / 8,468 | 784 | 1.22 / 2.16 | 1 / 246 |
| fix-020256 | 10 | 4,886 | 4,360 4,492 4,651 4,654 4,664 5,068 5,117 8,466 8,525 8,525 | 4,726 / 8,078 / 8,078 | 780 | 1.25 / 2.13 | 1 / 364 |
| bonly-020637 | 12 | 7,130 | 4,606 4,667 4,728 4,745 4,748 5,388 5,594 8,387 8,388 8,459 8,460 8,732 | 4,902 / 9,076 / 9,076 | 820 | 1.23 / 2.28 | 2 / 381 |
| bonly-021032 | 10 | 4,672 | 4,300 4,441 4,481 4,648 4,653 4,688 5,154 5,324 9,400 9,430 | 4,753 / 8,585 / 8,585 | 805 | 1.21 / 2.19 | 2 / 377 |
| fix-021418 | 12 | 6,989 | 4,461 4,641 4,667 4,726 4,735 5,209 5,312 8,360 8,391 8,470 8,471 8,652 | 4,667 / 8,551 / 8,653 | 801 | 1.20 / 2.20 | 1 / 258 |
| main-021813 | 12 | 6,887 | 4,335 4,365 4,558 4,582 4,588 5,098 5,177 8,286 8,316 8,338 8,339 8,561 | 4,663 / 8,412 / 8,561 | 789 | 1.22 / 2.19 | 1 / 360 |

- The cold-role requests take the same multiple of an unshared prime on every arm: p50 1.19 to
  1.25, p95 2.13 to 2.28 (ten per boot, so the p95 is close to the slowest one). The fence does
  not stretch a cold request beyond what the clock explains.
- The fence's longest run is one or two ticks, 135 to 418 ms of prefill, against a chunk of about
  800 ms: it stays inside its bound of one chunk plus one tick in every boot. (`main` has such
  ticks too; there the suffix shares them with nothing or follows a chunk.)
- What does differ is M itself. Over the boots on these bases: `main` 8, 8, 9, 11, 11, 11, 12, 13
  (mean 10.4); fence only 10, 12, 12, 12, 12, 12 (11.7); the fix 10, 10, 11, 11, 12, 12, 12, 12
  (11.3); tails-first 10, 12, 12, 12. The ranges overlap, and the fence arms average about one
  more not-restored request per boot of 90. Which hot entry a cold miss's insert evicts from the
  full budget depends on the order of the touches, and the fence moves that order. Each such
  request is one more cold prime, which is where the 1 to 2% of output tok/s goes. At GB10's
  budget every arm restores 90 of 90.

Cold TTFT p50 follows the clock across the four boots (4,586 ms at 1,667 MHz, 4,677 at 1,632,
4,920 at 1,573, 4,872 at 1,540), with the fix at 1,540 MHz ahead of `main` at 1,573.

Gates on this code: `prefix-exact-resend-gate.py` PASS on the RTX 5090 and on GB10 (resend 2/2,
growth 2/2, identical 9/9, tails 4, covered 4/4); `prefix-newest-turn-fits-gate.py` at the
unchanged day-16 shape PASS (identity 12/12, grid 31/31, cohort evictions 4, refused 0);
`prefix-restore-identity-gate.py` PASS (5/5); `seeded-invariance-gate.py` PASS (54 rows).
`rtx5090/confirm/`, `gb10/confirm/`. CPU: `cpu/checks-f1ccab0dc7.log` and `cpu/red-fence/` (the
three red arms and the green tree on this code).

## The fix against main, the round before the review

The cells of this and the following sections ran on 6af34243a2 against `main` d1ed663256.

Hit TTFT in ms over the 90 hit-role requests of a boot; "restored" splits them as whole + grid.
`prime_wait` is the time from admission to the start of the row's own prime call.

### GB10, 4,096 MiB

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| main-235305 | main | 90 (0 + 90) | 320 / 1,210 | - / - | 320 / 1,210 | 18 / 1,011 | 5,203 / 9,571 | 32.8 | - |
| bonly-235821 | b-only | 90 (0 + 90) | 319 / 566 | - / - | 320 / 566 | 12 / 128 | 5,338 / 9,780 | 32.6 | - |
| bonly-011104 | b-only | 90 (0 + 90) | 321 / 564 | - / - | 321 / 564 | 15 / 237 | 5,283 / 9,788 | 32.6 | - |
| main-011621 | main | 90 (0 + 90) | 321 / 1,217 | - / - | 321 / 1,217 | 12 / 926 | 5,197 / 9,645 | 32.7 | - |
| fix-013156 | fix | 90 (78 + 12) | 88 / 218 | 87 / 215 | 207 / 323 | 9 / 68 | 5,028 / 9,543 | 34.3 | 8 / 78 / 22 / 2 |
| fix-013710 | fix | 90 (78 + 12) | 90 / 214 | 89 / 210 | 205 / 213 | 9 / 12 | 4,986 / 9,473 | 34.3 | 8 / 78 / 22 / 2 |

Two boots per arm, one night on one unit; the `main` and b-only boots ran 90 minutes before the
fix boots, in the window whose tails-first and LRU-item boots are in the next section. The fence
alone (b-only) takes the p95 from 1,210 to 565 ms: the
restored hit no longer waits behind a cold chunk (`prime_wait` p95 1,011 to 128 to 237 ms). The
tail takes the p50 from 320 to 88 ms: a whole restore has no prime call. Cold TTFT and output
tok/s do not get worse (5,200 to 5,000 ms, 32.7 to 34.3 tok/s: the hits that skip their prime call
leave the ticks to the cold rows).

### RTX 5090, 2,800 MB

Two windows. The second logged the SM clock and power at 1 Hz.

Window 2, main / fix / b-only / b-only / fix / main:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| main-015856 | main | 79 (0 + 79) | 370 / 4,904 | - / - | 280 / 1,212 | 6 / 948 | 4,648 / 8,224 | 42.9 | - |
| fix-020256 | fix | 80 (7 + 73) | 276 / 4,886 | 155 / 360 | 166 / 509 | 3 / 241 | 4,726 / 8,078 | 42.4 | 18 / 7 / 83 / 17 |
| bonly-020637 | b-only | 78 (0 + 78) | 281 / 7,130 | - / - | 167 / 431 | 2 / 230 | 4,902 / 9,076 | 42.0 | - |
| bonly-021032 | b-only | 80 (0 + 80) | 253 / 4,672 | - / - | 163 / 519 | 2 / 243 | 4,753 / 8,585 | 42.1 | - |
| fix-021418 | fix | 78 (5 + 73) | 281 / 6,989 | 158 / 375 | 250 / 406 | 4 / 132 | 4,667 / 8,551 | 42.2 | 17 / 5 / 86 / 16 |
| main-021813 | main | 78 (0 + 78) | 285 / 6,887 | - / - | 278 / 1,016 | 6 / 810 | 4,663 / 8,412 | 43.3 | - |

Window 1, main / b-only / fix / fix / b-only / main:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| main-011903 | main | 82 (0 + 82) | 291 / 4,596 | - / - | 288 / 1,148 | 7 / 854 | 4,717 / 8,257 | 43.8 | - |
| bonly-012238 | b-only | 78 (0 + 78) | 282 / 7,136 | - / - | 167 / 513 | 3 / 130 | 4,856 / 8,976 | 41.3 | - |
| fix-012639 | fix | 80 (6 + 74) | 271 / 5,168 | 98 / 378 | 183 / 535 | 4 / 142 | 5,187 / 9,306 | 39.1 | 18 / 6 / 84 / 17 |
| fix-013039 | fix | 79 (6 + 73) | 262 / 5,561 | 98 / 270 | 186 / 440 | 3 / 139 | 5,084 / 9,138 | 39.7 | 18 / 6 / 84 / 17 |
| bonly-013440 | b-only | 78 (0 + 78) | 291 / 7,517 | - / - | 169 / 547 | 2 / 138 | 4,957 / 9,308 | 41.4 | - |
| main-013845 | main | 79 (0 + 79) | 415 / 6,919 | - / - | 310 / 1,118 | 6 / 920 | 4,911 / 8,798 | 40.4 | - |

Restored grid hits: 278 to 310 ms p50 and 1,016 to 1,212 ms p95 on `main`; 163 to 169 and 431 to
547 with the fence alone; 166 to 250 and 406 to 535 on the fix. The whole-prompt restores (5 to 7
per boot) take 98 to 158 ms p50. The budget holds nine entries and no free room, so a tail attaches 17 to 18 times and is
skipped 83 to 86 times per boot.

**What the fence costs cold requests.** In window 1 the fix read 6.7% more cold TTFT p50 and 6.4%
less output tok/s than `main`. Window 2 shows why that is not the fence:

| window | boot | hit-role requests not restored | cold-chunk ticks / tokens / s | ms per 1,000 cold tokens | short prime ticks / s | decode s | cold TTFT p50 ms | output tok/s | SM clock MHz / power W, mean |
|---|---|---|---|---|---|---|---|---|---|
| abba-telemetry | main-015856 | 11 | 99 / 140,604 / 110.2 | 784.0 | 65 / 11.0 | 59.7 | 4,648 | 42.9 | 1654 / 110 |
| abba-telemetry | fix-020256 | 10 | 94 / 135,408 / 105.6 | 780.0 | 75 / 11.9 | 59.7 | 4,726 | 42.4 | 1654 / 115 |
| abba-telemetry | bonly-020637 | 12 | 98 / 145,044 / 119.0 | 820.2 | 81 / 13.0 | 60.4 | 4,902 | 42.0 | 1590 / 109 |
| abba-telemetry | bonly-021032 | 10 | 98 / 135,436 / 109.0 | 805.0 | 82 / 12.9 | 60.1 | 4,753 | 42.1 | 1605 / 111 |
| abba-telemetry | fix-021418 | 12 | 98 / 145,044 / 116.2 | 801.4 | 76 / 12.2 | 60.2 | 4,667 | 42.2 | 1612 / 112 |
| abba-telemetry | main-021813 | 12 | 98 / 145,212 / 114.6 | 789.2 | 66 / 11.8 | 59.7 | 4,663 | 43.3 | 1643 / 116 |
| abba | main-011903 | 8 | 88 / 126,052 / 101.9 | 808.5 | 66 / 11.7 | 60.2 | 4,717 | 43.8 | - |
| abba | bonly-012238 | 12 | 98 / 145,044 / 121.9 | 840.2 | 82 / 13.2 | 60.7 | 4,856 | 41.3 | - |
| abba | fix-012639 | 10 | 98 / 135,436 / 119.0 | 878.4 | 77 / 13.1 | 62.6 | 5,187 | 39.1 | - |
| abba | fix-013039 | 11 | 98 / 140,240 / 120.0 | 855.8 | 77 / 12.8 | 61.9 | 5,084 | 39.7 | - |
| abba | bonly-013440 | 12 | 98 / 145,044 / 124.8 | 860.7 | 82 / 13.5 | 61.5 | 4,957 | 41.4 | - |
| abba | main-013845 | 11 | 94 / 140,436 / 118.6 | 844.6 | 69 / 12.5 | 61.1 | 4,911 | 40.4 | - |
| abba | main-235003 | 13 | 103 / 150,044 / 121.4 | 809.0 | 70 / 12.4 | 61.1 | 4,872 | 40.5 | - |
| abba | bonly-235342 | 12 | 98 / 145,044 / 122.1 | 841.9 | 82 / 13.3 | 61.3 | 4,976 | 40.5 | - |
| abba | fix-235742 | 12 | 98 / 145,044 / 127.2 | 876.7 | 79 / 13.4 | 62.8 | 5,544 | 39.5 | - |
| abba | fix-000154 | 10 | 98 / 135,436 / 117.5 | 867.5 | 80 / 13.6 | 62.4 | 5,116 | 39.6 | - |
| abba | bonly-000555 | 12 | 98 / 145,044 / 124.9 | 861.2 | 78 / 13.8 | 62.1 | 5,121 | 40.1 | - |
| abba | main-000956 | 11 | 101 / 140,688 / 117.0 | 831.6 | 66 / 11.4 | 61.0 | 4,872 | 41.5 | - |
| policy-middle | tf-010336 | 12 | 98 / 145,044 / 113.1 | 779.6 | 77 / 12.2 | 59.7 | 4,703 | 43.1 | - |
| policy-middle | fix-010726 | 11 | 99 / 140,268 / 113.4 | 808.2 | 73 / 12.1 | 60.0 | 4,720 | 42.7 | - |
| policy-middle | fix-011057 | 12 | 99 / 145,044 / 112.4 | 775.0 | 80 / 12.1 | 59.8 | 4,641 | 42.6 | - |
| policy-middle | tf-011452 | 12 | 98 / 145,044 / 112.6 | 776.1 | 80 / 12.2 | 59.8 | 4,725 | 43.1 | - |

- The speed of the cold prime itself (the summed prefill time of the ticks that primed at least
  512 tokens, per 1,000 tokens) follows the SM clock and not the arm: 784 and 780 ms at 1,654 MHz,
  789 at 1,643, 801 at 1,612, 805 at 1,605, 820 at 1,590. The clock dips in the middle of a window
  whatever runs there. Window 1 and the tails-first window have no telemetry, and both read the
  slowest cold tokens in their two middle boots (878 and 856 ms with the fix in the middle, 877
  and 868 with the tails-first build in the middle).
- A hit-role request that restores nothing is one more 4,860-token cold prime. That count ran 8
  to 13 per boot on every arm.
- At equal count, a boot's GPU time (cold chunks, short primes and decode) scales with the clock:
  12 not restored: `main` 186.1 s at 1,643 MHz, fix 188.6 s at 1,612, b-only 192.4 s at 1,590
  (186.1 x 1,643 / 1,612 = 189.7; x 1,643 / 1,590 = 192.3). 10 not restored: fix 177.2 s at 1,654,
  b-only 182.0 s at 1,605 (177.2 x 1,654 / 1,605 = 182.6).
- Window 2, fix against `main`: cold TTFT p50 4,697 against 4,656 ms (+0.9%; the two `main` boots
  differ by 0.3%, the two fix boots by 1.3%), output tok/s 42.3 against 43.1 (-1.9%; `main` boots
  0.9% apart). At the same clock (1,654 MHz) the fix reads 4,726 ms and 42.4 tok/s against 4,648
  and 42.9: +1.7% cold TTFT and -1.2% output tok/s at equal clock, inside two `main`-to-`main`
  spreads.
- GB10 restores 90 of 90 on every arm, so the count plays no part there: the fence alone reads
  5,338 and 5,283 ms cold TTFT p50 against 5,203 and 5,197 (+2.1%, one tick per fenced suffix) and
  32.6 against 32.75 tok/s; with the tails 5,028 and 4,986 ms and 34.3 tok/s.
- Scope: two boots per arm and about ten cold requests per boot, one telemetered window here and
  one in the PR's round, 1 Hz telemetry, and 28-token suffixes only (the fence also takes longer
  suffixes, up to the tick budget, which this cell does not send). The policy window's middle
  boots were not slow (808 and 775 ms against 780 and 776 outside), so the dip is not a law of
  the window. What these boots support: the fence's cost to cold requests on this card is not
  resolved below about 2%, and window 1's 6.7% is the clock.

The restored count (77 to 82 of 90 on `main`, 78 to 80 on b-only and the fix) moves with the order
in which requests retire against a full budget, and is not a property of an arm.

## The eviction rule

A tail is 157.9 MB on this checkpoint (the recurrent state; 24 to 46 tokens of KV rows are under
1 MB), against 300 MB for a 4,832-token entry. Three rules were built and measured.

| rule | GB10: hits through a tail / restored, hit p50 / p95 | RTX 5090: through a tail / restored, hit p50 | day-16 newest-turn-fits |
|---|---|---|---|
| LRU-item: a proven tail outlives every entry older than it | 78 / 89, 89 / 215 and 78 / 89, 89 / 216 | 9 / 78, 390 and 8 / 79, 420 (previous base) | REFUSED: the cohort lost an entry |
| tails-first: every tail goes before any entry | 38 / 90, 202 / 565; 38 / 90, 203 / 508; 39 / 90, 202 / 329; 38 / 90, 202 / 564 | 3 / 78, 274 and 3 / 78, 253 | PASS |
| chosen: a proven tail outranks only an oldest entry that never served a hit | 78 / 90, 88 / 218 and 78 / 90, 90 / 214 | 6 / 79, 278 and 5 / 78, 235 | PASS |

The windows the rows come from, in run order.

GB10, 4,096 MiB, LRU-item and tails-first (with the `main` and b-only boots above), then
tails-first / chosen / chosen / tails-first:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| lru-000338 | LRU-item | 89 (78 + 11) | 89 / 215 | 88 / 212 | 202 / 209 | 6 / 15 | 4,989 / 9,534 | 34.4 | 9 / 78 / 21 / 3 |
| fix-000849 | tails-first | 90 (38 + 52) | 202 / 565 | 89 / 208 | 269 / 566 | 12 / 237 | 5,216 / 9,493 | 33.2 | 8 / 38 / 62 / 8 |
| fix-001403 | tails-first | 90 (38 + 52) | 203 / 508 | 88 / 324 | 320 / 563 | 12 / 128 | 5,168 / 9,484 | 33.3 | 8 / 38 / 62 / 8 |
| lru-010547 | LRU-item | 89 (78 + 11) | 89 / 216 | 88 / 210 | 202 / 217 | 6 / 18 | 4,985 / 9,483 | 34.3 | 9 / 78 / 21 / 3 |
| tf-012640 | tails-first | 90 (39 + 51) | 202 / 329 | 88 / 146 | 319 / 564 | 12 / 122 | 5,347 / 9,538 | 33.3 | 8 / 39 / 61 / 8 |
| fix-013156 | fix | 90 (78 + 12) | 88 / 218 | 87 / 215 | 207 / 323 | 9 / 68 | 5,028 / 9,543 | 34.3 | 8 / 78 / 22 / 2 |
| fix-013710 | fix | 90 (78 + 12) | 90 / 214 | 89 / 210 | 205 / 213 | 9 / 12 | 4,986 / 9,473 | 34.3 | 8 / 78 / 22 / 2 |
| tf-014222 | tails-first | 90 (38 + 52) | 202 / 564 | 88 / 134 | 320 / 565 | 12 / 236 | 5,284 / 9,559 | 33.2 | 8 / 38 / 62 / 8 |

RTX 5090, 2,800 MB, tails-first / chosen / chosen / tails-first:

| boot | arm | restored (whole + grid) | hit TTFT p50 / p95, all 90 | whole p50 / p95 | grid p50 / p95 | grid prime_wait p50 / p95 | cold TTFT p50 / p95 | output tok/s | tails attached / hit / skipped / stripped |
|---|---|---|---|---|---|---|---|---|---|
| tf-010336 | tails-first | 78 (3 + 75) | 274 / 6,883 | 31 / 31 | 233 / 433 | 4 / 230 | 4,703 / 8,297 | 43.1 | 24 / 3 / 81 / 23 |
| fix-010726 | fix | 79 (6 + 73) | 278 / 5,336 | 97 / 283 | 276 / 459 | 4 / 243 | 4,720 / 8,597 | 42.7 | 19 / 6 / 83 / 18 |
| fix-011057 | fix | 78 (5 + 73) | 235 / 6,820 | 154 / 379 | 160 / 411 | 3 / 126 | 4,641 / 8,261 | 42.6 | 18 / 5 / 85 / 17 |
| tf-011452 | tails-first | 78 (3 + 75) | 253 / 6,810 | 32 / 146 | 236 / 425 | 4 / 126 | 4,725 / 8,316 | 43.1 | 24 / 3 / 81 / 23 |


- The LRU-item rule lets a tail cost an entry. On the day-16 shape of
  `tools/prefix-newest-turn-fits-gate.py` (four cohort prompts, each sent twice, 1,024 MiB) four
  entries and four proven tails are 1,420 MB, and the oldest entry went before the younger tails:
  `REFUSED: the cohort is not intact` (`first-round/rtx5090/gates/newest-turn-fix2.log`). The
  gate's shape was not changed.
- Tails-first never costs an entry and gives up half of the tail hits on GB10: every entry a cold
  miss publishes strips proven tails, although the entry it would otherwise evict is another
  one-off miss's seed.
- The chosen rule strips a proven tail only when the oldest evictable entry is one a request
  restored since its insert. While the oldest entry is a never-used one, plain LRU evicts it and
  the tails stay. An entry that served a hit is never evicted before a tail, so the day-16 gate
  passes at its original shape, and the entries' own order is plain LRU in every case.
- The policy windows: `gb10/` (cells `lru-*`, `tf-*`, `fix-*`), `rtx5090/policy-middle/`
  (tf / fix / fix / tf), `rtx5090/policy-lru/` (lru / tf / tf / lru on the previous base
  98337756cf, where the `fix-*` labels are the tails-first build).
- `tools/tailsim3.py` is the toy model the first choice was made with; its numbers are not
  evidence.

## The fence: what it takes and its bound

- The fence takes a resumed row (`n_cached > 0`) whose remaining suffix is at least `PRIME_MIN_T`,
  at most the tick budget, with no capture boundary inside, on a model that primes a carried
  suffix. A row on an eager-only model, under the confidence trace or on a GLM plain-prime route
  is not fenced (its suffix does not complete in one call).
- The bound in these boots: the fence stood down for one tick once its consecutive deferred ticks
  added up to the longest undeferred prefill tick the worker had run. The PR's code measures
  against one chunk's time instead (`The PR's code` above).
- Two forms were rejected by measurement, `rtx5090/rejected-lease-filter/`: a fence only for rows
  that still hold their prefix-cache lease, and a bound measured against the last undeferred
  prefill tick. A hit's source lease is released right after the restore, so the filter took the
  hits out of the fence: 21, 21 and 19 requests of a boot waited more than 300 ms for their prime
  call (3 and 4 on b-only, 15 to 22 on `main`), restored grid hits' `prime_wait` p95 995, 1,098
  and 969 ms. Without the filter (the ABBA boots above): 5 and 2 requests, p95 142 and 139 ms. The
  bound's yardstick changed in the same round because a cold prompt's last call is its own
  28-token remainder; its effect was not measured apart from the filter's.

## Exactness

| gate | RTX 5090 | GB10 |
|---|---|---|
| `prefix-exact-resend-gate.py` (new), fix | PASS: resend 2/2, growth 2/2, identical 9/9, tails 4, covered 4/4 | PASS: resend 2/2, growth 2/2, identical 9/9, tails 4, covered 4/4 |
| `prefix-exact-resend-gate.py`, main d1ed663256 | FAIL: resend 0/2, growth 1/2, identical 9/9, tails 0, covered 0/4 | FAIL: resend 0/2, growth 1/2, identical 9/9, tails 0, covered 0/4 |
| `prefix-newest-turn-fits-gate.py --cohort-tokens 1250,1350,1450,1550 --turns 12 --start-tokens 11000 --grow-tokens 150` (day-16 shape, unchanged), fix | PASS: cached 11/11, identity 12/12, grid 31/31, cohort evictions 4, refused 0 | |
| `prefix-restore-identity-gate.py` (memra#602), fix | PASS: identical 5/5, grid 5/5 | |
| `seeded-invariance-gate.py`, fix | PASS: 54 rows, 10 prompts | |
| `prefix-exact-resend-gate.py`, fix, Ornith-1.5-35B-A3B NVFP4 + Q5_K | PASS: resend 2/2, growth 2/2, identical 9/9, tails 4, covered 4/4 (a tail is 66.1 MB) | |

The exact-resend gate serves P1 (11,000 ids, 24 past the grid), P2 (P1 plus 150) and P3 (P1's grid
entry and P1's remainder reversed) cold with the cache off, then with the cache on: P1, P1, P2, P1,
P2, P3, P3, P1, P1. Every re-send restores the whole prompt with no prime call, growth restores the
grid entry with every prime call on the grid, P3 replaces P1's tail on the shared entry and each
tail serves its own prompt only, and all nine texts equal the cold texts. On `main` the re-sends
restore 10,976 tokens and prime 24.

The same gates passed on the tails-first build (`rtx5090/tails-first/gates/`,
`rtx5090/tails-first/inv-fix/`: seeded-invariance PASS, 54 rows).

## Normal cells

`tools/normal_cell.py`, c = 1, 2, 4, 8 closed-loop, streaming, one boot per arm, RTX 5090:

| model | c | aggregate tok/s, main / fix | TTFT p50 ms, main / fix | ITL p50 ms, main / fix | errors |
|---|---|---|---|---|---|
| Qwen3.8-27B | 1 | 72.9 / 73.0 | 171 / 172 | 0.0 / 0.0 | 0 / 0 |
| Qwen3.8-27B | 2 | 91.4 / 88.2 | 177 / 205 | 0.0 / 0.0 | 0 / 0 |
| Qwen3.8-27B | 4 | 119.1 / 119.1 | 617 / 652 | 27.8 / 27.7 | 0 / 0 |
| Qwen3.8-27B | 8 | 146.3 / 147.7 | 1315 / 1397 | 42.6 / 42.2 | 0 / 0 |
| Ornith-1.5-35B-A3B | 1 | 152.0 / 153.0 | 98 / 99 | 5.2 / 5.2 | 0 / 0 |
| Ornith-1.5-35B-A3B | 2 | 239.5 / 238.9 | 153 / 153 | 6.7 / 6.7 | 0 / 0 |
| Ornith-1.5-35B-A3B | 4 | 320.7 / 320.9 | 298 / 339 | 9.2 / 9.2 | 0 / 0 |
| Ornith-1.5-35B-A3B | 8 | 382.3 / 381.4 | 716 / 670 | 14.5 / 14.5 | 0 / 0 |

Single boots, 3 requests per client; the two arms of a model ran back to back. No difference
beyond what one boot resolves (the c = 2 Qwen row is 91.4 against 88.2 tok/s on 6 requests).

## CPU

- `cpu/checks-6af34243a2.log` (the round before the review), `cpu/checks-rebased-1d3be5b0df.log`,
  `cpu/checks-f1ccab0dc7.log` (the PR's code) and `cpu/checks-rebased-340a5e5bfa.log` (the head's
  crates, 1,209 server tests): `cargo fmt --check`, `cargo clippy --all-targets -D warnings`, `cargo test -p
  memra-server` (1,203 passed), the flags census and the numeric-scope census.
- `cpu/red-fence/`: the unit tests green on the tree (`green.log`) and red with the mechanism
  removed, one arm each, with the patch that removed it: `red-fn` (the bound's function answers
  false), `red-wire` (the tick loop does not consult it), `red-rule` (a proven tail goes before
  any entry). `tools/dd-red-fence.sh` runs the arms.
- `cpu/red-witness/`: the sellgate replay's tail-witness check, green on the tree and red with
  the comparison removed (`tools/red-witness.sh`). A whole-prompt `cached_tokens` that the server
  did not count in `prefix_cache_tail_hits` is the off-grid prompt-end entry of memra#602, and the
  replay fails it as `cell_accounting`.

## What the first round found

`first-round/` holds the runs on the first build (d5708a9a14 on 98337756cf's predecessor
ede1b8119f, the LRU-item rule, no bound): the red exact-resend gate on `main` (resend 0/2), the
green one on the fix on both devices, the refused newest-turn-fits gate, and the first cells
(GB10: 78 of 90 through a tail, 88 / 215 ms). The RTX 5090 ran in a slower regime that evening (32.5
output tok/s, cold TTFT 6.0 s), so its numbers are not comparable with the later windows.

## Files

- `rtx5090/abba-telemetry/` (window 2, with one `telemetry-*.csv` per boot) and `rtx5090/abba/`
  (window 1): `cell.jsonl.gz`, `summary.log`, `guard.log`, one `server-*.log.gz` per boot with the
  tick and TTFT traces. Files over 32 KB are gzipped; the tools read either form.
- `rtx5090/confirm/`, `gb10/confirm/`: the PR's code.
- `rtx5090/gates/`, `rtx5090/inv-fix/`, `rtx5090/normal-27b/`, `rtx5090/normal-orn/`: the round
  before the review.
- `rtx5090/policy-middle/`, `rtx5090/policy-lru/`, `rtx5090/tails-first/`,
  `rtx5090/rejected-lease-filter/`: the comparisons above.
- `gb10/gb10-hitcold/`: every GB10 boot in run order; `gb10/gates/`.
- Paths under the lane's scratch directory read `<lane>` in the logs.
- `first-round/` keeps its gate logs, verdicts and cell summaries; its server logs and request
  rows, and the invariance gates' per-request text dumps, are not in the repo.
- `tools/`: `readme-tables.py`, `boot-accounting.py` and `slow-tail.py` (the tables above from
  these files), `hitcold-cell.py` (the cell), `ttft-split.py` (per-class split by request id),
  `cells-table.py` (the tables above), `normal_cell.py`, `gguf_ssm.py` (the state dimensions that
  size a tail), `tailsim3.py`.
