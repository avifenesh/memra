# A grid entry carries its prompt-end tail; a restored hit's suffix primes first (2026-11-11)

**Status:** adopted. Issue memra#1233. Receipts:
[`research/hit-prompt-end-20261111/`](../../research/hit-prompt-end-20261111/).

## Question

Since memra#602 the prompt-end seed publishes at the largest GDN-grid length under an off-grid
prompt end, so a growing conversation restores an on-grid entry and primes its suffix from a grid
start, the cold prime's own split. The price fell on the exact re-send: a 4,860-token prompt
restores 4,832 tokens and primes the 28-token remainder before its first token. On the frozen
mixed hit/cold c = 4 cell that remainder shared its scheduler tick with unrelated cold 1,024-token
prime chunks, because the first-token fence keyed on "fully restored" and no longer fired.

Two things to decide: how an exact re-send gets its whole prompt back without giving up the grid
law, and what the scheduler does with a restored hit that still has a short suffix to prime.

## Chosen

**(a) The prompt-end tail.** The grid entry carries the rest of the state its publishing prime
ended in (`PromptEndTail`, worker.rs): the KV rows past the entry, the recurrent (GDN) state at
the prompt end and the prompt-end logits. A prompt equal to the entry's key plus the tail's tokens
restores both in one launch and decodes its first token from the restored logits. Every longer
prompt restores the grid entry alone, so memra#602's law and its gates are unchanged. A covered hit
(it restored the entry and primed its own remainder) attaches its tail too, in place of another
prompt's tail on that entry.

**(b) The hit-suffix fence.** A resumed row that has not emitted and whose remaining suffix
completes in one prime call (at least `PRIME_MIN_T`, at most the tick budget, no capture boundary
inside) primes ahead of every cold chunk; cold prefill waits that tick
(`hit_suffix_fits_one_call`).

**Memory: a tail never costs an entry that served a hit.** The recurrent state dominates a tail:
157.6 to 158.3 MB at 24 to 46 tokens on Qwen3.8-27B (66.1 MB on Ornith-1.5-35B-A3B), against 483
MB for an 11,000-token entry. A tail attaches only into free budget room and never evicts
anything. An unproven tail (no re-send has gone through it) is evicted before any entry. A proven
tail is evicted before any entry that a request restored since its insert, and outlives only an
oldest entry that never was, which plain LRU evicts anyway (`PrefixCache::tail_victim`). The
entries' own order is plain LRU in every case. Tails never spill to the host tier.

**The fence is bounded.** It stands down for one tick once its consecutive deferred ticks add up
to the time of one prefill chunk at the measured rate (`hit_suffix_fence_yields`,
`hit_suffix_fence_allowance`), while the deadline-aware prefill order is on, and on a tick where a
whole-prompt hit waits for its first decode (that hit then decodes before any suffix call). The
bound is proven by its unit test; no measured cell queues enough hits to reach it.

## Rejected

- **Proven tails as LRU items beside the entries** (a proven tail outlives every entry older than
  it). Built first and measured: 78 of 90 hits through a tail on GB10, 89 of 90 restored. It cost
  a growing conversation's cohort an entry: the day-16 shape of
  `tools/prefix-newest-turn-fits-gate.py` (four cohort prompts at 1,024 MiB) refused with `the
  cohort is not intact`, because four entries plus four proven tails (1,420 MB) do not fit and the
  oldest entry went before the younger tails. An entry lost to a tail is not acceptable. The
  gate's shape was not changed.
- **Strict tails-first** (every tail before any entry). Built second and measured. It never costs
  an entry and passes the day-16 gate, and it gives up half of the tail hits on GB10: 38 of 90,
  hit TTFT p50 / p95 202 / 565 ms against 88 / 218 ms for the chosen rule, because every entry a
  cold miss publishes strips proven tails although the entry it would otherwise evict is another
  one-off miss's seed. On the RTX 5090 at a full 2,800 MB budget it keeps 3 of 90 where the chosen
  rule keeps 5 to 6, with the same restored count, hit TTFT, cold TTFT and output tok/s.
- **Publishing the seed at the off-grid prompt end again.** That is the memra#602 defect: a
  restore at an off-grid entry end followed by an off-grid suffix prime is a second numeric
  program for a growing prompt.
- **A fence without a bound.** Hits that keep arriving could hold cold prefill back without end.
- **A bound measured against the last undeferred prefill tick, or against the longest one.** A
  cold prompt's last call is its own off-grid remainder (28 tokens), so against the last tick one
  fenced tick uses the whole allowance. The longest tick the worker ever ran is set by a
  solo-widened prime (3.9 s on the RTX 5090, 3.1 s on GB10 in the banked boots), which allows 25
  to 30 fenced ticks in a row: a bound in name only.
- **A fence only for rows that still hold their prefix-cache lease.** A hit's source lease is
  released after the restore, before its suffix primes, so the filter took the hits out of the
  fence: 19 to 21 requests of a boot waited more than 300 ms for their prime call, 3 to 4 with the
  fence alone and 2 to 5 after the filter was removed. That build also carried the last-tick
  bound; the two were not measured apart.
- **Suffix calls ahead of a whole-prompt hit.** The first build primed every fenced suffix on a
  tick a whole-prompt hit waited in, and 11 of 17 such hits on the RTX 5090 waited 126 to 278 ms
  behind one or two of them. The fence now stands down on that tick.
- **Batching concurrent hit suffixes into one prime call.** Not built here; filed as its own
  issue.

## Measured

Frozen mixed90 c = 4 shape (4,860-token prompts, 60 completion tokens, nine hits to one miss, four
closed-loop clients, 20 requests per repetition, five repetitions per boot, 90 hit-role requests),
Qwen3.8-27B NVFP4 + Q5_K, plain path. Full tables, per-class splits and tick traces are in the
receipts.

| device, budget | arm | hits through a tail / restored, of 90 | hit TTFT p50 / p95, all 90 | restored grid hits p50 / p95 | cold TTFT p50 | output tok/s |
|---|---|---|---|---|---|---|
| GB10, 4,096 MiB | main | 0 / 90, 0 / 90 | 320 / 1,210, 321 / 1,217 ms | 320 / 1,210, 321 / 1,217 ms | 5,203, 5,197 ms | 32.8, 32.7 |
| GB10, 4,096 MiB | fence only | 0 / 90, 0 / 90 | 319 / 566, 321 / 564 | 320 / 566, 321 / 564 | 5,338, 5,283 | 32.6, 32.6 |
| GB10, 4,096 MiB | fix | 78 / 90, 78 / 90 | 88 / 218, 90 / 214 | 207 / 323, 205 / 213 | 5,028, 4,986 | 34.3, 34.3 |
| RTX 5090 Laptop, 2,800 MB | main | 0 / 82, 0 / 79 | 291 / 4,596, 415 / 6,919 | 288 / 1,148, 310 / 1,118 | 4,717, 4,911 | 43.8, 40.4 |
| RTX 5090 Laptop, 2,800 MB | fence only | 0 / 78, 0 / 78 | 282 / 7,136, 291 / 7,517 | 167 / 513, 169 / 547 | 4,856, 4,957 | 41.3, 41.4 |
| RTX 5090 Laptop, 2,800 MB | fix | 6 / 80, 6 / 79 | 271 / 5,168, 262 / 5,561 | 183 / 535, 186 / 440 | 5,187, 5,084 | 39.1, 39.7 |

These rows are the round before the PR's review. The review's changes (a whole-prompt hit decodes
before the fenced suffixes; the allowance of one chunk) were measured again, main / fix / fix /
main per device: GB10 main 322 / 1,257 and 322 / 1,212 ms, fix 88 / 218 and 89 / 256 ms, 78 of 90
through a tail, 90 of 90 restored, whole-prompt hits' p95 101 to 106 ms (210 to 215 before); RTX
5090 restored grid hits 391 / 1,216 and 293 / 1,251 ms on `main`, 175 / 424 and 158 / 410 on the
fix, whole-prompt hits 31 ms p50 (98 to 158 before), 7 of 90.

Two boots per arm. On the RTX 5090 the all-90 p95 is the cold prime of the hit-role requests whose
entry the full budget had evicted (8 to 13 per boot on every arm); the restored hits are the
columns to read there. In that window the fix read 6.7% more cold TTFT and 6.4% less output tok/s
than `main`. A second window with 1 Hz telemetry (main / fix / fence only / fence only / fix /
main) showed the cause: the speed of the cold prime follows the SM clock, which dips in the middle
of a window whatever arm runs there (784 ms per 1,000 cold tokens at 1,654 MHz, 820 at 1,590), and
the fix had run in the middle. In the second window the fix reads +0.9% cold TTFT p50 (the two
`main` boots differ by 0.3%) and -1.9% output tok/s (`main` boots 0.9% apart). On GB10, where
every arm restores 90 of 90, the fence alone reads +2.1% cold TTFT p50 and the fix reads less
than `main`.

## Exactness

`tools/prefix-exact-resend-gate.py` (new): every text equals the cold text of its prompt, the
re-sends restore the whole prompt with no prime call, growth restores the grid entry with every
prime call on the grid, and a tail attached by a covered hit serves its own prompt and no other.
PASS on the fix and FAIL on `main` on both devices; PASS on Ornith-1.5-35B-A3B on the RTX 5090. memra#602's gates
(`prefix-restore-identity-gate.py`, `prefix-newest-turn-fits-gate.py` at the unchanged day-16
shape) and the seeded-invariance gate pass on the fix.

## Limits

- Tails help where the budget has room beside the entries. A tail attaches only into free room,
  and at a budget the workload fills with entries there is none (84 of 102 attaches skipped per
  boot on the RTX 5090); the fence is then most of the gain.
- A tail attaches to plain entries only (no draft plane, no latent planes, no TP shards), so an
  exact re-send on a speculative route still primes its remainder.
- On the RTX 5090's full budget the fence arms average about one more hit-role request per boot
  of 90 that restores nothing (11.3 to 11.7 against 10.4 on `main`, ranges overlapping), because
  the fence moves the order of touches a full LRU evicts by. Each is one more cold prime. The
  all-90 hit TTFT p95 there (4.5 to 5.2 s with up to 10 such requests, 6.8 to 7.6 s with 12, on any
  arm) is the rank of those requests, not a wait behind the fence: cold requests take the same
  multiple of an unshared prime on every arm, and the fence's longest run is two ticks.
- The fence costs cold requests a tick per fenced suffix: +2.1% cold TTFT p50 on GB10 with the
  fence alone, 1 to 2% of the cell's output tok/s on the RTX 5090 at a full budget. The bound
  limits it, it does not remove it.
