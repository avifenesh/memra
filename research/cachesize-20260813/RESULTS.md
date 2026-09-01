# cx-cachesize — one-card cache-hit capacity

Date: 2026-08-13  
Runtime: `v0.81.2` (`18885ec479d897a3e8c42b0d408a71fa3edaa708`)  
Rig: box1, physical GPU0, one RTX PRO 6000 Blackwell Server Edition (`97,887 MiB`)

## Verdict

No production capacity recommendation is emitted from the partial campaign. The v0.81.2 runner
failed closed at Q27 repetition 2 / 16,384 MiB after three restored hits stopped at 11/60 tokens.
That repeats the earlier Q27 8,192 MiB exactness class outside its one planned exclusion, so the
fixed 59-valid-boot reduction contract is no longer satisfied.

## What was measured

The campaign reused the frozen sell-gate request shape: the exact 4,860-token prompt, 60 output
tokens, temperature zero, and a 9:1 intended working-prefix/cold-churn request mix. The new
independent variable was a fixed `N=96` tenant-isolated working set, matching the requested
`MEMRA_MAX_SESSIONS=96`. Every logical prefix carries the same frozen prompt token identity but
uses its own cache namespace. Working keys are visited by deterministic shuffled permutations
without replacement, continue across concurrency cells, and reshuffle only after all 96 keys
have been visited. A cell never contains the same working key twice.

The six budgets were 1,024 / 4,096 / 8,192 / 16,384 / 32,768 / 49,152 MiB. The planned design gives
each model/budget/concurrency cell five interleaved repetitions, except Q27 8,192 MiB at N=4 after
its explicit repetition-1 exactness exclusion. Within a model/repetition, every budget gets
the identical role order, concurrency order, and working-key permutation. Odd repetitions ran Q27 then Q35 and budgets
ascending; even repetitions ran Q35 then Q27 and budgets descending. The recovered valid boots span
two explicitly separate lock segments; the resumed segment held `/tmp/memra-gpu.lock` plus
`/tmp/memra-gpu-1.lock` continuously until its fail-closed stop. Both segments used physical GPU0,
with GPU1 idle, one model resident at a time, no clock changes, and no artificial cooldown.

The runtime exposes `prefix_cache_bytes` as its exact device-resident cache accounting. Source
inspection confirms that it is the cache's `total_bytes`, where each entry's bytes comprise the
prefix KV snapshot plus recurrent state. Each entry probe also reconciled one miss, one insert,
one retained entry, and zero evictions, admission defers, or OOM parks.

## Actual snapshot bytes

| Model | Prefix tokens | Device bytes | Device MiB |
|---|---:|---:|---:|
| Q27 | 4,096 | 278,528,000 | 265.625 |
| Q27 | 4,860 | 301,215,744 | 287.2617 |
| Q27 | 8,192 | 400,162,816 | 381.625 |
| Q35 | 4,096 | 103,874,560 | 99.0625 |
| Q35 | 4,860 | 110,964,480 | 105.8240 |
| Q35 | 8,192 | 141,885,440 | 135.3125 |

The `~343 MB/entry` text in `docs/FLAGS.md` is explicitly a Step-model 4k example. It is not
verified or reusable for either model here: at 4k, Q27 measured 278.528 decimal MB and Q35
measured 103.875 decimal MB. For this lane's sold 4,860-token prompt, the sizing values are
301,215,744 bytes for Q27 and 110,964,480 bytes for Q35.

The steering cross-check of 1,763,997,696 bytes across six Q27 entries (293,999,616 bytes per
entry) and the corresponding 108,709,440-byte Q35 average did not contain six sold-shape entries.
The serial exactness gate retained three full 4,860-token entries and three 4,374-token partial
entries. The measured geometry is exactly linear: Q27 is 156,893,184 fixed bytes plus 29,696 bytes
per token, and Q35 is 65,863,680 fixed bytes plus 9,280 bytes per token. Those mixed-shape averages
therefore reconcile exactly. This lane's isolated probes each retained one full entry and repeated
twice; the older 14-entry full-prompt requalification also divides exactly to 301,215,744 bytes for
Q27. The full-entry values in the table are the authoritative sold-shape sizing inputs.

The corresponding floor capacities are exact byte arithmetic against the binary-MiB runtime
budget. “Paired” means one Q27 and one Q35 snapshot for the same logical session:

| Budget MiB | Q27-only entries | Q35-only entries | Paired sessions |
|---:|---:|---:|---:|
| 1,024 | 3 | 9 | 2 |
| 4,096 | 14 | 38 | 10 |
| 8,192 | 28 | 77 | 20 |
| 16,384 | 57 | 154 | 41 |
| 32,768 | 114 | 309 | 83 |
| 49,152 | 171 | 464 | 125 |

This is one shared per-worker cache, not one allowance per loaded model. Model and namespace are
part of each pool key, while `PrefixCache.total_bytes` and its LRU budget are global across the
worker's resident models. The paired-session divisor above therefore charges one Q27 plus one Q35
entry against the same configured budget.

Exactly 96 sold-shape entries consume 27,577.125 MiB for Q27 or 10,159.102 MiB for Q35. Keeping
both models' prefix for every one of the 96 logical sessions would consume 37,736.227 MiB.

## Capacity curve

Not reduced: 19 valid boots are sealed, but the runtime exactness failure stopped the campaign
before the prescribed grid completed.

## Active concurrency at the sold hit-latency envelope

Not reduced from an incomplete, exactness-invalid campaign.

## Does a large budget cost anything?

Not reduced from an incomplete, exactness-invalid campaign.

## Boot-warning proposal

Add a boot-time warning, not a refusal, after all models load. For each resident model, derive the
exact prefix-entry bytes at the configured `MEMRA_CTX` from the same snapshot allocation geometry;
compare the largest result with the one worker-global `MEMRA_PREFIX_CACHE_MB` allowance. If even
one such entry cannot fit, name the model, token count, required bytes, configured bytes, and the
current consequence: full-length inserts for that model will be skipped. Keep the existing
per-insert skip receipt as the request-shape backstop.

At the requested `MEMRA_CTX=8192`, the measured warning thresholds are 400,162,816 bytes
(381.625 MiB) for Q27 and 141,885,440 bytes (135.3125 MiB) for Q35. The naked 256 MiB default would
therefore warn for Q27; the current production 4,096 MiB override would not. Using configured
context makes the check conservative and available at boot without inventing a product-specific
sold-prompt environment variable.

## Production recommendation

Withheld. Resuming with the sibling lane's one-program fix (or the equivalent
`MEMRA_SERVE_B1FAST=0` control on the old binary) would change the numerical and performance
program relative to the 19 valid v0.81.2 boots. Mixing those regimes would violate the frozen
runtime contract; restarting the whole grid on the repaired runtime requires an explicit owner
decision because it necessarily repeats completed measurements.

## Evidence and exclusions

All stdout and stderr were teed to a raw log before reduction. The scored reducer requires 59 valid
budget boots plus the one explicit Q27 repetition-1 / 8,192 MiB exclusion, exact response usage,
exact `/metrics` reconciliation, frozen artifact hashes, the prescribed alternating boot order,
empty server-failure scans, and valid raw manifests.

Six pre-score attempts are retained but excluded:

- Attempt 1 stopped after an unexpected independently locked GPU1 workload appeared on the
  shared `PIX` topology.
- Attempt 2 found a workload-generator cycle-boundary duplicate; the server correctly reported
  two 1,024-token same-window dedup credits, and the harness failed closed.
- Attempt 3 found that batched working-set seeding did not preserve the frozen sequential-prime
  numerical class. One HTTP 200 response stopped at 11/60 tokens; captured logs contain no CUDA,
  OOM, fatal, or error line, so no runtime cause is assigned. A sequential-seed diagnostic then
  passed 80/80 exact completions across all Q27 widths at the same 16 GiB arm.
- Attempt 4 stopped during source preflight before acquiring the lock because its launcher
  pre-created the harness's fail-closed output directory.
- Attempt 5's coarse c=4/8 grid bracketed the sold-latency maximum but could not identify it.
  After seven complete passing boots, the owned in-progress sweep was stopped cleanly and the
  final campaign restarted with c=5/6/7 included for both models.
- Attempt 6 found that the otherwise equivalent budget arms used different shuffled access
  orders. It was stopped before one expanded-grid boot completed; the final protocol pairs the
  exact role/key/concurrency trace across all six budgets within each model/repetition.

No excluded request contributes to a scored statistic.

The resumed segment later stopped at Q27 repetition 2 / 16,384 MiB. Prefixes 49 (c=4), 95
(c=16), and 88 (c=6) were full 4,860-token hits but returned HTTP 200 with
`finish_reason=stop`, 11 completion tokens, and the same SHA-256
`ddbbcf35ae93821874be64e15f63feecb2471bbe6ea0c23b63fa00b1e98a9b73` as the earlier excluded
key 87. Their paired positions at 32,768 and 49,152 MiB completed 60 tokens with the same normal
hash. Counters reconcile, defers/OOM parks are zero, all tenant guards pass, and the server-failure
scan is empty. This failed boot is retained as correctness evidence, not silently added to the
exclusion set or replaced with a passing rerun.
