# The admission reclaim spares the restorable tail (2026-11-07)

Verdict: on the RTX 5090 Laptop the admission reclaim now keeps the drafter-tail entry the restorable-hit preference takes. In the r1/r2/r3 receipt shape r3 is served by a DFlash restore instead of plain. On the landing head (5564612bdb, with memra#1083, against main 9805016bb1): total latency 2.50 s -> 1.78 s greedy (-29%) and 2.48 s -> 2.05 s sampled (-17%), TTFT 0.23 s -> 0.68 s (it primes a 569-token suffix instead of 57), median of 3 interleaved runs per arm; on the first head (ed28931123 against main a5a84b3373) 4.04 -> 3.04 s greedy and 4.04 -> 3.54 s sampled. Greedy and sampled r3 are byte-equal to the cache-off reference.

## Cause

Receipt: `before-receipt/` (the 2026-11-07 re-validation on the RTX 5090 Laptop, fix-stack tip
a0a6816ece). Qwen3.8-27B NVFP4 + Q5_K GGUF with the DFlash2 drafter, `MEMRA_MTP_SKIP=1`, context
8,192, one session, prefix cache 4,000 MB, no host tier, the three restore doors forced to 1 (they
default off on the 5090).

1. r1 = P+X, 1 token, greedy with presence penalty: served plain, publishes a plain entry
   (`insert (lcp-split): 3104 tokens, 249.1MB`).
2. r2 = P, greedy: cold-primes DFlash, the boundary capture publishes a tail entry
   (`insert (dspark-boundary): 2592 tokens, 318.1MB`).
3. r3 = P+X plus a question, 96 decode tokens (3,161 prompt tokens). It does not fit
   (`request cost ... = 1862MB`), so admission runs reclaim-on-defer:
   `reclaim spared the prompt's prefix entry (3104 tokens)`, then
   `evicted 1 prefix entries ...; effective free 3509MB -> 3827MB`. The evicted entry is the
   tail. The worker logs `cache-preferred: prompt=3161 decode_budget=96 lcp=3104` and serves the
   3,104-token hit plain.

The reclaim leased only `PrefixCache::lookup`, the longest whole-entry match. The restorable-hit
preference (`dspark_prefer_restorable_hit`) runs later, in `admit`, and needs the shorter tail
entry to still exist. Under device-memory pressure it was already gone.

## Rule

The reclaim spares the entry the preference would take, decided before the flush with the same
function on the same inputs, and keeps it when memory can hold it.

- `PrefixCache::pin_reclaim_spare` leases the longest match as before. It also leases a shorter
  entry, which becomes the one the request keeps, when all of these hold: the longest match cannot
  re-arm the drafter, the shorter one can (`dspark_entry_restorable`, the routing predicate), no
  host-tier entry outranks the longest match (`admit` would promote it and skip the preference),
  and `dspark_prefer_restorable_hit(longest, restorable, budget,
  dspark_prefers_cold_over_prefix(inputs))` holds. `inputs` come from `admission_dflash_inputs`,
  which the #361 retained-prefix plan reads too.
- The flush is unchanged. If the arrival is still short with both entries leased,
  `longer_droppable` says whether the longer entry may go (both solely leased, the host check
  passing, a single shortfall reading), and the site climbs the parked-session ladder
  (`reclaim_climb`, oldest session first, re-reading the unbooked shortfall) until the longer
  entry's bytes cover what remains, then drops it. If the ladder alone makes room, both entries
  stay. If the ladder runs out, or the longer entry may not go, the restorable entry is dropped and
  the longer one stays: the outcome before the change. Every fallback ends with the longer entry
  resident, and admission is never below main.
- The trade: dormant parked sessions are evicted to keep the tail, where main kept them; on the
  ladder-out fallback at most the tail's extra bytes over the longer entry's in parked sessions.
- When the preference does not hold, only the longest match is spared and the path is main's.

No new constant and no new flag. The threshold is the cold-prime ratio
(`DSPARK_COLD_DECODE_RATIO_DIV`, 1/8): the restorable entry is preferred when the decode budget
repays the extra suffix prefill at that ratio. On the receipt's shape that is 96 >= 512 / 8.

Rejected, in review order (the PR's six review rounds are on memra#1082):

- Sparing only the restorable entry: drops a longer hit other requests can use when memory holds
  both.
- A second full flush pass to drop the longer entry: it can demote the longer entry to the host
  tier, which `admit` then promotes back over the restorable hit.
- Choosing by bytes against the shortfall (rounds 2 to 4): measured on the RTX 5090 at
  48e8b3b414 (`served-bytes-rule-48e8/`), r3 was short by 259 MB with the plain entry at 249 MB and
  the tail at 318 MB, so the rule dropped the tail and r3 served plain exactly as before
  (`fix-dflash-restore FAIL`), against the owner's rule to keep the faster entry when memory holds
  one.
- Dropping the longer entry before the ladder, with the tail as a last resort (round 5): when the
  ladder and the plan cannot make up the rest, the prompt keeps neither entry where main kept one.

## Unit fail-before / pass-after (`unit/`)

`admission_reclaim_spares_the_restorable_tail_the_preference_takes` builds the receipt's three
entries (plain 3,104 tokens and 249 bytes, tail 2,592 tokens and 318 bytes, an unrelated one) and
checks the spare, the flush, `longer_droppable` against leases, the host flag and the reading,
`reclaim_climb` on the receipt's shortfall (no parked session, 5, 20, 3 x 5 and 1,000 MB), the two
`drop_one` outcomes, the no-preference paths, and the reclaim site's wiring (anchored on
production source).

- Red: 2cbf7eb4ec with `pin_reclaim_spare`'s body replaced by main's choice (the longest match
  alone, `red-seam.patch`): fails, `left: (3104, 249)`, `right: (2592, 318)` (`red-test.log`).
- Green: a10c650e06 passes (`green-test.log`); memra-server lib 1,167 passed, 0 failed, 28 ignored
  (`green-lib.log`); `cargo clippy -p memra-server --all-targets -- -D warnings` clean. The head
  was rebased unchanged onto main a5a84b3373 as ed28931123, and onto main 9805016bb1 as 7bfc72d250.

## Served cell on the landing head (RTX 5090 Laptop, `served-5564/`)

The same cell on the stacked head that lands: base = main 9805016bb1, fix = 5564612bdb (this
lane's 7bfc72d250 plus memra#1083). memra#1083 folds the three restore doors into one switch and
the boot refuses the retired names, so the fix binary runs with `MEMRA_DSPARK_PREFIX_RESTORE=1`
alone and the base with the three doors (`CELL_DOORS=base:three,fix:one`, the script's default);
the window's first try set the three doors on the fix and stopped at the env audit, so it was
rerun. GPU 72 C at start, 81 C at the end.

| r3 | base TTFT | fix TTFT | base total | fix total | cache-off ref (TTFT / total) |
|---|---:|---:|---:|---:|---:|
| greedy | 0.234 s | 0.681 s | 2.504 s | 1.784 s | 2.493 / 3.625 s |
| sampled (T=1, seed 1234) | 0.231 s | 0.644 s | 2.483 s | 2.051 s | 2.507 / 3.893 s |

Medians of 3 (`summary.json`, `runs.jsonl`); all 13 cells PASS (`cells.txt`). The logs show the
same branches as below: the base evicts the tail and serves the 3,104-token hit plain; the fix
keeps both entries (1 parked session covered the shortfall), prefers the restorable hit and
restores 2,592 tokens plus the draft tail. Greedy r3 is sha 8e25a371633a6ffa on base, fix and
reference; sampled r3 on the fix equals the reference (5f8db50ab11044f4).

## Served cell on the first head (RTX 5090 Laptop, `served-ed28/`)

`reclaim_restorable_cell.py` (with `CELL_DOORS=base:three,fix:three`, before the switch), binary
pair main a5a84b3373 (base) and ed28931123 (fix, the PR head rebased on it), one fresh boot per run, order base/fix/fix/base/base/fix per mode, plus one
cache-off reference boot per mode on the fix binary. Qwen3.8-27B NVFP4 + Q5_K GGUF + DFlash2,
`MEMRA_MTP_SKIP=1`, FR-Spec trim, context 8,192, one session, prefix cache 4,000 MB, no host tier,
the three restore doors forced to 1. GPU 58 C at start, 73 C at the end (`gpu-before.csv`,
`gpu-after.csv`). r3 streamed: TTFT = first text chunk, total = end of stream, 96 completion tokens
in every run.

| r3 | base TTFT | fix TTFT | base total | fix total | cache-off ref (TTFT / total) |
|---|---:|---:|---:|---:|---:|
| greedy | 0.453 s | 1.153 s | 4.037 s | 3.037 s | 4.009 / 5.892 s |
| sampled (T=1, seed 1234) | 0.457 s | 1.162 s | 4.042 s | 3.541 s | 4.038 / 6.416 s |

Medians of 3; the runs per arm are within 0.02 s of each other (`summary.json`, `runs.jsonl`).

What the logs show (`cells.txt`, all 13 cells PASS):

- base: `reclaim spared the prompt's prefix entry (3104 tokens)`, `evicted 1 prefix entries`,
  `cache-preferred`, plain.
- fix: `reclaim spared the prompt's restorable entry (2592 tokens, draft tail) and its longer
  non-restorable entry (3104 tokens)`, then `reclaim kept the prompt's restorable and longer
  entries: 1 parked sessions covered the shortfall` (`evicted 0 prefix entries + 1 plain`), then
  `restorable-preferred: ... plain hit 3104 -> restorable hit 2592` and `DSPARK restore: 2592 of
  3161 prompt tokens + draft tail (569 suffix tokens to prime)`.
- bytes: greedy r3 sha 8e25a371633a6ffa on base, fix and the cache-off reference; sampled r3 on
  the fix equals the reference (5f8db50ab11044f4), while the base serves the plain program
  (1a48e22e76ab09fa), as expected for a plain-vs-DFlash sampled pair.

TTFT rises because the restore primes 569 suffix tokens where the plain hit primed 57; the DFlash
decode repays it inside the 96-token budget, which is the trade the preference's ratio prices.
