# Recurrent prefix restores obey the prime-grid law; the DFlash restore doors are one switch (2026-11-07)

Verdict: on the RTX 5090 Laptop a resumed chat turn left the cold text in the DEFAULT configuration (restore switch off), not only with the single door: an off-grid plain restore of the DFlash prompt-end entry. Cutting the cold DFlash capture on the prime grid and refusing an off-grid recurrent restore at the hit site makes every warm turn byte-equal to its cold twin, with the switch on, off and at the device default; the grid law alone already fixes the reported case. Measured on the landing head 5564612bdb against main 9805016bb1: base red with the single switch and twice at the device default (turn 3 383 vs 371 characters), fix green at 1, 0 and the device default. Cost: the first turn's cold DFlash prime splits at the grid, +0.15 s on a 2,992-token prompt (3.54 -> 3.69 s, 2 runs per arm).

## Finding and root cause

Reported on the RTX 5090 Laptop (Qwen3.8-27B NVFP4 + Q5_K GGUF with the DFlash2 drafter, context
8,192, two sessions): with only `MEMRA_DSPARK_PREFIX_RESTORE=1`, the third turn of a three-turn chat
left the cold text (383 vs 371 characters on main 65d5887596, 414 vs 371 on the fix-stack tip; the
receipt's `run.txt` counts 384 / 415 / 372 because its length includes the reasoning/content
separator); with all three restore doors set it was byte-identical. The cell is
`dspark-pair-cell.sh` (copied from `research/dspark-partial-restore-20261104/` on the fix stack,
commit 765ed89e3c; its verdict moved to `grid_pair_cells.py`, which judges bytes and reports
restores separately). The before-receipt cells keep the server log, the restore lines and the
binary's sha only, not the per-turn JSON.

The warm boot's log (main 65d5887596,
`before-receipt/dspark-cell-1door-main/warm-restore-lines.txt` and `warm-server.log`):

1. Turn 1, 2,992 prompt tokens: cold DFlash prime. With capture off the tail entry sits at the
   prompt end, `insert (dspark-boundary): 2992 tokens`.
2. Turn 2, 3,109 tokens: `cache-preferred ... lcp=2992`, `hit: 2992 of 3109 prompt tokens`. The
   hit is served plain and the suffix primes from 2,992, which is off the 32-token GDN prime grid
   (2,992 = 93.5 x 32). Turn 2 also publishes `insert (seed): 3072 tokens` and parks with a
   checkpoint at 3,072, both computed by that prime.
3. Turn 3, 3,144 tokens: `plain-affinity: rewound to 3072 of 3144 prompt tokens`, so it resumes
   rows the off-grid prime wrote, and its greedy text leaves the cold one.

By the prime-grid law (docs/SERVING.md, "The THIRD axis"), a prime call that starts off the grid
materializes recurrent state the monolithic prime never computes. The plain seed and the spec
captures obey that law at the capture since memra#602. The DFlash prompt-end capture was the one
site left, and the hit site did not refuse an off-grid recurrent entry.

## Change

- The capture: a cold DFlash prime with no render-stable cut (the switch off, or no armable
  boundary) cuts its entry at the seed's grid boundary (`dspark_cold_capture`,
  `seed_capture_boundary`); the prompt end stays the cut only when it is on the grid or the
  prompt is too short for a grid entry. The exact re-send dedupe looks for that key (the served
  cells do not send an exact re-send, so they do not exercise the dedupe; the wiring test does).
- The hit site: `prefix_restore_exact` (a recurrent entry restores only as a whole cover or as a
  strict prefix ending on the 32-token grid with at least `PRIME_MIN_T` suffix tokens) in
  `PrefixCache::lookup`, `lookup_where`, `deepest_covering`, `HostPrefixCache::lookup`,
  `host_demoting_hit`, `admission_restore_rows` and `partial_prefix_decision`. A refused longer
  cover prints `[prefix-cache] restore grid law: ...` and counts
  `prefix_cache_restore_grid_refusals` (`/metrics`, operator-only).
- One switch: `MEMRA_DSPARK_PREFIX_RESTORE` arms the whole-entry restore, the strict-prefix restore
  and the boundary capture; `MEMRA_DSPARK_PARTIAL_RESTORE` and `MEMRA_DSPARK_BOUNDARY_CAPTURE` are
  no longer read (docs/FLAGS.md "Removed doors, 2026-11-07").

## Served pair cells on the landing head (RTX 5090 Laptop, `cells-5564612bdb/`)

`grid_pair_cells.py` runs `dspark-pair-cell.sh` per arm: one warm boot serves turns 1, 2, 3 in
order, then one cold boot each for turns 2 and 3 (prompts built from the warm answers), Qwen3.8-27B
NVFP4 + Q5_K GGUF + DFlash2, context 8,192, two sessions, prefix cache at its derived 800 MB,
greedy, 96 tokens, thinking off. Binaries: base = main 9805016bb1, final = 5564612bdb (memra#1082's
7bfc72d250 plus this lane, on that main). Order as run: base-1switch, base-default, final-on,
final-off, final-default, base-default2. Character counts exclude the reasoning/content separator.

| arm | warm env | turn 1 sha | turn 3 warm / cold chars | bytes | plain hits | DSPARK restores |
|---|---|---|---|---|---|---|
| base-1switch | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 523ca36af0 | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| base-default | device default (OFF) | 523ca36af0 | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| base-default2 | device default (OFF), repeat | 523ca36af0 | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| final-on | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 523ca36af0 | 371 / 371 | pass | 2976, 3072 | 2976 of 3109, 3072 of 3144 |
| final-off | `MEMRA_DSPARK_PREFIX_RESTORE=0` | 523ca36af0 | 371 / 371 | pass | 2976 of 3109 | none |
| final-default | device default (OFF) | 523ca36af0 | 371 / 371 | pass | 2976 of 3109 | none |

Turn 2 is equal in every arm (4 characters). Turn 1's greedy text is byte-identical across all six
boots: the grid cut keeps turn 1's bytes. Its split prime yields once and its first speculative
burst is one round (`[prime-yield] count=1`, `burst=1/7`), because the yield check sees the
session's own pending prime; base-3doors below, which splits at its boundary capture on the base,
shows the same, so this is not new to the change. The cells send no exact re-send, so they do not
exercise the re-send dedupe; the wiring test covers it.

Timing, client wall time per request, switch OFF, order A B B A (base-default and base-default2 at
the device default, final-off with `=0`, final-default at the device default), one run each:

| turn | base | fix |
|---|---:|---:|
| 1 (cold DFlash prime, 2,992 tokens) | 3.545 / 3.533 s | 3.754 / 3.618 s |
| 2 (hit, 3,109 tokens) | 0.315 / 0.329 s | 0.330 / 0.321 s |
| 3 (resume, 3,144 tokens) | 1.934 / 1.985 s | 1.973 / 1.940 s |

Turn 1 pays 0.15 s on average for the two-call prime at the grid cut; turns 2 and 3 are flat, and
turn 3 is now correct. With the switch on, turn 3 restores DFlash and takes 1.64 s.

GPU unit tests at 5564612bdb (`cells-5564612bdb/gpu-unit.log`, the command and test binary in
`gpu-unit.cmd.txt`), the ignored set without the multi-device and fixture ones, with only
`MEMRA_DSPARK_PREFIX_RESTORE=1` set: 24 of 24 pass, `the_prefix_lookup_skips_an_off_grid_recurrent_entry`
and `dflash_retained_gpu_plan_fault_matrix` included, so the one switch arms the restore plan on GPU.

## Root-cause cells (RTX 5090 Laptop, `cells-6c957b5f2f/`)

The same cell on the earlier heads, with two arms that isolate the mechanism. Binaries: base =
48e8b3b414 (an earlier memra#1082 head, no grid law), grid = b3a39cdd5f (the hit-site law alone,
doors still separate), final = 6c957b5f2f. This run's `verdicts.jsonl` counts the separator (384 /
372 / 5 below are 383 / 371 / 4).

| arm | warm env | turn 3 warm / cold chars | bytes | plain hits | DSPARK restores |
|---|---|---|---|---|---|
| base-1switch | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| base-default | device default (OFF) | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| base-default2 | device default (OFF), repeat | 383 / 371 | FAIL | 2992 of 3109 (off-grid) | none |
| base-3doors | all three doors `=1` | 371 / 371 | pass | 2976, 3072 | 2976 of 3109, 3072 of 3144 |
| grid-1switch | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 371 / 371 | pass | none (1 grid-law refusal) | none |
| final-on | `MEMRA_DSPARK_PREFIX_RESTORE=1` | 371 / 371 | pass | 2976, 3072 | 2976 of 3109, 3072 of 3144 |
| final-off | `MEMRA_DSPARK_PREFIX_RESTORE=0` | 371 / 371 | pass | 2976 of 3109 | none |
| final-default | device default (OFF) | 371 / 371 | pass | 2976 of 3109 | none |

Turn 1 is sha 523ca36af0 in all eight boots. Root cause confirmed: the grid law alone
(grid-1switch) turns the reported case green, and its turn 2 then misses (the only cover is the
off-grid 2,992 entry) and primes cold, which the capture change (final-off) turns back into an
on-grid 2,976-token hit. On that build the timing at the switch OFF was 6.03 / 6.03 s base vs 6.24
/ 6.24 s fix on turn 1, flat on turns 2 and 3. That run's GPU unit tests (`cells-6c957b5f2f/gpu-unit.log`,
no switch set) passed 23 of 24; the fault matrix needs the switch, which the landing-head run sets.
