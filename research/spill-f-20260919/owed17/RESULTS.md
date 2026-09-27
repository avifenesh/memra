# OWED 17: cold read-once bypass (`MEMRA_MOE_COLD_BYPASS`), results so far

Registration: `../M1-PREREG.md` section F, its precedent note, and the D and F fallback amendment.
Code: `crates/memra-engine/src/moe_cache.rs` (door, doorkeeper, bypass branch, `dispatch_source_once`,
`consumed`, `payload`), `spill_pread.rs` (mapped views, ownership, `mapped_serves`),
`hybrid_forward.rs` (the two batch-1 decode expert GEMMs). Frozen run-gen: `build/`.

## Correctness (all green)

- CPU unit cells: `cold_ghost_first_miss_bypasses_second_admits`,
  `cold_ghost_remembers_only_the_last_cap_first_misses`, `cold_bypass_parse_refuses_unknown_values`.
  Full suites: engine lib 576 passed, engine tests and bins, server 940 plus integration
  (`suites/`), fmt and diff checks clean.
- GPU ownership cell (`5090/mapped-gpu-cell/`): `mapped_serve_reads_in_place_and_owns_the_buffer_until_its_event ... ok`;
  the device alias read the landed bytes, `mapped_serves=1`, `h2d_submits=0`, the buffer was not
  handed out before its consumer event and returned to the pool after it.
- Forced-ON smoke on the 5090 (`5090/f17-smoke/`, gate `5090/f17-smoke-gate.queue.log`): all three
  arms generated 128 tokens equal to the byte oracle; `refused=[]`. Bypass lines:
  `[moe-bypass] mode=staged bypassed=664940 ghost_admits=113104` and
  `[moe-bypass] mode=mapped bypassed=662095 ghost_admits=112831` with `mapped_serves=662095`
  (equal to bypassed) and `h2d_submits=112831` (equal to the repeat-miss admits).
- Door off is the same worker16 program as the B3 build (`5090/doorless-diag/`, unscored, three
  alternating pairs): tokens equal the oracle in all six visits; tok/s 15.65, 9.41, 8.81 (B3 build)
  and 9.11, 9.66, 9.84 (door-off build); fallbacks 895, 36, 0 and 7, 27, 355. The fallback count
  follows run conditions on both binaries (OWED 26), not the build.

## Timing and decision

PRO 6000 (BOX36, scored, `../box36/RESULTS.md`): `bypass-staged` flat (cold 1.0018, bounded
1.0095), `bypass-mapped` loser (cold 0.8725, bounded 0.8958), zero fallbacks.

RTX 5090 (`5090/f17/`, ten rounds on the OWED 26 fix build, capped scope; `5090/f17/pool.log`):

```
M1-5090-VERDICT arm=bypass-staged vs worker16: unscored-regime median_ratio=1.0009822820456946 pairs=2
M1-5090-VERDICT arm=bypass-mapped vs worker16: unscored-regime median_ratio=0.6945136362874433 pairs=2
M1-5090-VERDICT regime_scored=False contaminated={'worker16': 7, 'bypass-staged': 6, 'bypass-mapped': 7}
rounds=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] visits=30 refused=[] gpu_cotenant_unclean=0 visits_with_mmap_fallbacks=0
```

Unscored on shared-volume contamination (6 to 7 visits per arm above 2% foreign device bytes);
zero fallbacks in all 30 visits (the OWED 26 fix at serving shape). Descriptive medians: worker16
16.80 tok/s, staged 16.75, mapped 11.45; the card hit its software power cap and, on the staged arm,
the software thermal slowdown (per-visit telemetry, up to 87 C).

Decision (`../M1-PREREG.md` section F decision): no row on either rig supports turning either
value on. `MEMRA_MOE_COLD_BYPASS` is deleted whole (`31247ab134`); the verdict sits in the FLAGS
"Removed doors, 2026-09-27" section. Reading: copying a cold block into a slot and computing from
device memory beats reading it in place over PCIe on both cards, and skipping admission for first
misses neither helps nor hurts at 8 slots.
