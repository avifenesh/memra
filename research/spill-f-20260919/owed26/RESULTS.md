# OWED 26: worker demand reads that fell back to mmap, results so far

Registration: `../M1-PREREG.md` section G, its failed serving-shape check and the correction.

## G1, visibility (done)

The runner now reads the fallback count for every positioned-read arm: a direct arm with any
fallback is refused (unchanged), any other positioned-read arm with a fallback is unclean for
timing and counted toward the contamination limit (runner test
`test_fallbacks_gate_every_positioned_read_arm`, stub seam `M1_STUB_FALLBACKS`). Per-visit GPU
telemetry (SM clock, power, temperature, throttle reasons) runs in every timed visit from the next
cell on; recorded cells got post-hoc per-visit summaries (`visits-gpu.json`).

## G2, the fix

- `e5d899500`: a demand submit waits for a buffer (oldest H2D event, next worker completion,
  cancel one prefetch, one stream drain) with a 30 s liveness bound. Red arm
  `5090/owed26-cells-attempt1`: on the pre-fix build the cell failed as intended,
  `demand submit returned None with every buffer in H2d: the ring-busy mmap fallback`; on the fix
  both new cells and all five pool GPU cells passed.
- Serving-shape check on that build FAILED (`5090/owed26-serve-attempt1`): `worker2` generated
  tokens equal to the oracle at 0.65 tok/s and stalled; stopped at 729.6 s. Cause: a wait with a
  free buffer and nothing in flight still blocked for the 30 s bound.
- Correction `50e1cf3f6` (build commit in `build/commit.txt`): return at once with a free buffer and
  nothing in flight or after a reap that freed one; cap each completion wait at 50 ms. Red arm
  (`5090/owed26-cells`, the e5d899500 build with only the two cells added): both failed as
  predicted, `a free buffer with nothing in flight waited 30.000227536s` and
  `completed H2D events waited 30.000136012s`. Green: all seven pool GPU cells pass on the
  corrected build.
- Serving-shape check on the corrected build (`5090/owed26-serve`): `OWED26-SERVE-CHECK PASS
  visits=6`. Every visit: zero fallbacks, zero demand-wait timeouts, tokens equal the oracle.
  Unscored visit rates, capped scope, shared rig: worker2 8.29 and 7.68 tok/s (5,774 and 7,406
  demand waits, 4.2 s and 5.6 s waiting), worker16 15.14 and 14.68 (1,930 and 1,843 waits),
  bypass-mapped 10.43 and 10.19 (4,672 and 3,979 waits). `prefetch_cancels` was 0 in every visit.
- The OWED 17 cell and both PRO sittings use this build (`build/`).

## G3, census (done)

`CENSUS.md`: 67 of 261 directories with a totals line show ring-busy fallbacks; nothing rewritten.
