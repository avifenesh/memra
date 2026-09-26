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
- Correction (next commit after e5d899500): return at once in that case and after a reap that freed
  a buffer; cap each completion wait at 50 ms. Two red-arm cells run on the e5d899500 build (each must
  take the 30 s) and on the corrected build (each within 1 s). Results and the rerun of the
  serving-shape check land in `5090/owed26-cells` and `5090/owed26-serve`.

## G3, census (done)

`CENSUS.md`: 67 of 261 directories with a totals line show ring-busy fallbacks; nothing rewritten.
