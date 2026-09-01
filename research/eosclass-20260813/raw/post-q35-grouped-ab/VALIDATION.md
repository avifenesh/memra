# Validation

Verdict: **PASS as a discriminating grouped-dispatch A/B; grouped remains a
correctness NO-GO.** `overall.exit=1` is the expected aggregate result because the
grouped-ON diagnostic arm fails the frozen gate. Both arms used the same repaired
server binary, model, workload, and one uninterrupted `/tmp/memra-5090.lock` hold;
only `MEMRA_MOE_GROUPED` changed.

## Grouped ON (`MEMRA_MOE_GROUPED=1`)

- Client exit 1; frozen gate verdict FAIL.
- All 8 serial seeds returned HTTP 200 but stopped at 25/60 tokens, with one SHA-256
  `ef6fa7bc01474781a869b7fee31c7224305d6673456d1a9dc71126f636acceec`.
- All 20 mixed-c4 requests (18 full-prefix hits and 2 misses) likewise stopped at
  25/60 with that same hash; 0/20 passed.
- Counters reconcile: 20/20 admitted/completed, 18 hits, 2 misses, 87,480 cached
  prompt tokens, 500 output tokens, zero cache evictions, admission/VRAM defers, or
  OOM parks.

## Grouped OFF (`MEMRA_MOE_GROUPED=0`)

- Client exit 0; frozen gate verdict PASS.
- All 8 serial seeds completed 60/60 with one SHA-256
  `b723be26c76590659d44165c5feabc0ad705653a81df16846bbf3aa248ec7be1`.
- All 20 mixed-c4 requests (18 hits and 2 misses) completed 60/60 with that same hash;
  20/20 passed and the cell reported no integrity failures.
- Counters reconcile: 20/20 admitted/completed, 18 hits, 2 misses, 87,480 cached
  prompt tokens, 1,200 output tokens, zero cache evictions, admission/VRAM defers,
  or OOM parks.

Neither arm carried routed-MoE continuation-prime batches. Each failure-signature
scan contains only the startup gpu-watch banner enumerating fatal Xid codes; neither
log contains an observed Xid, CUDA error, OOM, panic, illegal address, or carried-prime
violation.

This falsifies the hypothesis that the global eager-B1 default repair also closes the
grouped-dispatch 25-token failure. Grouped dispatch is a separate class and must remain
off; this lane does not unfence or modify it. The binary SHA-256 for both arms is
`17a222026e08b65f9344407ba9108cb554688c0431365932bb4e78de1033597d`.
The provenance snapshot records a pre-existing ColBERT process using 1,390 MiB; no
throughput claim is made from this exactness run.
