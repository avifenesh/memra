# Scoped CPU companion — bounded component results

Frozen `65dae02279d2a10f03f055b6907c421a0fd1f2c2` builds with GCC 13.3/OpenMP and
warnings denied on Linux x86_64. Four separate tiny processes pass: buffered/direct I/O, each
with pipeline off/on. The complete five-file quoted-include closure is read-only and verified
before and after the run. Compiler, libgomp, source and executable hashes are retained in
[the exact result bank](attempt-1-65dae022/evidence.json).

All four cases verify:

- ABI2 and scoped single-token/two-row outputs are bit-identical on the same Q8_0 fixtures.
- Scoped cold reads execute; a second demand hits the existing cache without another read.
- Detached callbacks remain blocked after prefetch returns, caller owners are released and the
  pathname is replaced. Readers remain alive until explicit unblock. Original annex bytes and
  final destruction are then verified.
- Failure on the second reader rolls back the first prepared annex claim and counter, with no
  submitted I/O or retained reader leak.
- Demand and prefetch EIO drain, publish no cache/annex bytes, and permit successful same-key retry.

The executable SHA-256 is
`697e9e8b15c7bb0428463c2e2594cc62a80ff6aa7911882a65ab1b63e78b0ce0`.
Controls used four OpenMP threads, four I/O workers, one compiler process, an empty
`CUDA_VISIBLE_DEVICES`, a small private cache, separate target/receipt directories and synthetic
128/512-wide matrices. Recorded elapsed times are harness durations, not performance claims.
No GPU initialization, model-wide benchmark, shared configuration change or rental occurred.

Earlier attempts remain intact: `5fc078c4` omitted IQ3S from its transport snapshot; `d71f4440`
reached GCC but failed its internal-linkage warning. Neither ran test cases. The namespace-only
fix at `65dae022` suppresses no warning and changes no numerical loop.

Scope remains limited: this qualifies the tested CPU companion behavior, not all model formats,
Rust engine dispatch, scoped mirror reads, root activation, model support, serving, GPU execution
or main integration. The Rust bridge is separate work; active scoped mirrors still refuse until
the alternate-reader contract exists. Independent exact-ref review remains required.
