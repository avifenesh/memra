# Eager-only prime fallback program

The source `65510819` native campaign remains **FAILED**: 26/27 cases passed;
the required T16 ordinary-prime fallback failed. All twelve worker diagnostic
cases and the five other gate groups passed with closed leases. Those results
do not qualify the failed selected bundle, whose index remains quarantined.

Seed max absolute error is **4.4796142578125**. The independent T1 comparison
then measures logits at prompt completion and four own-greedy continuations:
**0.35795366764068604 / 0.35865211486816406 / 0.328216552734375 /
0.4310474395751953 / 0.3623616099357605**. The absolute/relative limits remain
0.005 and argmax equality is still required. All three arms happened to emit
`[2, 3, 2, 3, 2]`; this is not a numerical pass.

## Cause and scope

The admission/dispatch contract is wrong: a bundle without CarriedPrime sends
`prime_cache_batch` through ordinary `prime_cache`, while ordinary prime only
requires DecodeEager. Ordinary prime is explicitly a separate prefill program: its
chunk/layer walker uses prefill projections, grouped kernels, and attention; at
16 rows generic matmul can choose prefill branches before consulting FAST. It
does not enter `matmul_decode_exact`, whose separate scope hole #580 already
fixed. The raw ordinary and batch-fallback vectors are byte-identical, so the
measured divergence already exists in ordinary prime. The precise first divergent
primitive is **not** established by this source analysis.

The ordinary driver, chunk/layer, attention and generic matmul function bodies
match current main `0175e39d` byte-for-byte. `main-comparison.json` records each
hash. This is source comparison, not a numerical run of main. #585 retains the
prime-program evidence; the earlier padded PrimeGraph failure stays separate and
unqualified. A common first divergent primitive has not been proven.

## Correction and verification

When prime permission is absent, the public ordinary-prime entry now invokes the
already-qualified `decode_step_h` for each token and retains the complete hidden
stack plus the last seed/logits. Batches reach the same corrected entry. Capacity,
empty input and unsupported overlays refuse before execution; errors after partial
work taint the cache and the batch transaction. Qualified-prime and legacy
prefill dispatch remain unchanged. No new flag, kernel, format, activation
substitution, tolerance change or architecture allowlist is introduced.

`../run-prime-dispatch-tests.py` compiles the actual public entry, eager loop and
batch fallback prefix against explicit CPU stand-ins for CUDA, T1 arithmetic and
the legacy backend. Six tests pass; replaying the frozen655 entry fails five
controls including strict direct and batch routing. These are dispatch/lifecycle
proofs, not GPU math. Linux-target engine/gate/server clippy and all21 runner
controls pass.

The native gate retains the original T16 reproducer, then adds fresh T15/T17,
carried T15/T16/T17, and B=3 inputs in both length orders. Every batch output is
compared against its own independent T1 reference, including seed, full hidden
stack, logits, positions and four own-greedy continuations. All27 selected cases
remain mandatory; the final case contains these eight scenarios. CarriedPrime
and positive PrimeGraph coverage remain excluded. Fresh native execution awaits
independent exact-source review; no corrected native pass is claimed.

`raw-manifest.json` binds the original raw float arrays, token arrays, case logs,
failed result/census and CPU logs to lossless compressed copies. The complete
36ec4ea1 archive and all original ELFs are preserved off-host. The old0ea0 failed
attempt and its16 unrun cases remain unchanged.
