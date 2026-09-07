# DSV4 device sampler

Default host; device opt-in, decide-by 2026-09-21. No promotion or merge.

Numeric class: `device-f64-exp-tree-cdf-v1`. Candidate IDs and penalty arithmetic
retain CPU order. CUDA double exp and parallel unnormalized prefix sums can
change a nucleus cutoff or inverse-CDF decision at a rounding boundary relative
to the CPU libm/sequential normalized program. The gate requires token identity
on its frozen tape; this is not universal probability-bit identity. Nonfinite
penalized inputs refuse. Prefill/restore retain their existing host-row contract;
plain forward decode returns only the sampled u32 after the device chain.

Target component tape: 256 rows per device, vocabulary 129280, ties, signed zeros,
subnormals, dominant logits, penalties, tiny top-p, temperatures down to minimum
positive normal f32. Full-model: 10 interleaved ABBA cycles, 20 rows per arm,
256 source-prime and 256 sampled tokens; `timing_scope=sample_plus_forward_envelope`.
Tokens/final logits/cache/hidden digests and actual sampler engagement must pass.

Validation pending on the development pair. Local perf CI skipped per owner.
Raw receipts will be banked in the private ops GPU-sampler namespace.
