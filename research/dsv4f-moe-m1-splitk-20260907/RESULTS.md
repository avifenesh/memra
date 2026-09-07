# M=1 expert split-K

Base: 80310418b. Numeric class: `moe_m1_splitk_f32_fixed_order`.

The existing CSR prefix sums `ceil(m_e/32) * ceil(out_f/64)` for eligible
nonempty groups. For three M=1 experts, GU has 96 tiles and down has 192.
The profiled 564-block launch therefore has 468 and 372 idle blocks respectively.
All four warps load weights; only warps 0 and 2 execute valid-row MMA in M1.
Half2 changes stores, not the tile count. GU reads 25,165,824 packed-weight bytes
plus 3,145,728 scale bytes. Down reads 12,582,912 plus 1,572,864 scale bytes.
At 99/40 us those are 286/354 GB/s including scales, before cache effects.

The candidate uses 16 contiguous K slices (256 GU elements, 128 down elements),
1,536/3,072 useful blocks with three experts. Each slice keeps ascending
m16n8k16 f32 accumulation over the same dequantized half operands. A second
kernel adds partials 0 through 15 using `__fadd_rn`, then applies the original
scale and GU epilogue. The class is deterministic but is not asserted bitwise
identical to the unsplit chain. No atomics. Scratch is owned by each grouped
workspace and reused on that rank's stream. The process gate defaults OFF.

The component mode uses the first real source token routed through the loaded
checkpoint, both rank partitions, GU and down independently. It checks all
outputs for finiteness, bitwise repeat stability, and 64-word canaries before
and after each output and scratch allocation. Ten ABBA cycles give 20 event
timings per arm after two warmup cycles. Candidate timings include both kernels.
Admission is `abs(delta) <= 2e-5 * max(abs(reference)) + 2e-4 * abs(reference)`.
For K<=4096, gamma_K is approximately 2.45e-4; the plane-scale floor accounts for
cancellation near zero. This empirical component threshold is not a full-model
quality guarantee. Raw max absolute and relative deviations are printed.

Remote build and gate receipts pending. No local build, test, CI, or GPU run.
Pushes use `MEMRA_SKIP_PERF_CI=1` with normal hooks under the owner prohibition.
