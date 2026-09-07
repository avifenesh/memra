# M=1 expert split-K

Base: 80310418b. Numeric class: `moe_m1_splitk_f32_fixed_order`.

The existing CSR prefix sums `ceil(m_e/32) * ceil(out_f/64)` for eligible
nonempty groups. For three M=1 experts, GU has 96 tiles and down has 192.
The profiled 564-block launch therefore has 468 and 372 idle blocks respectively.
All four warps load weights; only warps 0 and 2 execute valid-row MMA in M1.
Half2 changes stores, not the tile count. With route validation disabled,
Rust `live_slots` is six on each rank, an upper bound rather than the observed
local count. The device CSR endpoint remains authoritative. Candidate launch
bounds are 3,072/6,144 blocks, with 1,536/3,072 useful blocks for three local
experts; the reduction explicitly zeros the inactive slot tail. GU reads 25,165,824 packed-weight bytes
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

Dispatch readback: `MEMRA_MOE_F16G=2` admits the direct grouped matrix
visitor; `MEMRA_F16G_SK=32` selects the small-tile branch, with tail ON.
`Dsv4Gpu::set_grouped_m1_tc_for_gate` drains ranks and arms the M1 down
visitor. GU fusion separately composes `set_moe_f16g_gu_m1_tc_for_gate`
with `set_moe_f16g_gu_half2_for_gate` as `GuLaunchKind::M1Half2`;
`set_moe_f16g_down_m1_half2_for_gate` chooses the down packed-store twin.
None of these switches increase N tiles or split K. The new process setter
selects GU and down together for plain single-token matrix transactions.

The supplied profile was read directly: the GU/down aggregate means are
99.1558/40.4526 us, each over 2,752 launches. GPU 0 reports 43 launches of
each per step, GU 98.9114 us and down 40.5850 us, grid 564x1x1 and block
32x4x1. These instrumented timings motivate the component test; they are
not the denominator for its same-plane comparison.

## R3 fixed-16 result and adaptive revision

The owner read the r3 receipt while this session's SSH route was unavailable:

| Local slots | GU oracle / candidate us | GU speedup | Down oracle / candidate us | Down speedup |
| --- | --- | --- | --- | --- |
| 1 | 100.8 / 25.7 | 3.93x | 38.1 / 24.6 | 1.55x |
| 5 | 101.6 / 83.5 | 1.22x | 48.4 / 76.5 | 0.63x |

The fixed-16 design is rejected. Full-model ABBA was not started on it.
Heavy-rank time determines the EP step; the one-slot GU result cannot stand
in for the five-slot rank. Raw r3 logs still need retrieval from the controller.

The current revision uses `moe_m1_adaptive_splitk_f32_fixed_order`. It chooses
`slices = clamp(round(target / (live * ceil(out_f/64))), 1, 16)` from the
device CSR count, with targets 1280 for GU and 768 for down. If the unsplit
tile count reaches the target, slices is one. K boundaries are
`floor(slice * (K/64) / slices)`; every 64-element block is visited once.
The reduction reads adjacent output columns coalesced for each slice, once,
adding only the live slices in ascending order. Inactive compact slots are
zeroed without reading stale scratch. Each CTA binary-searches the existing
CSR offsets instead of rebuilding all 128 experts' prefix.

| Local slots | GU slices / useful blocks | Down slices / useful blocks |
| --- | --- | --- |
| 1 | 16 / 512 | 12 / 768 |
| 2 | 16 / 1024 | 6 / 768 |
| 3 | 13 / 1248 | 4 / 768 |
| 4 | 10 / 1280 | 3 / 768 |
| 5 | 8 / 1280 | 2 / 640 |
| 6 | 7 / 1344 | 2 / 768 |

The component now captures the first eight routed prompt tokens, first layer,
both ranks, GU and down independently. Each cell has 20 ABBA timings per arm,
finite/repeat/canary checks, and raw max abs/rel errors. A separate phase probe
prints partial and reduction timings outside scored ABBA samples. The summary
reports slots 1 through 6, with unobserved counts explicit. Acceptance applies
to every observed token/rank/projection cell: at most 1.02x oracle time, and at
least 2x speedup for one or two slots. No full-model run before acceptance.

Full-model modes retain one loaded model per gate and allocate fresh request
state for every row, draining both ranks before changing the process setter.
Correctness checks both arms, including the six refusal cells each. Sampled
performance uses ABBA, five rows per arm, attention TP enabled, and the
`sample_plus_forward_envelope` timing scope.
