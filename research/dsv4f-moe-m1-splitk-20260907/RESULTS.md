# M=1 expert split-K

The r4 adaptive shape passes full-model correctness and sampled ABBA.
Split-K reaches **38.8228 tok/s versus 35.5695 tok/s, +9.146%**, pooled
across ten sampled rows per arm with radix sampling and attention TP2.
There is no regression in any of the 32 observed token/rank/projection cells. Pooled GU speedup is
3.46x at one local expert and 2.07x at two. Down improves 1.23-1.74x by slot
count; the owner accepted this gain as sufficient for this lane on 2026-09-08.
The original fixed-16 prototype was rejected for regressing the heavier rank.

Base: `80310418b`. Measured r4 source: `e05360e7fa98a4e2a497727ee5c28b627098f3d9`.
Numeric class: `moe_m1_adaptive_splitk_f32_fixed_order`. The process gate stays
OFF by default, decide-by 2026-09-21. No serving default changes.

The CSR prefix sums `ceil(m_e/32) * ceil(out_f/64)` for eligible groups.
Three M=1 experts produce 96 GU tiles and 192 down tiles. In the profiled
564-block launch, 468 GU and 372 down blocks therefore only do setup.
All four warps load weights; only warps 0 and 2 perform valid-row MMA in M1.
Half2 changes stores, not tile counts. GU reads 25,165,824 packed-weight bytes
plus 3,145,728 scale bytes; down reads 12,582,912 plus 1,572,864 scale bytes.
At the supplied 99/40 us, those are 286/354 GB/s including scales, before
cache effects. The actual profile aggregate means are 99.1558/40.4526 us,
over 2,752 launches of each kernel. Those instrumented timings motivate the
change and are not the component comparison's denominator.

`MEMRA_MOE_F16G=2` admits the direct grouped matrix visitor;
`MEMRA_F16G_SK=32` selects the small-tile branch with tail ON.
`set_grouped_m1_tc_for_gate` drains ranks and selects M1 down.
GU fusion composes `set_moe_f16g_gu_m1_tc_for_gate` and
`set_moe_f16g_gu_half2_for_gate` into `GuLaunchKind::M1Half2`;
`set_moe_f16g_down_m1_half2_for_gate` selects the down packed-store twin.
None of these older gates split K or add N tiles. Their kernels remain intact
as the oracle.

The adaptive rule is
`slices = clamp(round(target / (live * ceil(out_f/64))), 1, 16)`, with targets
1280 for GU and 768 for down. Counts come from the device CSR endpoint.
When the unsplit tiles already reach the target, slices is one. Integer
K-block boundaries `floor(slice * (K/64) / slices)` cover every input block,
including non-divisor slice counts. Each slice retains the original half
operands and ascending m16n8k16 chain. The second pass reads adjacent output
columns coalesced and adds only live slices in ascending order with
`__fadd_rn`, then applies the original epilogue. No atomics.

Each CTA binary-searches the CSR offsets instead of rebuilding the 128-expert
prefix. The unused A rows 16-31 are removed from candidate shared staging.
The compiled kernels use 18,176 shared bytes, 84 registers and no local/stack
allocation. R3 used 26,128 shared bytes and 88/82 registers for GU/down.
Scratch belongs to the grouped workspace and is reused on its rank stream.
Inactive compact slots are zeroed without reading stale scratch.

| Local slots | GU slices / useful blocks | Down slices / useful blocks |
| --- | --- | --- |
| 1 | 16 / 512 | 12 / 768 |
| 2 | 16 / 1024 | 6 / 768 |
| 3 | 13 / 1248 | 4 / 768 |
| 4 | 10 / 1280 | 3 / 768 |
| 5 | 8 / 1280 | 2 / 640 |
| 6 | 7 / 1344 | 2 / 768 |

With route validation disabled, Rust `live_slots` is six on both ranks, a
launch upper bound. The device CSR count is authoritative. The component
reads the observed count for its guarded allocations and launch; full-model
runs retain the six-slot upper bound. The full-model results below include the
runtime effect of that difference. The owner explicitly accepted the r4
binary for those runs; no kernel change was made after that decision.

Private ops receipt: `moe-m1-splitk-r4-20260907`, under
`research/dsv4f-devpair-20260905/receipts/ondemand/`. Raw logs, hardware and
process inventories, and validator snapshots stay in the private ops repo.
The first eight routed prompt tokens are captured at the first layer on both
ranks, GU and down independently. Each cell has two warmup ABBA cycles and
ten scored ABBA cycles, 20 event timings per arm. These are warm-plane
measurements on two RTX PRO 6000 Blackwell Max-Q cards, with reported L2 of
134,217,728 bytes per GPU. Candidate timing includes partial and reduction
kernels. Separate phase probes are diagnostic only.

| Local slots | Cells per projection | GU current / new us | GU speedup | Down current / new us | Down speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 2 | 100.8000 / 29.1568 | 3.4572x | 37.8536 / 21.7464 | 1.7407x |
| 2 | 5 | 101.6189 / 49.0714 | 2.0708x | 41.4928 / 28.2298 | 1.4698x |
| 3 | 2 | 102.2552 / 59.4000 | 1.7215x | 48.1680 / 34.2712 | 1.4055x |
| 4 | 5 | 102.1392 / 64.0102 | 1.5957x | 52.9427 / 38.9830 | 1.3581x |
| 5 | 2 | 102.4848 / 70.3992 | 1.4558x | 49.5560 / 40.3048 | 1.2295x |
| 6 | 0 | not observed | n/a | not observed | n/a |

Every cell passed finite outputs, bitwise repeat stability and 64-word canaries
before and after output and scratch allocations. The comparison threshold is
`abs(delta) <= 2e-5 * max(abs(reference)) + 2e-4 * abs(reference)`.
For K<=4096, gamma_K is about 2.45e-4; the plane-scale floor handles cancellation
near zero. This is an empirical component bound, not an oracle bit-identity
claim. Max absolute/relative deviation across cells is 9.53674316e-7 /
0.00108104793 for GU and 0.001953125 / 2.16689423e-5 for down.

The original checker required 2x down at one or two slots and returned EXIT=1
after the kernel gate passed. The owner removed that down requirement.
`summarize-accepted.py` and `summary-accepted.json` retain the accepted rule:
no cell may exceed 1.02x oracle time, and pooled per-slot GU speedup must be
at least 2x at slots 1 and 2. Pooled slot-2 GU passes even though individual
cells range from 1.93x to 2.39x. Unobserved slot counts remain explicit.
The original controller exit and parser are preserved in the private receipt.

The earlier fixed-16 prototype was rejected:
Fixed 16 slices gave GU 100.8480 -> 25.6768 us at one slot, but only
101.6016 -> 83.4880 us at five. Down was 38.0976 -> 24.5504 us at one slot
and regressed 48.3888 -> 76.4688 us at five. Numeric, finite, repeat and canary
checks passed, but full-model ABBA was not run on this rejected shape.
The monitor has 624 entries, all for the expected gate PID and the assigned pair.

Private full-model receipt: `moe-m1-splitk-full-abba-r8-20260908`, under the
same private ops receipt directory. Armed correctness passed for both ranks
and all 43 layers, including deterministic repeats and all six refusal cells.
Refused steps preserve committed cache contents and reject retries.

Sampled ABBA uses four fresh-process loads in new/current/current/new order,
five eligible rows each, with 256 prompt tokens and 256 output tokens per row.
Attention TP2 is ON, radix is the default sampler, and
`timing_scope=sample_plus_forward_envelope`. Both arms use the same binary.

| Load | Arm | Pooled tok/s | Eligible rows |
| --- | --- | --- | --- |
| 1 | split-K | 38.744490 | 5/5 |
| 2 | current | 36.188693 | 5/5 |
| 3 | current | 34.971091 | 5/5 |
| 4 | split-K | 38.901424 | 5/5 |

Pooled over 2,560 generated tokens per arm: split-K **38.822799 tok/s**,
current **35.569475 tok/s**, **+9.146392%**. Pooling sums token counts and
decode wall time, not rates. All 20 rows are eligible, non-looped, non-EOS,
and reach position 512. Tokens and final logits are identical within each arm
across its ten rows and both fresh loads. Cross-arm tokens differ because the
numeric class changed; this is recorded and is not an admission failure.
Each new-arm prime/decode row records 22,016 GU and 22,016 down split-K
submissions, while current-arm split-K counters remain zero.

Measured source commit: `e05360e7fa98a4e2a497727ee5c28b627098f3d9`.
Correctness binary SHA256: `f7f56edc6d24c82a9cb98b66b185101c8150353ed2bd52f7a0c3d7b6399a1e14`.
Sampled binary SHA256: `42dea6efa5d92c6f2e1fe8860e5643aaa27d1933a0c14c4d2134d0dce4cb42f4`.
The controller exited 0 at 2026-09-07T22:58:39Z. Later changes affect the
validator and documentation only; the engine sources and measured binaries
remain unchanged. The controller waited with `flock -w 7200`; monitor shutdown
used a stop file and normal exit, with no process kill.

Source-file SHA256s for the measured implementation:

```text
d034320fb62091cc83c27a515642ba52b66171eadc49f1306e01a88f00d8aad6  crates/memra-engine/cu/moe_f16_grouped.cu
6e44a1a24d885f5262458720aa9155697f896fa92b9cdb01e83714789c27af4a  crates/memra-engine/src/dsv4_grouped.rs
a009ecdccadd0fab1b8d552fe280a5d313605c68ad0d262670402e67fd38b6b5  crates/memra-engine/src/lib.rs
673e48d8029da407517310144f4720b5a6d5ad45a25fbe08daa5efdaa75ec0d2  crates/memra-engine/src/mmq_ffi.rs
222ce6cec3ab5137b2f481a9d5b2046159e3b6e0d807d2c3d675bc9344f6d07f  crates/memra-engine/src/dsv4_gpu.rs
bb0783f6c69a05381184707104c04ac9e882f6268c7f902bf034e37631e94621  crates/memra-engine/src/bin/dsv4_tp_ep_gate.rs
5024d75d9463401a7986567bba24146127fc7ff01bbfeec60cf53b592ad01274  crates/memra-engine/src/bin/dsv4_tp_ep_sampled_perf_gate.rs
```

No local build, test, CI or GPU run. Pushes export `MEMRA_SKIP_PERF_CI=1` with
normal hooks, as required by the owner. Hosted CI is green for the r4 source.
