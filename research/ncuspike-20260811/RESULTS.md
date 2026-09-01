# Step-3.7 decode SOL anatomy — box1

Date: 2026-08-11. Lane: `lane/cx-ncuspike`. Verdict: the device-resident sigmoid router
did move B=1 decode beyond the frozen model's 27–29% band. The same-lock N=5 server median
is **101.282 tok/s**, or **34.526%** of the frozen 1.79 TB/s SOL and **38.698%** of the
actual Server Edition card's 1,597 GB/s memory-bandwidth SOL. The remaining gap is
memory/shape dominated: Nsys leaves only **5.90%** as launch gap, while
`qmatvec_iq4_XS_dp4a` alone occupies **30.17%** of the token wall.

No runtime source was changed. All runtime code below is clean commit
`1808220ead39d515a0854df49d1bb6452b558209`; the separate research harness only selects
the production B=1 call and cache-allocation seams.

## Measurement contract

| item | pinned value |
|---|---|
| rig | hyperscaler sbox-2card `<private-host-redacted>`, 2x RTX PRO 6000 Blackwell Server Edition |
| power / clocks | stock 600 W limits; no application clock, power cap, or NCU clock control |
| source | clean `1808220ead39d515a0854df49d1bb6452b558209`; full release rebuild, 3m53s |
| profile binary | measurement-only `ncuspike-profile`, SHA-256 `5339bc01dfbcfd21d1e8e9c36df68b2485f508db2c4b09955f5eef8fc5746268` |
| artifact | Step-3.7-Flash IQ4_XS, three GGUF parts, 104,993,562,624 bytes total; part hashes in the raw contract |
| runtime shape | PP2 devices 0,1; stage-owned KV; resident experts; default device sigmoid router; serving B=1 sampled/lean path; depth 512 |
| profilers | Nsight Systems 2026.1.3; Nsight Compute 2026.1.0; CUDA 13.2; sm_120a |
| isolation | detached from minute one; `/tmp/memra-gpu.lock` held around each GPU block; compute-app idle gate and host-pressure gate passed |
| cleanliness | `window_clean=false`: box1 was shared with dualpp1/moesd this tick, although every GPU block was serialized and exclusive |

The Nsys run started from P8 at 26–27 °C. Its exit snapshot was 35–36 °C, 2,400 /
2,317 MHz SM, 12,481 MHz memory, and 600 W limits. This is one cold-to-warm stock-clock
capture, not an artificially cooled steady-state claim. The adjacent throughput comparison
used one exclusive lock hold, a fresh server and warmup per arm, alternating arm order, and
250 ms NVML sampling.

Raw contract: [Nsys contract](raw/box1/nsys/contract.txt),
[host before](raw/box1/nsys/host-before.txt), [host after](raw/box1/nsys/host-after.txt),
[release build](raw/box1/build-release.log), and
[profile-harness build](raw/box1/profile-build.log).

## Nsys: exact 32-step serving window

The CUDA profiler range contains exactly N=32 serving decode steps after a depth-512 prime
and four uncaptured warmups. The profiled harness reported 106.3 tok/s; that is a validation
of path shape, not the throughput number used for SOL. First-to-last GPU activity is
**9.401 ms/token**, the two-device busy union is **8.846 ms/token**, and wall minus busy is
**0.555 ms/token (5.90%)**. Device 0 contributes 4.097 ms/token and device 1 contributes
4.750 ms/token; only 0.084 ms overlaps across the entire 32-token trace, confirming the
serial PP schedule. The P2P boundary is 0.000741 ms/token.

Times below sum both GPUs. Because the two PP stages are serial, kernel wall shares are
directly meaningful (the measured cross-GPU overlap is negligible).

| kernel | launches/token | ms/token | token-wall share |
|---|---:|---:|---:|
| `qmatvec_iq4_XS_dp4a` | 315 | 2.8363 | 30.17% |
| `moe_gate_up_silu8_dev_q8_v_rows` | 40 | 1.3733 | 14.61% |
| `moe_down8_fma_dev_q8_rows_g` | 40 | 0.9045 | 9.62% |
| `rms_norm_f32` | 136 | 0.5855 | 6.23% |
| `fa_decode_vec_q_v3` | 45 | 0.5775 | 6.14% |
| `add_rms_norm_f32` | 45 | 0.4621 | 4.92% |
| `quantize_q8_1` | 307 | 0.3615 | 3.84% |
| `qmatvec_q6_K_mmvq` | 1 | 0.2881 | 3.06% |
| `moe_router_sigmoid_topk_f32` | 42 | 0.2550 | 2.71% |
| `qmatvec_q5_K_mmvq_mr2_il` | 45 | 0.2463 | 2.62% |
| `router_gemv_f32_w8` | 42 | 0.1924 | 2.05% |
| `fa_decode_combine_f32` | 45 | 0.1545 | 1.64% |

At the family level, trunk/shared/head matvec is 3.371 ms/token (35.85%), norm/quant/
activation glue 1.618 ms (17.21%), the two resident routed-expert kernels 2.278 ms
(24.23%), and attention including combine 0.732 ms (7.79%). The device sigmoid router is
now 0.255 ms/token of GPU work and produces no per-layer D2H: the trace has only the one
4-byte sampled-token D2H per step.

Raw: [GPU trace CSV](raw/box1/nsys/cuda-gpu-trace.csv),
[kernel summary text](raw/box1/nsys/cuda-gpu-kern-sum.txt),
[CUDA API text](raw/box1/nsys/cuda-api-sum.txt),
[memory-op text](raw/box1/nsys/cuda-gpu-mem-time-sum.txt), and
[run log](raw/box1/nsys/nsys-run.log). The binary `.nsys-rep`, which embeds invocation and
host details, was hashed and removed before collection; its SHA-256 receipt is
[here](raw/box1/nsys/report.sha256).

## NCU: dominant decode kernels

The replay was restricted to four exact symbols, one countered launch per distinct launch
configuration on device 0, stock clocks, with cache-control `all`. It covered 12 configurations:
eight IQ4_XS matvec shapes, two depth-dependent FA shapes, and one shape for each resident-expert
kernel. Nsys remains the timing authority: each NCU launch took 11 replay passes, so neither the
680.7-second replay process nor its per-launch duration is a throughput measurement.

The four selected symbols account for 65.34% of unperturbed Nsys kernel time and 60.54% of the
token wall across both PP devices. The actual countered device-0 configurations account for
31.69% of all kernel time: one PP device was deliberately sampled, and the two-token replay did
not reach the third, later-depth FA grid. The valid Nsys window shows the same kernel classes on
both stages with similar per-device times, but the counters below remain explicitly device-0
mechanism evidence rather than a fabricated two-device average.

| countered kernel | counter-covered Nsys ms/token, dev0 | launch shapes | DRAM GB/s | card-BW SOL | SM peak | achieved occupancy | waves/SM |
|---|---:|---:|---:|---:|---:|---:|---:|
| `qmatvec_iq4_XS_dp4a` | 1.4482 | 8 | 811.4 | 50.81% | 44.89% | 68.02% | 3.33 |
| `moe_gate_up_silu8_dev_q8_v_rows` | 0.6499 | 1 | 1,133.8 | 70.99% | 17.56% | 43.38% | 2.27 |
| `moe_down8_fma_dev_q8_rows_g` | 0.4276 | 1 | 588.2 | 36.83% | 17.05% | 44.91% | 0.91 |
| `fa_decode_vec_q_v3` | 0.2350 | 2 | 67.1 | 4.20% | 21.51% | 32.91% | 0.33 |

Values are weighted by each launch configuration's time in the unperturbed Nsys window. “Card-BW
SOL” is measured `dram__bytes.sum.per_second` divided by the RTX PRO 6000 Server Edition's
1,597 GB/s GDDR7 specification; it is not a claim that every kernel should saturate DRAM. FA at
depth 512 is deliberately small and underfilled. The resident gate/up kernel is already the
closest of the four to the bandwidth roof; down has only 0.91 waves/SM.

IQ4_XS is the larger target. Its eight shapes span 1.67–66.18% of card bandwidth. The
`grid=(4096,1,1)` shape alone consumes 0.6038 ms/token on device 0—41.7% of that device's
IQ4_XS time—while reaching only 54.99% of card bandwidth. The 11,264- and 12,288-row shapes
reach 63.56% and 66.18%, whereas the 64-, 96-, 1,024-, and 1,280-row shapes reach only
1.67%, 2.43%, 17.69%, and 21.07%. Long-scoreboard stalls dominate every selected IQ4_XS
shape (10.13–21.14 warps per issue), which is consistent with a latency/shape-sensitive
weight walk, not a remaining router-readback or aggregate launch-gap bottleneck.

Raw: [NCU CSV](raw/box1/ncu-device0/raw.csv),
[full text export](raw/box1/ncu-device0/details.txt),
[driver log](raw/box1/ncu-device0/driver.log), and
[host before](raw/box1/ncu-device0/host-before.txt) /
[after](raw/box1/ncu-device0/host-after.txt). The replay began at P8 and 27 °C on both cards
and exited at 33/32 °C. The binary `.ncu-rep` was hashed and removed before collection; its
SHA-256 receipt is [here](raw/box1/ncu-device0/report.sha256).

## Did router D2H removal move the SOL fraction?

Yes. The active-parameter bill remains 6,101,901,312 bytes/token, so the controlled N=5
server medians can be placed on the same denominator without changing the model geometry.

| denominator | 1-card SOL | increment 1 (`MEMRA_MOE_DEV=0`) | resident default | movement |
|---|---:|---:|---:|---:|
| frozen lane model, 1.79 TB/s | 293.351 tok/s | 84.520 tok/s = 28.812% | 101.282 tok/s = 34.526% | +5.714 pp |
| actual Server Edition, 1,597 GB/s | 261.722 tok/s | 84.520 tok/s = 32.294% | 101.282 tok/s = 38.698% | +6.404 pp |

Throughput rose **19.831%**, with resident default winning all five paired repetitions.
Thus the old 27–29% statement accurately describes the increment-1 control on the frozen
denominator, not the post-removal default. The profile's 5.90% launch gap and absence of the
42 full-row router readbacks provide the mechanism receipt.

The brief calls this “HBM BW,” but the Server Edition card actually uses GDDR7. NVIDIA's
current product specification gives **1,597 GB/s**; that is the hardware denominator used
above. The 1.79 TB/s row is retained only because the frozen SOL model used it, allowing a
controlled before/after comparison. Source: [NVIDIA RTX PRO 6000 Blackwell Server Edition](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).

Raw N=5 receipt: [summary](raw/box1/sigrouter-perf-receipt/summary.json),
[paired points](raw/box1/sigrouter-perf-receipt/points.jsonl), and
[driver log](raw/box1/sigrouter-perf-receipt/driver.log).

## Next SOL lever

The next experimental seam is the B=1 IQ4_XS matvec's shape-sensitive memory walk at runtime
commit `1808220`: `crates/memra-engine/cu/qmatvec.cu:5387`, with its shared hard-wired
`block=(128,1,1)` launch geometry in `crates/memra-engine/src/lib.rs:4077-4084`,
`:5999-6009`, and `:7505-7515`. It assigns one output row per CTA. Start with an
exactness-gated launch/kernel geometry trial for `out_f=4096`, then the underfilled 1,280-
and 1,024-row cases; keep the current reduction-order path as the oracle and require the full
kernel-check / run-gen argmax / run-spec K=1..8 battery before any promotion.

That seam owns 30.17% of the token wall and reaches only 50.81% of card bandwidth when weighted
across its device-0 shapes. It offers more absolute recoverable time than the 9.62% down kernel,
while the entire measured non-GPU launch gap is only 5.90%. The next comparison should therefore
be an interleaved, same-lock Nsys N>=32 shape trial plus the same per-config counters, not another
router or host-launch change.

No optimization is implemented in this lane.

## Validity correction and excluded diagnostics

Before the successful rebuild, two detached inline launcher attempts exited before compilation
or any GPU use. Their quoted failures were `not a git repository`, `cargo: No such file or
directory`, and `could not find Cargo.toml`; the standalone build script then completed cleanly.
The failed launcher logs are retained as
[attempt 1](raw/box1/build-launch-attempt1.log) and
[attempt 2](raw/box1/build-launch-attempt2.log).

The first two captures are retained but excluded from every result above:

1. `run-gen` captured the requested generation and then its default prompt replay plus
   32-token steady-state measurement (155 full-vocabulary D2H endpoints total), so it was not
   a clean one-window trace.
2. The legacy `decode-window-profile` predates PP stage-owned caches and calls `Cache::new`.
   Under PP2 that placed stage 1 KV on device 0; stage 1 then peer-read its cache. The trace's
   `fa_decode_vec_q_v3` asymmetry (0.278 vs 17.095 ms/token by device) exposed the invalid
   harness. Production allocates with `pp::new_cache` (`crates/memra-server/src/worker.rs:5272`
   at runtime commit `1808220`).

The replacement [measurement harness](profile-harness/src/main.rs) is an external path-dependent
crate: it links the clean pinned runtime, allocates with `pp::new_cache`, and calls the worker's
B=1 sampled/lean batch seam. It does not alter the runtime source tree. Its valid 106.3 tok/s
Nsys window agrees with the adjacent 101.282 tok/s N=5 server median; the invalid legacy harness
managed only 38.4 tok/s.

Excluded raw evidence is under
[`diagnostic-run-gen`](raw/box1/diagnostic-run-gen/) and
[`diagnostic-legacy-cache-nsys`](raw/box1/diagnostic-legacy-cache-nsys/). A device-0 NCU
replay completed against that same invalid legacy harness before the cache-placement error was
isolated; it is retained under
[`diagnostic-legacy-cache-ncu-device0`](raw/box1/diagnostic-legacy-cache-ncu-device0/) and no
counter from it enters `summary.json`. These diagnostics are preserved so the rejected numbers
cannot be mistaken for missing or silently discarded runs.

The machine-readable reduction, including launch configurations and both SOL denominators, is
[`summary.json`](summary.json).
