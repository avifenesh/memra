# Matched single-token compute probes, 2026-09-07

Verdict: do not integrate the current mixed-MMA layout or the compiler pragma.
These are component measurements on two RTX PRO 6000 Blackwell cards, not model
token rates, and neither changes runtime defaults.

`tools/dsv4-mixed-vs-f16-gate.cu` includes the actual `moe_f16_grouped.cu`
and calls `memra_moe_kq_gemm_sk_m1_half2`. Both arms use the same payloads,
selected experts, normalized-half-equivalent activation values and outputs.
The control uses a full 128-entry sparse expert bank, not a compact six-entry
metadata table. Each timed launch follows a 256 MiB flush, twice queried L2.
Three ABBA cycles give six observations per arm and active count on each card.

| Active experts | Current f16 median, us (GPU 0 / 1) | Mixed median, us (GPU 0 / 1) |
|---|---:|---:|
| 1 | 43 / 43 | 66 / 66 |
| 3 | 49 / 49 | 72 / 72 |
| 4 | 53 / 51 | 76 / 76 |
| 6 | 62 / 61 | 90 / 90 |

The mixed output matches its own K16 oracle exactly on these fixtures; the
current f16 reduction differs by up to 1.5 absolute / 0.00214 relative.
This remains a named numeric class, not an exact replacement. Both devices
pass memcheck. A prior grouped-versus-sequential self-comparison was not a
matched engine comparison and must not be cited as an engine gain.

Mixed source SHA256 `ece4b8feeec7e885e603d96a5a97cf7d453b078d1c5babef17b56c26b40844ac`;
binary `710fa79347829897dcec41997bbe79333bbc932e9890eee97eb55c7252df9e2b`.

## CUDA 13.3 compiler probe

Current [PTX 9.3 documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/)
introduces entry-scope `mma_throughput`, a compiler-loop optimization hint,
not a hardware throughput-unlock switch. CUDA 13.1 accepts but ignores this
pragma. CUDA 13.3 changes the target kernel from 86 to 98 registers; driver
readback reports 25,104 static shared bytes and occupancy 3 in both arms.

`tools/dsv4-f16-compiler-gate.cu` uses the CUDA 13.1 runtime to load precompiled
13.3 SASS cubins. Both load and run on driver 595.71.05, without PTX JIT or a
driver/system-toolkit change. All 15 parameter offsets and sizes are checked,
including the `proj=1` argument before `n_expert`. Each external arm is compared
against the current CUDA 13.1 kernel, not directly against the other external
arm. Both external arms pass complete-output bit identity and memcheck.

The plain 13.3 compiler is approximately flat on these rounded component
timings. With the pragma, active 3/4/6 medians are 51/57/68 us on GPU 0 and
53/57/67 us on GPU 1, against current 47/51/61 and 49/51/61 us respectively.
The hint is a regression for this kernel and will not enter the engine.

Loader source SHA256 `f5bb3c897f4b00fcdf5b20ece708ccacb6520e383dae6bc8e985c69207410d5c`;
binary `548806f72a90e1cfab7ad8ec164b356ef1410ffd4e5cbc2094ad5a2091e20648`.
13.3 base cubin `166890a8defd528115c4214f87f1cfb63d18c864df19c6d7d39a0978de0c417a`;
pragma cubin `5b9e24023c6110b25e02d6081ab11a2ded0053c8bd4ff89501001c3c71e89d0f`.

Raw paired receipts are retained by the lane controller under
`matched-mixed-{memcheck,cold}-20260907-r2` and
`compiler-{load,memcheck,rate}-20260907-r1`. Their instrumented timings do not
enter the rate verdict. Standalone tools remain reproduction instruments;
no runtime flags or dispatch arms were added for either losing candidate.
