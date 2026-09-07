# DSV4 chunk-32 long-prefill profile

The frozen mechanism probe ran on two RTX PRO 6000 Blackwell Workstation cards
from Memra `6c604f9bf`, exact 0731 NVFP4 Safetensors artifact, with chunk 32 and
no host-cache hit. The 9,952-token request took 83.36 seconds TTFT. Power-limit
readback is metadata only and no power attribution is made.

Nsight Systems 2025.6.3 captured the request after model load. Report:
`/root/dsv4-chunk32-long-6c604f9bf.nsys-rep` (195 MiB on the dev host).

| kernel | total time | instances | share |
| --- | ---: | ---: | ---: |
| selected-expert FP4 GEMM | 28.939 s | 35,655 | 42.0% |
| block-FP8 dense GEMV M=32 | 13.426 s | 184,091 | 19.5% |
| sparse sink score | 9.792 s | 11,886 | 14.2% |
| f32 island dots M=32 | 4.085 s | 70,480 | 5.9% |
| indexer score | 3.982 s | 185,686 | 5.8% |
| indexer top-k | 3.411 s | 185,686 | 4.9% |

The request issued 2,051,245 kernel launches and 1,665,250 D2D copies. CUDA API
time was 52.647 s in launches and 10.889 s in D2D submissions. This profile led
to the native batched indexer change, which reduced the same 9,952-token TTFT to
75.65 s while retaining the scalar T<=8 decode/spec path. It also disproved two
plausible follow-ups: FMA improved short decode but regressed long TTFT, and the
existing block-FP8 tensor-core prefill kernel regressed the frozen on/off A/B.

Nsight Compute hardware counters remain unavailable on this provider
(`ERR_NVGPUCTRPERM` on both cards), so the profile does not classify the top
kernels as power-, bandwidth-, or compute-limited.

## Two-card execution overlap

An interval-union query over every row in `CUPTI_ACTIVITY_KIND_KERNEL` measured
the actual PP2 schedule rather than inferring it from utilization samples:

| device | kernel instances | union kernel-busy wall |
| --- | ---: | ---: |
| 0 | 951,777 | 33.632904 s |
| 1 | 1,170,860 | 35.300691 s |

The two devices' merged kernel intervals overlap by only 0.000000451 s across
68.933595 s of union busy time, about 0.00000065%. The current layer-split PP2
request is therefore effectively serial across the cards. This is direct
evidence that peak single-request performance requires a topology which runs
both GPUs on the same layer or otherwise pipelines independent work. It is not
evidence for any power-limit claim.
