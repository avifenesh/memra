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
