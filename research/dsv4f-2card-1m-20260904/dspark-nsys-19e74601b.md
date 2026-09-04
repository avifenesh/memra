# DSV4 DSpark steady-round Systems profile

Date: 2026-09-04

## Contract

- Memra: `19e74601bc5f656e3a1c40d2293a73a100360a5f`
- binary sha256: `1fb4c6f6abb23b43eb20156fc7e8a4a2b761be12fa6325f606e710aa7aad4b7b`
- artifact: `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`
- hardware: 2x RTX PRO 6000 Blackwell Workstation Edition, PCIe5 x16, PHB, no NVLink
- observed configured power limit: 500 W/card. This is metadata only; the profile does not diagnose a power-limited critical path.
- request: `/v1/chat/completions`, fixed public 24-token prompt, 256 output, temperature 0, ignore EOS
- capture: Nsight Systems 2025.6.3, `cudaProfilerApi`, DSpark steady rounds 4 through 11 only
- report: `/root/dsv4-dspark-fixed24-19e74601b.nsys-rep`, sha256 `53201a72079e836950045fcbbb81f9af94abfca3a11a29f1a235cb6855a8dfc3`
- observed request: 6.478268 s wall, 39.5167 tok/s, 76 rounds, 376 drafted, 179 accepted

## GPU kernel-time census

| kernel | total ms | instances | share |
| --- | ---: | ---: | ---: |
| `dsv4_fp4_gemm_sel_kernel` | 162.700 | 1,032 | 40.5% |
| `dsv4_gemv_fp8_m_kernel<6>` | 88.622 | 5,328 | 22.0% |
| `dsv4_dots_f32acc_mrow_kernel<6>` | 31.748 | 2,040 | 7.9% |
| `dsv4_dots_f32_kernel` | 14.453 | 152 | 3.6% |
| `dsv4_sink_scores_mq_f32acc_kernel` | 14.176 | 344 | 3.5% |
| `dsv4_fp4_gemm_kernel` | 12.712 | 1,131 | 3.2% |
| `dsv4_rmsnorm_f32acc_kernel` | 11.815 | 1,888 | 2.9% |
| `dsv4_hc_sinkhorn_m_kernel` | 9.831 | 688 | 2.4% |

The captured eight rounds issued 32,526 `cudaLaunchKernel` calls, 14,568
device-to-device copies, 2,422 async allocations and 2,422 async frees. CUDA API
submission time was led by kernel launches (187.971 ms), D2H copies (120.614 ms)
and D2D copies (103.147 ms). This establishes a launch/copy-heavy path and names
the top kernels; it does not by itself classify those kernels as bandwidth- or
compute-limited.

## Counter-level follow-up

Nsight Compute 2026.1.1 targeted the 101st
`dsv4_fp4_gemm_sel_kernel` launch with SpeedOfLight and MemoryWorkloadAnalysis.
Both devices returned `ERR_NVGPUCTRPERM`: this provider does not expose NVIDIA
performance counters to the container, including to root. No counter-derived
memory/compute/power conclusion is made. A provider with counters enabled is
required for that classification.
