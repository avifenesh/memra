# Ada

| | Recommended use |
|---|---|
| **Role** | Portable compatibility lane, not a tuned performance target |
| **Build** | `sm_89` (source build; no prebuilt ships since 2026-09-02) |
| **Best fit** | Correctness bring-up and hardware reports on Ada NVIDIA GPUs |
| **Start with** | Build from source with `MEMRA_CUDA_ARCH=89`, then run `kernel-check` |

Do not transfer Blackwell performance or default claims to Ada. A clean compatibility report is a
useful contribution even when no optimization is proposed. See [installation](../INSTALLATION.md)
and [contributing](../../CONTRIBUTING.md#validation-reports-from-your-rig-are-the-easiest-contribution).
