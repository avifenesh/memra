# Memra v0.134.0

- Retry transient GPU-canary timeouts at startup before marking the GPU unhealthy (#353).
- Use eager MTP verification when a MoE MTP head is non-resident, avoiding an illegal
  CUDA stream capture operation (#354).
- Include the DSV4 small-kernel diet experiment (#339), controlled by
  `MEMRA_DSV4_SMALL_KERNEL_DIET` and disabled by default.

No serving defaults or published performance numbers change in this release.

Publicity: skipped (maintenance release).
