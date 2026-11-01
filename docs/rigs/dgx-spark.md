# DGX Spark (GB10)

| | Recommended use |
|---|---|
| **Role** | New machine: builds and starts, no model supported yet |
| **Build** | `sm_121a`, auto-detected from GB10 compute capability 12.1; build from source on the GB10 host (aarch64, no prebuilt) |
| **Best fit** | Not measured yet |
| **Start with** | Qwen3.8-27B, the first candidate to measure here |

GB10 is a 120a-family target: the same sources, kernels and arch-keyed defaults as the RTX 5090
and RTX PRO 6000 builds, compiled to its own SASS, because sm_120a SASS does not load on 12.1
([decision](../decisions/SM121-SPARK-TARGET.md)). The `121a` build is compile-covered in CI. The
120a (RTX 5090, RTX PRO 6000) and 90a (H100) builds and their kernel selection are unchanged by it.

Model support is per machine. No model has a support state on this machine until it is measured
here: module load, `kernel-check`, `run-gen` argmax and `run-spec` K=1..8 on the GB10 itself,
then the model's own gates. `docs/support-records.toml` has no GB10 record.

- **Qwen3.8-27B:** first candidate. Not measured on GB10.
- **Qwen3.8-Flash-Next:** not planned for this machine.
- Every other model: not planned until Qwen3.8-27B has a result here.

The kernel and tuning defaults GB10 inherits from the 120a family were measured on RTX 5090 and
RTX PRO 6000 only. GB10 has 48 SMs and unified LPDDR5x memory, so treat each as unconfirmed on this
machine until a GB10 A/B confirms it or re-keys it on the device.

Memory and deadline defaults are keyed on unified memory (`CU_DEVICE_ATTRIBUTE_INTEGRATED`), not on
the 120a family, and take these values here with no flags
([SERVING.md](../SERVING.md#unified-memory-devices-gb10-dgx-spark),
[`research/gb10-defaults-20261101/`](../../research/gb10-defaults-20261101/)):

| default | GB10 value | was |
|---|---|---|
| free memory for sizing | larger of `cuMemGetInfo` free and MemAvailable (page cache counts) | `cuMemGetInfo` free |
| `MEMRA_PREFIX_CACHE_MB` | larger of 2 full-context entries and half of (boot free - 1.5 GiB - 2 GiB headroom - host tier): 34,048 MiB at 70 GiB free | 2 entries: 15.9 GB for Qwen3.8-27B at 262,144 tokens |
| `MEMRA_STREAM_TTFT_MS_MAX` | 997,311 ms at 262,144 tokens (served ctx / (4118 x 48 / 188 / 4) tok/s) | 90,000 ms |
| `MEMRA_SSE_PREFILL_COMMIT_MS` | 60,000 ms | off |

Both boot lines state them: `[prefix-cache] on: ... unified memory: ...` and
`[deadline] unified memory (48 SMs, served ctx N): ...`. Not yet measured on the GB10 itself.
