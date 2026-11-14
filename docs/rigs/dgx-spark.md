# DGX Spark (GB10)

NVIDIA's GB10 (DGX Spark): a 48-SM Blackwell part with unified LPDDR5x memory on an aarch64 host.
This card is for readers serving on a Spark: build, the defaults that differ there and what was
measured.

| | |
|---|---|
| **Build target** | `sm_121a`, auto-detected from compute capability 12.1; its own SASS in the `120a` policy family ([decision](../decisions/SM121-SPARK-TARGET.md)). Build from source on the GB10 host |
| **Prebuilt** | no |
| **Push gate** | yes: a push that reaches `sm_121a` needs a passing GB10 cell record (`tools/change-impact.toml`); CI compiles the `121a` cell of the arch matrix |
| **Maintained** | yes: push-gated, with its own tuning lanes |
| **Pack records** | none: no record in `docs/support-records.toml` names a GB10 topology |

GB10 builds the same sources, kernels and arch-keyed defaults as the RTX 5090 and RTX PRO 6000,
compiled to its own SASS, because `sm_120a` SASS does not load on 12.1 (and `121a` does not load on
12.0). New policy keys on the `120a` family, never on the target string. The family defaults were
measured on discrete cards; a GB10 A/B confirms each or re-keys it on the device.

## Device-keyed defaults

Memory and deadline sizing keys on unified memory (`CU_DEVICE_ATTRIBUTE_INTEGRATED`); the kernel
routes key on a unified-memory `120a`-family device; the rest key on the device. No flags.

| Default | GB10 value | Discrete cards | Record |
|---|---|---|---|
| free memory for sizing | the larger of `cuMemGetInfo` free and MemAvailable (page cache counts) | `cuMemGetInfo` free | [SERVING](../SERVING.md#unified-memory-devices-gb10-dgx-spark) |
| free memory for a delta (the draft-state measurement) | `MemFree + Buffers + Cached + SReclaimable`, one `/proc/meminfo` read | `cuMemGetInfo` free | [UNIFIED-MEMORY-READERS](../decisions/UNIFIED-MEMORY-READERS.md) |
| `MEMRA_PREFIX_CACHE_MB` | the larger of 2 full-context entries and half of (boot free - 1.5 GiB - 2 GiB headroom - host tier): 34,048 MiB at 70 GiB free | derived from free VRAM and the session peak | [`research/gb10-defaults-20261101/`](../../research/gb10-defaults-20261101/) |
| `MEMRA_STREAM_TTFT_MS_MAX` | 997,311 ms at 262,144 tokens (served ctx / (4118 x 48 / 188 / 4) tok/s) | 90,000 ms | same |
| `MEMRA_SSE_PREFILL_COMMIT_MS` | 60,000 ms | off | same |
| non-streaming `timeout_ms` default and max | 3,045,311 ms at 262,144 tokens (the streaming value plus 8,192 tokens at 4 tok/s) | 90,000 ms | same |
| prefill GEMM | F8F4 (`MEMRA_MMQ_F8F4`, e4m3 activations), v2 tile for GEMMs of 2,048 rows or more, v3 TMA mainloop | W4A8 | [V2](../decisions/GB10-F8F4-GEMM-V2.md), [V3](../decisions/GB10-F8F4-GEMM-V3.md) |
| prime attention | the h2p mbarrier ring | the four-plane db twin | [H2](../decisions/GB10-PRIME-ATTENTION-H2.md), [H2P](../decisions/GB10-PRIME-ATTENTION-H2P.md) |
| prime GDN and glue fusions | on | partly; see the records | [first pass](../decisions/GB10-PRIME-GDN-GLUE.md), [second pass](../decisions/GB10-PRIME-GLUE-2.md) |
| device penalty arm | on for the dense and MoE Qwen3.5 classes | dense class only | [GB10-DEVICE-PENALTY-DEFAULT](../decisions/GB10-DEVICE-PENALTY-DEFAULT.md) |
| DFlash prefix restore (`MEMRA_DSPARK_PREFIX_RESTORE`) | on | off (on for the PRO 6000 Max-Q) | its row in [FLAGS.md](../FLAGS.md) |
| cold-long speculative depth | K=3 (K=3 measured ahead) | K=2 on the RTX 5090 Laptop and Max-Q | [SPEC-K-COLD-LONG-K2](../decisions/SPEC-K-COLD-LONG-K2.md) |

Boot lines state the memory defaults: `[prefix-cache] on: ... unified memory: ...` and
`[deadline] unified memory (48 SMs, served ctx N): ...`. Prefix-cache inserts also clamp to live
free memory, and admission books the cache's allowed growth. The values before the unified-memory
keying are in [history/MODELS.md](../history/MODELS.md#dgx-spark-card-the-pre-unified-memory-defaults-the-was-column).

## Models measured here

| Model | Artifact | Last measured | Receipt |
|---|---|---|---|
| [Qwen3.8-27B](../models/qwen38-27b.md) | NVFP4+Q5_K MTP GGUF; NVIDIA NVFP4 safetensors | 2026-11-10 | [decode](../../research/gb10-decode-20261110/README.md) |
| [Qwen3.6-35B-A3B](../models/qwen36-35b-a3b.md) | UD-IQ4_XS GGUF | 2026-11-10 | [device penalty](../../research/devpenalty-gb10-20261110/README.md) |

Qwen3.8-27B, GGUF with the masked MTP draft, 262,144 context, one cold request at a time
(2026-11-10, main `fcd66f8963` plus the verify-rows walk that landed after it, single runs):

| prompt tokens | TTFT | decode after the first token |
|---:|---:|---:|
| 1,036 | 1.13 s | 31.05 tok/s |
| 131,051 | 117.4 s | 17.00 tok/s |
| 259,951 | 313.9 s | 12.65 tok/s |

Correctness on the device (2026-11-02, `b014549cb9`): kernel-check ALL GREEN, `run-gen` argmax MATCH
on the GGUF and on NVIDIA's NVFP4 safetensors, `run-spec` K=1..8 PASS
([receipt](../../research/gb10-tuning-20261102/correctness-gb10-b014549cb9.log)). Since then every
push that reaches `sm_121a` carries a GB10 cell record under
[`research/release-qualification/cells/`](../../research/release-qualification/cells/).

Four penalised sampled requests at once with the device penalty arm (2026-11-10, three boots per
arm, the on arm ahead in nine of nine boot pairs):
Qwen3.8-27B 41.90 tok/s aggregate, Qwen3.6-35B-A3B 131.41.

## Limits

- Support is per machine: a model measured on discrete cards has no GB10 result until its gates
  run here.
- The agentic bench of 2026-11-02 and the prefill lanes of 2026-11-06 and 2026-11-07 were measured
  before the F8F4 v3 default; their cold-TTFT figures are kept in the
  [history file](../history/MODELS.md#dgx-spark-gb10-card-measured-2026-11-02-to-2026-11-07), not
  quoted as current.

## Deeper sources

- Prefill lanes: [`research/gb10-prefill-20261106/`](../../research/gb10-prefill-20261106/),
  [`research/gb10-prefill-attn-20261107/`](../../research/gb10-prefill-attn-20261107/),
  [`research/gb10-prefill-next-20261107/`](../../research/gb10-prefill-next-20261107/).
- Decode lanes: [`research/gb10-decode-20261109/`](../../research/gb10-decode-20261109/README.md),
  [`research/gb10-decode-20261110/`](../../research/gb10-decode-20261110/README.md).
- Tuning and defaults: [`research/gb10-tuning-20261102/`](../../research/gb10-tuning-20261102/),
  [`research/gb10-defaults-20261101/`](../../research/gb10-defaults-20261101/).
- The unified-memory mechanism: [serving](../SERVING.md#unified-memory-devices-gb10-dgx-spark).
