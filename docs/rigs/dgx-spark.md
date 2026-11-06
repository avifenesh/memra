# DGX Spark (GB10)

| | Recommended use |
|---|---|
| **Role** | Measured with Qwen3.8-27B; no model has a support record here yet |
| **Build** | `sm_121a`, auto-detected from GB10 compute capability 12.1; build from source on the GB10 host (aarch64, no prebuilt) |
| **Best fit** | Qwen3.8-27B NVFP4+Q5_K GGUF with its masked MTP draft, the derived unified-memory defaults, no flags |
| **Start with** | Qwen3.8-27B NVFP4+Q5_K GGUF with its masked MTP draft (measured below) |

GB10 is a 120a-family target: the same sources, kernels and arch-keyed defaults as the RTX 5090
and RTX PRO 6000 builds, compiled to its own SASS, because sm_120a SASS does not load on 12.1
([decision](../decisions/SM121-SPARK-TARGET.md)). The `121a` build is compile-covered in CI. The
120a (RTX 5090, RTX PRO 6000) and 90a (H100) builds and their kernel selection are unchanged by it.

Model support is per machine. No model has a support state on this machine until it is measured
here: module load, `kernel-check`, `run-gen` argmax and `run-spec` K=1..8 on the GB10 itself,
then the model's own gates. `docs/support-records.toml` has no GB10 record.

- **Qwen3.8-27B:** measured below; no support record yet (`docs/support-records.toml`).
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
| non-streaming `timeout_ms` default/max | 3,045,311 ms at 262,144 tokens (streaming value + 8192 tokens at 4 tok/s) | 90,000 ms |

Both boot lines state them: `[prefix-cache] on: ... unified memory: ...` and
`[deadline] unified memory (48 SMs, served ctx N): ...`, and the measured cells below ran on
them. Prefix-cache inserts also clamp to live free memory, and admission books the cache's
allowed growth ([SERVING.md](../SERVING.md#unified-memory-devices-gb10-dgx-spark)).

## Measured (2026-11-02, one GB10, `121a` built on the device from `b014549cb9`)

Correctness: kernel-check ALL GREEN (92 cells; 22 skipped for models not on the device, every
NVFP4 GEMM/MMVQ cell among them, so the 27B's NVFP4 path is covered end to end below);
`run-gen` argmax MATCH 4/4 on the GGUF and 4/4 on NVIDIA's NVFP4 safetensors; `run-spec`
K=1..8 self-consistency PASS ([receipt](../../research/gb10-tuning-20261102/correctness-gb10-b014549cb9.log)).

Agentic bench (48 turns from 24 coding-agent sessions, thinking on, T=1, 768 max new tokens,
262,144 context, 32 sessions), aggregate tok/s, one run per cell:

| arm | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| GGUF + masked MTP draft (default K table) | 21.43 | 39.36 | 59.36 | 66.77 | 66.45 |
| GGUF + DFlash2 drafter, `MEMRA_MTP_SKIP=1` | 12.02 | 39.77 | 59.70 | 66.61 | 68.02 |
| NVIDIA NVFP4 safetensors + DFlash2, `MEMRA_MTP_SKIP=1` | | 33.50 | 50.90 | 54.90 | 56.54 |

From c4 up the spec gate serves batched plain decode, so the GGUF arms converge. The DFlash2
arm serves plain at c1 too, because its multi-turn hits land on plain prefix entries (fix in memra#1031). `MEMRA_SPEC_K_CACHED_LONG_TRIM=2` is flat at every concurrency (21.29 at c1). No cell returned an
HTTP error, and no `[shed]` line was logged. Cold prefill: 11.3 s at 8,192 tokens, 49.3 s at
32,768.

Long context: the 264-task quality panel (one 131,550-token session, then 164 HumanEval and 100
GSM8K tasks on it, 8 concurrent) scored 158/164 and 98/100 on the GGUF arm; MemAvailable stayed
at or above 17.0 GiB (the bench's own 3 GiB MemAvailable watchdog never fired). A non-streaming
request with a 150,413-token prompt answered HTTP 200 in 317 s under the derived deadline (GGUF
DFlash2 boot with the MTP head loaded; 5 output tokens, so it tests the prefill allowance only).

Prefill: the W4A8 NVFP4 MMQ is 61% of an 8,788-token prefill (776.6 tok/s monolithic). GB10 now
takes the F8F4 route (`MEMRA_MMQ_F8F4`, e4m3 activations) by default: 15.4% faster in the
monolithic `run-gen` prefill, 43.5 s against 49.3 s of served cold prefill at 32,768 tokens, one
greedy acceptance run per arm on three prompts at 0.702 against 0.672, c32 68.21 against 66.45
tok/s, and the quality panel at 158/164 and 97/100 (W4A8: 158/164 and 98/100). `MEMRA_MMQ_F8F4=0`
restores W4A8 ([research/gb10-tuning-20261102](../../research/gb10-tuning-20261102/)).

Long-context prefill (2026-11-06, [research/gb10-prefill-20261106](../../research/gb10-prefill-20261106/)):
served cold TTFT of one fresh prompt with the exact levers of that lane (two-head prime attention
by default on GB10, no dead f16 operand pass, no slab copies of the dense FFN's GEMM outputs,
silu(gate)*up fused into the down projection's e4m3 quantize, each shared activation quantized
once per GEMM group, and 8,192-row prime chunks, which were then rejected for peer latency and
did not land; they were worth about 0.6% at 131k), same tokens on every prompt: 44.26 -> 36.16 s
at 32,784 tokens (-18.3%), 239.0 -> 181.1 s at 131,051 (-24.2%) and 637.5 -> 457.2 s at 259,951
(-28.3%), two fresh boots per arm in both orders in each of the lane's two campaigns on one unit.
At 131,051 tokens the remaining time is the F8F4 trunk GEMMs (about 81 s at 78 TFLOPS; the
tile's epilogue form tops out at 113 TFLOPS at its 8 warps per SM), attention (50.9 s at 66
TFLOPS) and the Gated DeltaNet kernels (about 20 s). MemAvailable stayed above 65.2 GiB at
259,951 tokens.
