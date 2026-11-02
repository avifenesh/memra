# DGX Spark (GB10)

| | Recommended use |
|---|---|
| **Role** | Measured with Qwen3.8-27B; no model has a support record here yet |
| **Build** | `sm_121a`, auto-detected from GB10 compute capability 12.1; build from source on the GB10 host (aarch64, no prebuilt) |
| **Best fit** | Qwen3.8-27B NVFP4+Q5_K GGUF with its masked MTP draft, the derived unified-memory defaults, no flags |
| **Start with** | Qwen3.8-27B, the first candidate to measure here |

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

Correctness: kernel-check ALL GREEN (92 cells, 22 skipped for models not on the device);
`run-gen` argmax MATCH 4/4 on the GGUF and 4/4 on NVIDIA's NVFP4 safetensors; `run-spec`
K=1..8 self-consistency PASS ([receipt](../../research/gb10-tuning-20261102/correctness-gb10-b014549cb9.log)).

Agentic bench (48 turns from 24 coding-agent sessions, thinking on, T=1, 768 max new tokens,
262,144 context, 32 sessions), aggregate tok/s, one run per cell:

| arm | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| GGUF + masked MTP draft (default K table) | 21.43 | 39.36 | 59.36 | 66.77 | 66.45 |
| GGUF + DFlash2 drafter, `MEMRA_MTP_SKIP=1` | 12.02 | 39.77 | 59.70 | 66.61 | 68.02 |

From c4 up the spec gate serves batched plain decode, so the arms converge. No cell returned an
HTTP error, and no `[shed]` line was logged. Cold prefill: 11.3 s at 8,192 tokens, 49.3 s at
32,768.

Long context: the 264-task quality panel (one 131,550-token session, then 164 HumanEval and 100
GSM8K tasks on it, 8 concurrent) scored 158/164 and 98/100 on the GGUF arm; MemAvailable stayed
at or above 17.0 GiB. A non-streaming request with a 150,413-token prompt answered HTTP 200 in
317 s under the derived deadline.

Prefill: the W4A8 NVFP4 MMQ is 61% of an 8,788-token prefill (776.6 tok/s monolithic). The
opt-in F8F4 route (`MEMRA_MMQ_F8F4=1`) is 15.4% faster here; it changes the activation program,
so it needs this artifact's acceptance and quality receipts before it is recommended
([research/gb10-tuning-20261102](../../research/gb10-tuning-20261102/)).

