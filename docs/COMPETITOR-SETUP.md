# Competitor setup

How the frozen competitor columns in [Performance](PERFORMANCE.md) were measured, and how to set
up llama.cpp, vLLM and SGLang to reproduce a paired cell against memra. For anyone re-running a
comparison. Each engine runs at its best configuration for the model and card, not its defaults.

## Frozen baselines

Each column is frozen at its date and never refreshed in place. A new comparison is a new
same-session interleaved cell with its own receipts; ratios belong only beside their protocol.

| Board column | Engine | Card | Date | Configuration | Driver and receipts |
|---|---|---|---|---|---|
| 5090 plain decode (`PERF-PLAIN`) and its 6,257-token depth rows | llama.cpp build 9833 (`c818263f2`), `llama-bench` | RTX 5090 Laptop | 2026-08-02 | `-ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 0 -n 128 -d 512,6257`, one rep per process round, N=5 | `run-board-remeasure.sh` in [research/board-remeasure-20260802/](../research/board-remeasure-20260802/) |
| 5090 speculative decode (`PERF-SPEC`) | llama.cpp `llama-server` with `--spec-type draft-mtp`; the 35B row on build b9837, self-MTP from the embedded NextN head (no `-md`), `--spec-draft-n-max` re-swept per class (3/2/4) | RTX 5090 Laptop | 2026-07-18; 35B row 2026-08-02 | short and medium code greedy, long agentic at temperature 0.7 (matched); N=2 or more, the 35B row N=3 | `research/tune-data/rig5090.jsonl`; for the 35B row [research/q35-spec-repair-20260802/](../research/q35-spec-repair-20260802/) and [research/f16g-default-rearb-20260802/](../research/f16g-default-rearb-20260802/) |
| H100 end to end (`PERF-H100`) | vLLM 0.26, offline `LLM.generate` | 1x H100 80 GB | 2026-08-01 | the HF checkpoints below, about 2,048 prompt tokens, 512 generated, single request, N=5 | `tools/h100-vllm-board.sh` with `bench_vllm.py`; [research/h100-flip-full-20260802/](../research/h100-flip-full-20260802/) |
| Gemma-4 cells (history only) | llama.cpp, `-fa 1`, f16 KV, draft depth swept per cell | RTX 5090 Laptop | 2026-07-15 to 2026-07-26 | best against best | [research/gemma4-bringup/](../research/gemma4-bringup/) |

The H100 board is cross-artifact by design: vLLM rejects the GGUFs memra serves, so it ran the HF
checkpoints an H100 user deploys, and memra ran its GGUF artifacts (named per cell in
`tools/h100-vllm-board.sh`):

| Model | vLLM checkpoint |
|---|---|
| Qwen3.5-9B | `RedHatAI/Qwen3.5-9B-quantized.w8a8` |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B-FP8` |
| Qwen3.6-35B-A3B | `Qwen/Qwen3.6-35B-A3B-FP8` |
| Gemma-4 12B | `unsloth/gemma-4-12b-it` (bf16) |
| Gemma-4 26B-A4B | `RedHatAI/gemma-4-26b-a4b-it-FP8-dynamic` |
| Gemma-4 31B | `RedHatAI/gemma-4-31b-it-FP8-dynamic` |
| Gemma-4 E4B | `unsloth/gemma-4-E4B-it` (bf16) |

No external baseline cell has been banked since these. The doctrine since 2026-09-12 is a monthly
cell per served shape against the current pinned SGLang (primary) and vLLM (control), recorded
with commit, container digest and flags under `research/external-baseline-YYYYMM/`
([research/benchmarks.md](../research/benchmarks.md)).

## 1. llama.cpp

The 2026-07 builds used CUDA 13.1, `sm_120a` SASS, `GGML_CUDA_FA=ON GGML_CUDA_GRAPHS=ON
GGML_CUDA_FA_ALL_QUANTS=ON` and `GGML_CUDA_FORCE_MMQ=OFF`. Set `GGML_CUDA_GRAPH_OPT=1` before every
run. `$MODELS` is wherever the GGUFs live.

```bash
export GGML_CUDA_GRAPH_OPT=1
# plain decode, the board denominator: tg128 at depth 512 and 6257
llama-bench -m "$MODELS/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf" \
  -ngl 999 -fa 1 -ctk q8_0 -ctv q5_1 -p 0 -n 128 -d 512,6257 -r 1
# speculative decode through the server; -md only when the GGUF has no embedded NextN head
llama-server -m "$MODEL" -md "$DRAFT" -ngld 999 -ngl 999 -fa on -ctk q8_0 -ctv q5_1 \
  -c 16384 --parallel 1 --temp 0 --spec-type draft-mtp --spec-draft-n-max 3 \
  --spec-draft-p-min 0.2 --host 127.0.0.1 --port 8099
```

- Qwen: `-fa 1 -ctk q8_0 -ctv q5_1` is the denominator configuration.
- Gemma-4: pass `-fa 1` explicitly. `llama-bench`'s `-fa auto` resolves flash attention off for
  the Gemma-4 GGUFs and costs llama.cpp 6-11% (26B short: 168 tok/s under auto, 190 with `-fa 1`).
  KV-quant flags lose on Gemma (26B short with `-ctk q8_0 -ctv q8_0`: 174.7); its best is f16 KV.
- An `-mtp.gguf` that carries no embedded NextN head needs its external draft file (`-md`).
- Check the resolved configuration in llama.cpp's own output; do not trust the flag.

## 2. vLLM

Pitfalls measured with vLLM 0.23.0 on the RTX 5090 Laptop (2026-07); the exact serve lines are in
the [history](history/PERFORMANCE.md#competitor-setupmd-as-of-2026-11-11), and `tools/bench.sh`
prints the 27B one.

- Use a pre-quantized HF checkpoint. vLLM runs GGUF only through a slow dequant path with no
  qwen3_5 hybrid loader, and a bf16 27B or 35B cannot load to quantize on the fly in 24 GB.
- Prefer `--quantization modelopt` NVFP4 checkpoints over compressed-tensors, which took the slow
  Blackwell fallback and usually dropped the MTP head (0% acceptance).
- Attention: `VLLM_ATTENTION_BACKEND=FLASHINFER`. The prebuilt FlashAttention kernels were sm_80
  only and reject fp8 KV; `TRITON_ATTN` is the fallback. KV: `--kv-cache-dtype fp8`, paired with
  FlashInfer.
- Keep CUDA graphs on; never `--enforce-eager` (5-8x slower on sm_120). `--max-cudagraph-capture-size
  32` avoided a Mamba-cache capture crash at low `--max-num-seqs`.
- Spec: `--speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'`. Report the
  no-spec number too (drop the flag).
- The H100 board ran vLLM offline through `bench_vllm.py` (`LLM.generate`, single sequence,
  `enable_prefix_caching=False`).

## 3. SGLang

Pitfalls measured with SGLang 0.5.9 on the RTX 5090 Laptop (2026-07):

- HF safetensors only. GGUF does not load for the hybrid GDN models (no tensor-name mapping for
  the GatedDeltaNet conv state).
- `--attention-backend triton` is required for hybrid GDN on sm_120 in that version (trtllm_mha was
  sm_100 only; FlashInfer and FA3/FA4 were refused by the hybrid-GDN check). KV:
  `--kv-cache-dtype fp8_e4m3`.
- Spec: `--speculative-algorithm NEXTN` with `--speculative-eagle-topk 1`, which the triton GDN path
  requires. The MTP head is embedded in the checkpoint.
- Keep CUDA graphs on with a small `--cuda-graph-max-bs` (4 to 8) for single stream; do not
  `--disable-cuda-graph` or `--enable-torch-compile`.
- On 24 GB, the 27B fits only as NVFP4 (`--quantization modelopt_fp4`); FP8 27B (about 27 GB) does
  not.

## 4. Known incompatibilities

As measured in 2026-07 on the 24 GB RTX 5090 Laptop:

- No engine fits a 27B in bf16 (about 54 GB) or FP8 (about 27 GB), or the 35B at Q6_K_XL (about
  31 GB).
- SGLang cannot load these GGUFs (Q8_0, NVFP4, IQ4_XS, Q6_K); they are llama.cpp and memra only.
- vLLM loads GGUF only on the slow dequant path; it needs the HF modelopt NVFP4 checkpoints.
- The prebuilt FlashAttention, FA3 and FA4 kernels in vLLM and SGLang do not run on sm_120.

## 5. Paired protocol

- **Serial.** One engine on the GPU at a time: vLLM and SGLang memory profiling races any other
  allocator during init. Check that
  `nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader` lists nothing before each
  arm (the 2026-08 board sessions allowed one small embedding server, recorded per run).
- **Same work.** The same prompt bytes for every engine, the same generation length, a single
  stream (`--parallel 1`, `--max-num-seqs 1`, `--max-running-requests 1`), greedy for the code
  classes and a matched temperature for sampled ones.
- **Interleaved.** Arms alternate in one session and thermal window, both orders, N of 3 or more
  per arm, medians reported. The power state is pinned and recorded per window; numbers from two
  power states never pair.
- **Warmup.** One discarded run per engine (weights, JIT, graph capture).
- **Two decode numbers** where an engine has spec: no-spec (kernel speed) and spec on (best
  configuration).
- **Definitions.** Prefill tok/s is prompt tokens over the time to the first token. Decode tok/s is
  (generated tokens - 1) over the time from the first to the last token. memra's arm is
  `MEMRA_NGEN=128 MEMRA_PROMPT_FILE=<prompt> run-gen <model.gguf>`: its `generated ... tok/s (...
  decode, gen-only; prime ...)` line is the decode rate, with the prime time beside it.
  `MEMRA_PP_ONLY=1` with `MEMRA_PP_REPS` times prefill alone ([flags](FLAGS.md)).
- **Record** the engine version or commit, container digest and every flag with the cell.

`tools/full-board-bench.sh` (the 2026-07-30 full-board run on the 5090, llama.cpp against memra)
and `tools/h100-vllm-board.sh` (H100, vLLM) are the interleaved drivers; `tools/bench.sh` is the
older single-stream driver.

## 6. History

The July 2026 session this page used to be (per-engine peak serve lines, the measured numbers on
llama.cpp b9743, vLLM 0.23.0 and SGLang 0.5.9, the old protocol section and the June/July ranking
table with its targets) is in
[history/PERFORMANCE.md](history/PERFORMANCE.md#competitor-setupmd-as-of-2026-11-11).
