# GB10 tuning, round 1 (2026-11-02)

Device: one DGX Spark (GB10, compute capability 12.1, 48 SMs, unified LPDDR5x), memra built
`121a` on the device (aarch64, CUDA 13.0) from `b014549cb9`. Model: Qwen3.8-27B NVFP4+Q5_K GGUF
(`Avifenesh/Qwen3.8-27B-NVFP4-MTP-GGUF`). Reference card: RTX 5090 Laptop (24 GB, power-limited),
same GGUF, `120a`. Prefill cells are `run-gen` with `MEMRA_PP_ONLY=1` (one warmup, one process per
arm, monolithic forward), the prompt from `tools/profile-prefill.sh`.

## Correctness on GB10 at this build

`correctness-gb10-b014549cb9.log`: kernel-check ALL GREEN (92 cells, 22 skipped for models not
on the device); `run-gen` argmax MATCH on 4/4 chat prompts for the GGUF and 4/4 for NVIDIA's
NVFP4 safetensors; `run-spec` K=1..8 self-consistency PASS (GGUF trunk, embedded MTP head).

## Where prefill time goes

| kernel (share of GPU time) | GB10 pp8788 | GB10 pp36766 | 5090 Laptop pp8788 |
|---|---|---|---|
| `mul_mat_q_nvfp4_w4a8` (W4A8 MMQ) | 60.9% | 50.4% | 72.5% |
| `fa_prefill_bf16kv_pp` (full attention) | 6.9% | 22.3% | 8.7% |
| `quantize_mmq_q8_1_d4` | 5.2% | 4.4% | 2.3% |
| `silu_mul_f32` | 4.3% | 3.6% | 2.1% |
| other element-wise and GDN | rest | rest | rest |
| prefill rate | 776.6 tok/s | 645.2 tok/s | 1075.5 tok/s |

Per call, the MMQ is 1.2x the laptop 5090's time on GB10 (17.2 vs 14.4 ms), while the
element-wise kernels are about 3x (`silu_mul_f32` 7.6 vs 2.6 ms, `add_f32` 2.2 vs 0.76 ms),
the bandwidth ratio of the two memories.

## Verdicts

- **Grouped MMQ rasterization: flat, deleted.** `MEMRA_MMQ_SWIZZLE=G` (G token tiles of one
  weight tile before the next) at G=4/8/16 against row-major: 773.2 to 778.4 tok/s, one balanced
  round in both orders (`swz-ab-gb10-8k.log`); on the laptop 5090 neutral within thermal drift
  (`swz-ab-5090-g8.log`). Logits byte-identical. The MMQ is not DRAM-bound on GB10. Ledger:
  `docs/FLAGS.md` "Removed doors, 2026-11-02".
- **F8F4 prefill route on GB10: +15.4%.** `MEMRA_MMQ_F8F4=1` (e4m3 activations on the
  block-scale MMA) 894.2/894.8 tok/s against W4A8 775.1/775.0, both orders, 3 reps each
  (`f8f4-pp-gb10-8k.log`). It is a different activation program (logits differ), so per its
  FLAGS row it stays per-artifact: adoption needs the acceptance and quality receipts for this
  artifact and drafter first. Not adopted in this round.
