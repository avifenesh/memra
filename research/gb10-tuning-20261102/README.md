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
  artifact and drafter first (acceptance below; quality panel pending).

## Serving validation at `b014549cb9` (`serving/`, `booking-ab/`, `f8f4-accept/`)

Agentic bench (`tools/arms.sh`, harness of the earlier Spark runs: 48 turns from 24 coding-agent
sessions, thinking on, T=1, 768 max new tokens, 262,144 context, 32 sessions), aggregate tok/s,
one run per cell, no HTTP error and no `[shed]` line in any arm:

| arm | c1 | c4 | c8 | c16 | c32 |
|---|---|---|---|---|---|
| GGUF + masked MTP draft (default K table) | 21.43 | 39.36 | 59.36 | 66.77 | 66.45 |
| same, `MEMRA_SPEC_K_CACHED_LONG_TRIM=2` | 21.29 | 39.05 | 60.47 | 66.77 | 67.89 |
| GGUF + DFlash2, `MEMRA_MTP_SKIP=1` | 12.02 | 39.77 | 59.70 | 66.61 | 68.02 |
| NVIDIA NVFP4 safetensors + DFlash2, `MEMRA_MTP_SKIP=1` | | 33.50 | 50.90 | 54.90 | 56.54 |
| GGUF + masked MTP, `MEMRA_MMQ_F8F4=1` | | 38.93 | | | |

- The K-rung override is flat at every concurrency.
- DFlash2 at c1 serves plain: the run-phase turns hit plain prefix entries (no draft tail) and
  the cache-preferred veto sends them down the plain path (28 vetoes, 3 DSPARK restores). Fixed
  in a follow-up (restorable-hit preference), not in this round's binaries.
- `MEMRA_MTP_SKIP=1` boot lines and MemAvailable: `serving/serve-gguf-dflash*.receipt.log`
  (skip on 98,854,496 kB, skip off 97,111,976 kB at boot).
- Non-streaming 150,413-token prompt under the derived deadline: HTTP 200 in 317.0 s
  (`serving/nonstream-150k.txt`).
- Quality panel (one 131,550-token session, 164 HumanEval + 100 GSM8K on it, 8 concurrent), GGUF
  masked-MTP arm: 158/164 and 98/100, MemAvailable at or above 17.0 GiB. Without the admission
  booking of the cache's growth the same panel fell to 1.02 GiB and the 3 GiB guard stopped the
  server (`booking-ab/`; the booking stays, its seam is deleted).
- F8F4 served acceptance A/B (`f8f4-accept/accept.log`, `tools/f8f4-accept.sh`: T=0, one server
  per arm, OFF/ON/OFF2, OFF and OFF2 byte-identical): accepted/drafted 259/369 = 0.702 on
  against 258/384 = 0.672 off (p2 0.615 vs 0.622, p1 0.781 vs 0.715, p3 0.725 vs 0.683);
  faster on every prompt; a 23,476-token prompt prefilled in 30.4 s against 34.4 s. Served cold
  prefill with F8F4: 10.02 s at 8,192 tokens (11.30 off) and 43.72 s at 32,768 (49.32 off).
