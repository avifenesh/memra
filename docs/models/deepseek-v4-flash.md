# DeepSeek-V4-Flash

DeepSeek's V4 Flash (the 0731 checkpoint) as NVFP4 safetensors on two cards, through its own
engine (`Dsv4Gpu`) and serve door rather than a ModelPlan pack. Bring-up and tuning continue in
[issue #4](https://github.com/avifenesh/memra/issues/4).

| | |
|---|---|
| **Serving route** | NVFP4 checkpoint directory, 2x RTX PRO 6000 Blackwell, TP/EP placement by default since 2026-09-25 (memra #710), PP-2 as the explicit `MEMRA_DSV4_TOPOLOGY=pp` placement (it aggregates more at c2, at a higher TTFT). Throughput last measured 2026-09-28; gate cells re-run on a Max-Q pair on 2026-11-08. Standing gate: none |
| **Pack** | `deepseek_v4`, support `None`: no pack state. The drafter pack `deepseek_v4_dspark` is NativeReference on a tiny fixture (record `deepseek_v4_dspark-reference`), which says nothing about the target <!-- support: deepseek_v4_dspark-reference --> |
| **Artifact** | `tiyuvta/DeepSeek-V4-Flash-0731-NVFP4@5bd16a784784585ca3bd7ccca102f648ceb19245`, the current revision; it differs from the first revision `3369bf0f52808b1bae86a7cab07bd1a7f5042e9d` only in `README.md`, and the 2026-11-06 fix-stack TP/EP cells loaded `3369bf0f` ([`research/fix-stack-path-coverage-20261106/`](../../research/fix-stack-path-coverage-20261106/)). The measurements below name `@bafd09f8cab4f4f4f25e1cdafbcdefc05b90ee38`, which the repository's history (it starts on 2026-11-02) no longer holds. No tracked receipt compares those bytes with the current files. Source `deepseek-ai/DeepSeek-V4-Flash-0731`. No GGUF conversion |
| **Drafters** | the checkpoint's own DSpark drafter, `MEMRA_DSV4_DRAFTER=dspark` (about 10.7 GiB on the second card); unset serves plain |
| **Hardware measured** | 2x RTX PRO 6000 Blackwell: Workstation, Server and Max-Q edition pairs |
| **Input** | text; tools through the DSML protocol |

## Run it

No cookbook block. `memra-server` loads a checkpoint directory whose `config.json` says
`model_type: deepseek_v4` onto cards `0,1` (`MEMRA_DSV4_DEVICES`) with no other flags.

- **TP/EP** (default): every layer on both cards, experts split by id, attention split by head
  (exact attention TP2), the DSpark drafter's experts split the same way. Sixteen plain lanes
  coalesce their steps into one captured B-row step.
- The served program implements the 0731 encoding's activation quantization (`RefFp8Round`) only.
  A preview-encoding checkpoint (`deepseek_v4_preview`) is gate-only: the server refuses it by name
  at load, before the weights load.
- The chat dialect is code (`encoding_dsv4`); the artifact ships no `chat_template`, and the pack
  names the engine renderer.
- Plain requests replay each step on full-token CUDA graphs with the device sampler.

## Measured

2x RTX PRO 6000 Blackwell, 256 max tokens, one boot per row, same text on every request of every
arm. Decision: [DSV4-TPEP-DEFAULT](../decisions/DSV4-TPEP-DEFAULT.md).

Single stream at the flip (2026-09-24/25, median of N=3 rows, N=2 for DSpark), tok/s:

| arm | Workstation pair, c1 greedy | Server Edition pair, c1 greedy / sampled | TTFT p50 (SE) |
|---|---|---|---|
| plain, TP/EP (default) | 80.73 | 71.20 / 72.65 | 172 ms |
| plain, PP-2 (`pp`) | 68.51 | 62.38 / 58.40 | 210 ms |
| DSpark, TP/EP | 82.73 | 72.79 / 63.57 | 207 ms |
| DSpark, PP-2 | 70.60 | 61.72 / 53.90 | 267 ms |

[Receipt](../../research/dsv4f-bringup-20260923/tpep-default/RESULTS.md).

After the September levers (same bits at every step), on Server Edition pairs:

- c1 by the close of 2026-09-27: greedy 97.10 / 97.26 tok/s (TPOT 9.8 ms), sampled
  97.96 / 97.97 ([`ceiling/CEILING.md`](../../research/dsv4f-bringup-20260923/ceiling/CEILING.md),
  "Close"). The same file measures 6.7 GB read per rank per step and a 1.45 ms launch floor.
- Aggregate, greedy, 16 serving lanes, after #906 and #909 (2026-09-28): c4 209-215, c8 289-298,
  c16 336-340 (TPOT 42 ms), c24 312-315 tok/s; a 16-row captured step takes 37.4 ms
  ([`levers-20260927/`](../../research/dsv4f-bringup-20260923/levers-20260927/)).
- Context on TP/EP: the C4 compressed stores split by position across the two ranks, 8.5 KB per
  token per rank at 1M. A served session reaches 1M tokens plain and 500k with DSpark
  ([`kv-split/`](../../research/dsv4f-bringup-20260923/kv-split/)).

## Limits

- `min_p` and `response_format` refuse by name. Penalized greedy serves on the plain path.
- Parked-prefix reuse is opt-in through `MEMRA_DSV4_KV_HOST_MB` (default 0); with it off,
  `n_cached` is 0.
- The drafter route keeps one lane; aggregate comes from the plain TP/EP route.
- A successful load or prompt is not a support state: the target pack declares none.

## Deeper sources

- Bring-up lane: [`research/dsv4f-bringup-20260923/`](../../research/dsv4f-bringup-20260923/)
  (`REBASELINE.md`, `tpep-default/`, `levers-20260926/`, `levers-20260927/`, `ceiling/`,
  `kv-split/`, `lanes16/`).
- Gate constants derived from the loaded program, Max-Q pair, 2026-11-08 (pass/fail cells, no
  performance claim): [`research/dsv4-census-derived-20261108/`](../../research/dsv4-census-derived-20261108/README.md).
- Serve door: `crates/memra-server/src/dsv4_serve.rs`; flags: the `MEMRA_DSV4_*` rows of
  [FLAGS.md](../FLAGS.md).
- Lever-by-lever history and the PP-2 rows before the flip:
  [history](../history/MODELS.md#deepseek-v4-flash-card-lever-history-2026-09-23-to-2026-09-28).
- The DeepSeek-V4.1-Flash pack is a different program with [its own card](deepseek-v41-flash.md).
