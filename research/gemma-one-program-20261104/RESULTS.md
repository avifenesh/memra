# Gemma 4 SWA prefill: one program at every prompt length (2026-11-04)

Verdict: `MEMRA_GEMMA_FA_ONE_PROGRAM=1` is bit-identical to the default on both 5090 cells,
so the owner's flip condition holds. The windowed arm is now the only SWA prefill program and
the switch is deleted.

## Question

lane/gemma-restore-exactness-20260819 found that dense-gemma SWA prefill switched attention
program at `t > sliding_window`. Below the window, SWA layers attended unwindowed on f32
operands; above it, windowed on bf16. On that lane's box, rows `[0..932)` of a 932-row prime
and of a 1048-row prime differed in 59 of 60 layers, so the published prefix KV depended on
the total prompt length. `=1` routed every length through the windowed arm. It stayed default
OFF because it changes the bytes short prompts get. The owner's call (2026-11-03): flip it ON
once the gemma4 dense and MoE run-gen, run-spec and acceptance rows pass on the 5090, then
delete the switch.

## Setup

- Card: RTX 5090 Laptop (24 GB), one card, `gpulock` lease per cell. Single runs per arm, run
  OFF then ON, not interleaved.
- Models: `gemma-4-31B_q4_0-it.gguf` (dense, "g31") and `gemma-4-26B_q4_0-it.gguf` (MoE, "g26"),
  QAT Q4_0. Drafters: the Gemma assistant MTP Q8_0 GGUFs.
- Binaries: g31 cells at `c5b7314734`; g26 cells at `a4d7bf0289`. The commits between them touch
  the gemma4 MoE GGUF loader and sm_100a-only kernels, neither on the g31 path. Each status file
  records the binary's commit stamp (`server-stamp`).
- Arms: OFF = the switch unset; ON = `MEMRA_GEMMA_FA_ONE_PROGRAM=1`.
- Cells (`gemma-c.sh`):
  1. `tools/argmax-margin-gate.sh` (calibrated run-gen argmax);
  2. `gemma-gate` spec, K=1..8, `MEMRA_NGEN=128`, on the 19-token prompt
     `research/gemma4-bringup/e4b-chat-watercycle-ids.txt` (stream agreement plus the
     drafted/accepted counts);
  3. `memra-server` with `MEMRA_KVPROBE_AT=900` on prompt A (964 rows, below the 1024 window) and
     prompt B (1584 rows, above it), comparing the rows `[0..900)` digest.

## Results

| model | arm | argmax | spec K=1..8 stream agreement | KV `[0..900)` A | KV `[0..900)` B |
|---|---|---|---|---|---|
| g31 | off | `SUMMARY flips=1 bad=0` | 128/128 at every K | `5511a669...` | `5511a669...` |
| g31 | on | `SUMMARY flips=1 bad=0` | 128/128 at every K | `5511a669...` | `5511a669...` |
| g26 | off | `SUMMARY flips=2 bad=0` | 128/128 at every K | `d2bd929c...` | `d2bd929c...` |
| g26 | on | `SUMMARY flips=2 bad=0` | 128/128 at every K | `d2bd929c...` | `d2bd929c...` |

Acceptance, drafted/accepted per K, is identical between the arms on both models:

| K | g31 off | g31 on | g26 off | g26 on |
|---|---|---|---|---|
| 1 | 75/54 | 75/54 | 75/54 | 75/54 |
| 2 | 126/66 | 126/66 | 116/70 | 116/70 |
| 3 | 174/72 | 174/72 | 137/74 | 137/74 |
| 4 | 224/72 | 224/72 | 145/75 | 145/75 |
| 5 | 228/72 | 228/72 | 149/75 | 149/75 |
| 6 | 229/72 | 229/72 | 150/75 | 150/75 |
| 7 | 230/72 | 230/72 | 151/75 | 151/75 |
| 8 | 230/72 | 230/72 | 151/75 | 151/75 |

The whole-prompt KV digests and the boundary logits digests also match between arms. A: g31
`7a6505e61de8...` / logits `f09d33133f95...`, g26 `cd55374f108a...` / `f85d8b551c34...`. B: g31
`569d3d7a5c94...` / `a351b3652762...`, g26 `9e035917797f...` / `80d7a55ee42c...`.

Throughput, plain decode on the same 19-token prompt (single runs, sequential; read as flat, not
as a measured delta): g31 39.47-39.83 tok/s on both arms, g26 206.50-208.19 tok/s on both arms.

## Reading

- The flip condition holds on both families: argmax, K=1..8 self-consistency and acceptance
  rows are unchanged.
- On this card and these two QAT Q4_0 artifacts, the OFF arm already wrote length-independent
  prefix KV (A and B digests match), and ON wrote the same bytes. The 2026-08-19 violation does
  not reproduce here. These cells therefore show the flip is safe on the 5090; they do not show
  it fixing a live divergence on this card. Where the two programs still differ (the lane's box,
  or a configuration whose short-prompt arm takes other operands), the flip makes the prefix KV
  length-independent by construction, since only one program remains.

Raw output: `raw/` (status files, every spec, argmax and kvprobe log). Driver: `gemma-c.sh`.
