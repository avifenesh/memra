# bf16 GDN state: research arm, GB10, Qwen3.8-27B (owner GGUF)

Date: 2026-11-03. Branch `research/gdn-state-bf16`. The switch is `MEMRA_GDN_STATE_BF16=1`, default off. The owner decides on these receipts; nothing here changes the default.

## What the arm is

Every store of the recurrent GDN state rounds to bf16 precision (round-to-nearest-even through `__float2bfloat16`), and the scans keep computing in fp32 (`gdn_st` in `cu/hybrid.cu`). That covers:

- the decode, batched-decode and verify scans with their snapshots;
- the chunked prime state;
- the tensor-core chunk-state kernels.

The arm reproduces the numerics of a bf16-resident state with fp32 math, but the storage stays fp32. It therefore measures **quality and acceptance**, not speed. Speed is measured separately with a bf16-storage bandwidth probe of the batched decode scan (`gdn-state-bench`, kernel `gdn_scan_s128_b_bf16st`).

A full bf16-storage implementation would need the state class changed in every state kernel, snapshot copy and prefix-cache entry. That is not built.

## Default off is unchanged

The bench (`bench-ab.txt`, `bench-tokfp.txt`) runs 16 sessions at depth 2048 with the arm off. Its token fingerprints are identical to the lever-3 binary (`ffd722c3ca34`), so the build with the switch off computes exactly what batch-spec computes.

## Quality panel

The 264-task long-context panel: 164 HumanEval and 100 GSM8K, each after a 131,550-token session (`quality-summary.json`, `quality-records.jsonl`).

| arm | long HumanEval | long GSM8K | truncated |
|---|---:|---:|---:|
| fp32 state (current memra, coordinator's reference) | 158 / 164 | 97 / 100 | |
| **bf16-rounded state** | **160 / 164** | **97 / 100** | 0 |
| panel band on the default (qual_panel.py) | 156-159 | 96-97 | |

In band. The prime's identity check passed with no prefix hits. For about 5 minutes mid-panel, a stray client (a tunnel-port collision on the rig, since fixed) also sent agentic traffic to this server. That changes panel timing, not grading.

## Coherence, greedy and sampled

Eight fixed prompts, 384 tokens each, greedy (T=0) and sampled (T=1, seed 1234), thinking off. Compared against the fp32 base on the same box (`coherence-base.jsonl`, `coherence-bf16.jsonl`).

- **Identical:** 3 of 8 greedy answers and 3 of 8 sampled answers. These are the short ones: the bash one-liner, the Hebrew translation and the SQL query.
- **Diverged:** the long answers part after 53 to 866 characters, and stay on topic and correct. For example, the train-trip arithmetic comes out as 225 minutes on both arms.
- **Bench streams:** over 48 tokens x 16 sessions, 7 of 16 greedy and 12 of 16 sampled streams are identical (`bench-tokfp.txt`).

## Agentic cell, c1..c32

48 real coding-agent turns, thinking on, T=1, one boot (`cells.jsonl`). The arm does not save bytes, so these numbers carry its acceptance and scheduling effects only.

| c | bf16-rounded state | fp32 base, same sequence (`base-full/`) |
|---:|---:|---:|
| 1 | 19.99 | see below |
| 2 | 26.91 | |
| 4 | 37.68 | |
| 8 | 52.58 | |
| 16 | 73.51 | |
| 32 | 76.18 | |

The fp32 base column comes from the same harness and sequence on the same box right after this arm. It is filled in the batch-spec copy of this file once that run lands.

At c=8 the cost gate chose K=1 speculation several times. The verdicts read "predicted 60.7 vs plain 58.5 tok/s", and plain at c=8 had served 61.8 in the lever-3 run. This happens once c1..c4 have filled the acceptance history. The fp32 base run in the same sequence shows whether it is the arm or the gate.

## Speed: bf16-stored state, measured (GB10)

`gdn-state-bench-gb10.txt` covers 48 layers x 48 heads x 128 x 128 per session, with a separate allocation per layer and session:

| sessions | fp32 state (ms per step) | bf16 state (ms per step) | ratio |
|---:|---:|---:|---:|
| 1 | 1.53 | 0.87 | 0.57 |
| 8 | 11.46 | 5.77 | 0.50 |
| 16 | 22.68 | 11.54 | 0.51 |
| 32 | 45.33 | 23.00 | 0.51 |

The fp32 column matches the decode-step nsys (22.7 ms at B=16). Both forms stream at about 210 GB/s, the part's ceiling, so halving the bytes halves the time.

Projected on the measured ticks, assuming no other change:

| cell | measured tick (ms) | GDN saving (ms) | projected tick (ms) | change |
|---|---:|---:|---:|---:|
| 16 sessions, depth 12288 | 173.6 | 11.1 | 162.5 | -6.4% |
| 32 sessions, depth 12288 | 337.4 | 22.3 | 315.1 | -6.6% |
| 32 sessions, depth 2048 | 249.2 | 22.3 | 226.9 | -9.0% |

On the RTX 5090 (9B), the same probe also gives 0.50x from B=8 up.

The Flash-Next mechanism (more prefix-cache blocks) does not apply to this cell. memra's prefix cache already hits 99.6% of prompt tokens in the GB10 agentic cell (budget 52 GB). The gain here is the per-step state bytes, as in the SGLang 27B result.
