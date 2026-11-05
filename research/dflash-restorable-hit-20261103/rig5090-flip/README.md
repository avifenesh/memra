# Restore doors on by device default: RTX 5090 Laptop served A/B (2026-11-05)

Verdict: on the 24 GB RTX 5090 Laptop, doors on beats doors off in every cell both arms complete: c1 +7.6% in a 2+2 ABBA with no overlap, c2 +13.3% on the shorter-prompt subset. Doors on also hits the card's memory limit at a lower concurrency. With doors on, the full workload's 17-18k-token prompts are refused (HTTP 400) from c2, and the subset gets an HTTP 408 at c4. Doors off serves both of those cells, and both arms fail at c8.

## Setup

- Card: RTX 5090 Laptop GPU, 24 GB, platform profile `balanced`. Every arm ran at about 80 C and 900 MHz SM under the software power cap (`gpu-250ms.csv`, 250 ms samples). The host load average was 10-23 throughout from other sessions' work, the same for every arm.
- One binary for every arm: the flip build 6561499516 (sm_120a, sha256 524234e77bbb), built for the 5090 exactness gate (`../cold-drafter-gate/5090-flip-6561/`).
  - DEF: the three restore doors unset. The boot line reads `default ON on "NVIDIA GeForce RTX 5090 Laptop GPU" ... restore=true partial=true capture=true`.
  - OFF: all three doors `0`. The boot line reads `default ON ... restore=false partial=false capture=false`.
  - Both arms run with `MEMRA_PREFIX_STABLE_BOUNDARY=0`.
- Harness: the earlier 5090 cells of this lane (`../cold-drafter-ab-5090.sh`):
  - Qwen3.8-27B NVFP4+Q5_K GGUF with the DFlash2 q4 drafter, FR-Spec trim and `MEMRA_MTP_SKIP=1`;
  - context 32,768 and a 4,000 MB prefix cache;
  - T=1 seed 1234, thinking on, `max_tokens` 768;
  - sessions raised from 1 to 8 for the c2..c8 cells.
- Fresh boot per arm, bench client on the same host (`rig5090_ab.sh`, `rig5090_conc.sh`, log `chain.log`).
- Workloads:
  - **full**: the 29 agentic bodies that fit 24 GB (sha256 5fe1afe2afdc3db0). The prompts run 1.7k to 17.9k tokens.
  - **le8k**: the 22 shortest of those 29 by message length, prompts up to about 8.1k tokens (sha256 331f269100d2). It was added after the full workload's c2..c8 cells failed. The workload files stay out of the repo; they are derived from private agent sessions.

## Results (aggregate generated tok/s)

| cell | DEF (doors on by default) | OFF (all doors 0) | DEF vs OFF |
|---|---:|---:|---:|
| c1, full, ABBA order DEF/OFF/OFF/DEF | 22.08 / 22.07 | 20.53 / 20.51 | **+7.6%** |
| c2, full | refused: 2 requests HTTP 400 `context_length_exceeded` (17.5k and 17.9k prompts) | 20.92 | DEF fails |
| c4, full | refused (HTTP 400) | refused (HTTP 400, a 17.2k prompt) | both fail |
| c8, full | refused (HTTP 400) | refused (HTTP 400) | both fail |
| c2, le8k | **29.34** | 25.89 | **+13.3%** |
| c4, le8k | HTTP 408 (`deadline_exceeded` before headers, queued behind VRAM defers) | 29.76 | DEF fails |
| c8, le8k | refused (HTTP 400) | HTTP 408 | both fail |

Receipts: `1-def/`, `2-off/`, `3-off-c1/` and `4-def-c1/` hold the full workload in run order; `5-def-le8k/` and `6-off-le8k/` hold the le8k subset. Each holds `arm.txt`, `cells.jsonl`, `cells.log`, `serve.log` and `gpu-250ms.csv`.

The completed cells generated the same token counts in both arms: 10,818 at c1 in all four runs, 7,052 at c2 le8k. The doors change speed, not the output.

## Why doors on fails earlier on 24 GB

The 400s come from the idle-box capacity check (`[admit-oom] capacity reject: ... does not fit an IDLE box`); the 408s come from requests that queued behind VRAM defers past their deadline.

With doors on, each request pays more memory:
- a 169 MB DFlash draft-state charge;
- a measured fixed-residual high-water of 493 MB against 340 MB with doors off;
- prefix entries that carry draft tails.

For the 17.9k-token body at c2 that comes to:
- doors on: 3,424 MB plus the 1,619 MB reserve, against 4,828 MB available, so the request is refused;
- doors off: 3,101 MB plus the reserve fits.

On the le8k subset, the doors-on residual high-water rose to 1,057 MB at c4.

The doors-on boots logged 14 (`1-def/`, every cell) and 11 (`4-def-c1/`) DSPARK restores and no cold-drafter conversion. The full workload logs no `cache-preferred` veto on this card, so the c1 gain comes from the restore path, not from the #1031 conversion.

## Reading

- On 24 GB at this context, doors on is faster wherever the card has room, and it runs out of room one step of concurrency sooner.
- The flip keys on the device name containing "RTX 5090", which also matches the 32 GB desktop card. That card was not measured.
- Every number above is one run per cell except the c1 ABBA.
