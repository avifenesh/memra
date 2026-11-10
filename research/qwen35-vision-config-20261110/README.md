# qwen35-vision-config-20261110

Receipts for memra #1166: the official Qwen3.5-4B safetensors checkpoint binds and serves.

**Verdict: with `qwen3_5` accepted as a `vision_config.model_type`, the unmodified official
Qwen3.5-4B directory binds all 738 tensors and passes `serve-st-gate` (0 failed) on one RTX 5090
Laptop GPU.** On main the same directory refuses before upload with `extra checkpoint tensors`
(the 297 `model.visual.*` tensors).

## What was run

- Checkpoint: `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, BF16 safetensors, 2
  shards, 738 tensors (426 trunk, 15 `mtp.*`, 297 `model.visual.*`), unmodified.
- Binaries: sm_120a, CUDA 13.1, built from this branch on main `42bd4b07e4`
  (`memra-server` sha256 `7d96f377c2e6`, `run-gen` `21e8955f1677`).
- Card: one RTX 5090 Laptop GPU (24 GB), driver 615.71.09. Greedy, one request at a time, N=1
  (identity checks, not timings).
- Gate script: the `tools/serve-st-gate.sh` of memra #1170, whose check 4 compares the default
  (speculative) server with the `MEMRA_SERVE_SPEC=0` server as an equality. Main's own script
  boots the scheduler seam #1170 removes; the two PRs land together.

## Results

`gate-official-dir/gate.log`: `serve-st-gate: 0 failed`.

| check | result |
|---|---|
| bind | `[tensor-contract] bound 738 semantic tensors (pack qwen35, HfSafetensors, ...)` in every boot |
| 1, `/models` | lists the model |
| 2, chat through the checkpoint's template | coherent |
| 3, CLI vs server, greedy | 64 token ids identical |
| 4, spec vs plain, short prompt | 211 tokens, text equal, 64 speculative rounds |
| 4, spec vs plain, window prompt | 400 tokens, text equal, 149 speculative rounds |

These are the numbers the copy without the tower gave (memra
`research/serve-batch-removal-20261110/`): the tower is censused and left unread, and the text
program is the same.

## Fail before, pass after (CPU)

Three tests in `memra-gguf` cover the config parse and the tensor contract. They do not load
weights.

| parser | `red-arm.log` / `green-arm.log` |
|---|---|
| main's matcher (without `qwen3_5`) | 0 passed, 3 failed |
| the fix | 3 passed, 0 failed |

The pack's declared tower was also compared with the checkpoint's safetensors headers: 297 of
297 names and shapes match, all BF16.

## Scope

Text serving. The tower stays side-loaded under `MEMRA_VISION_DIR`, and its executor is fixed to
the Qwen3.8-27B geometry; no Qwen3.5 tower was run and none has a parity receipt
(`docs/decisions/VISION-LANE.md`).

Paths under the run's scratch directory are replaced by `<scratch>`, `<bins>`, `<tree>`,
`<target>` and `<gate-script-repo>`.
