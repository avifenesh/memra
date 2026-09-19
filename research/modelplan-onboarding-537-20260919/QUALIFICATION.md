# Issue 537 qualification runner — 2026-09-20

The coordinating task provisions the non-serving host. This task does not rent another box.
No GPU phase has run yet. Execute GPU commands only through the supplied `memra-gpu-run`
wrapper after the coordinator assigns physical GPUs and the task's remote checkout.

The current owner instruction requires per-card exclusive locks and supersedes the older
blanket per-rig rule for this campaign. The runner requires `MEMRA_GPU_LEASE_FILE`, checks
both recorded PIDs against its ancestor chain, verifies the wrapper's actual exclusive
FLOCKs using `/proc/locks` and each file's device/inode, and checks the complete physical UUID
set, visibility order and sorted acquisition order. It refuses extra, missing or aliased GPU
locks. Canonical paths are `/tmp/memra-gpu-locks/<GPU-UUID>.lock`.

## Capacity and immutable artifacts

| Stage | Exact card set used | Target | Artifact |
|---|---:|---|---|
| `focused` | 1 | RTX PRO 6000 Blackwell, 96 GiB | Synthetic fixture; no download |
| `dense` | 1 | Same | Qwen3-1.7B Q8_0, real dense-loader/kernel battery |
| `ornith`, `serve-ornith` | 1 per stage | Same | Owner's Ornith NVFP4+MTP control |
| `step-gguf` | 2 | 2× RTX PRO 6000 Blackwell | IQ4_XS trunk + Q8_0 MTP compatibility arm |
| `step-fp8` | 3 | 3× RTX PRO 6000 Blackwell | Official FP8 safetensors, including its MTP |

Request **3×96 GiB GPUs, 256 GiB host RAM (512 GiB preferred), 16+ vCPUs and 1 TiB usable
local NVMe** for the complete campaign. The two-card set cannot hold the complete official
FP8 artifact even before workspace/KV overhead. Host RAM and disk are provisioning budgets;
actual peaks have not been measured. The focused gate itself needs only one card and a tiny
fixture, but cannot establish checkpoint or release qualification.

The 96 GiB Max-Q variant is suitable for these correctness stages. Record its exact model,
power limit, clocks and topology; no timing result transfers to a full-power Server card.

The selected downloads total **343,257,724,226 bytes**. Every file's exact length and SHA256,
including small configuration/tokenizer files, is in [artifacts.lock.json](artifacts.lock.json).

| Role | Repository at immutable revision | Selected files / bytes |
|---|---|---:|
| `step_fp8` | `stepfun-ai/Step-3.7-Flash-FP8@b3d7916fccac844cca050d7520f2aaa513f9a84f` | All 39 files / 212,534,420,770 |
| `step_gguf` | `stepfun-ai/Step-3.7-Flash-GGUF@0b69336d2fd2adfdef9c66e425f7778196c31482` | IQ4_XS 3 shards + `Step3.7-flash-mtp-Q8_0.gguf` / 108,700,839,040 |
| `ornith` | `Avifenesh/Ornith-1.5-35B-A3B-NVFP4-MTP-GGUF@e058c9f5bcce9234deffc87636bbe7fc61e2b8c5` | `Ornith-1.5-35B-A3B-NVFP4-Q5K-mtp.gguf` / 20,188,038,400 |
| `dense` | `Qwen/Qwen3-1.7B-GGUF@90862c4b9d2787eaed51d12237eafdfe7c5f6077` | `Qwen3-1.7B-Q8_0.gguf` / 1,834,426,016 |

The GGUF arm never substitutes for the official FP8 arm. Secondary hardware can collect
diagnostics using its matching build arch; its receipts say `secondary-backend diagnostic only`
and do not satisfy the RTX PRO 6000 target gate.

## Commands

Prerequisites: Linux, Python 3, Hugging Face `hf` CLI, Rust 1.97.1+, CUDA 13.1+ with the matching
driver and `nvidia-smi`, normal repository build dependencies. Run from this task's clean,
committed remote checkout. Receipt directories must be new and outside the checkout.

Download/build **outside** GPU leases. Durable artifacts remain under `/data`; stage identical
bytes to local NVMe under `/scratch`. Each GPU model phase rechecks all selected file hashes.

```bash
python3 research/modelplan-onboarding-537-20260919/qualify.py fetch \
  --models /data/q537-models --out /data/q537-receipts/fetch
rsync -a /data/q537-models/ /scratch/q537-models/
python3 research/modelplan-onboarding-537-20260919/qualify.py build \
  --arch 120a --jobs 4 --out /scratch/q537-receipts/build
```

Set `Q537_GPU0`, `Q537_GPU1`, `Q537_GPU2` to the coordinator-assigned full physical UUIDs.
The wrapper sets visibility in that order; runtime PP device numbers are logical indices.
No compilation occurs inside a lease, and each stage is one foreground runner session.

```bash
for q537_phase in focused dense ornith serve-ornith; do
  memra-gpu-run --gpus "$Q537_GPU0" --receipt "/scratch/q537-leases/$q537_phase" -- \
    python3 research/modelplan-onboarding-537-20260919/qualify.py "$q537_phase" \
      --models /scratch/q537-models --build-record /scratch/q537-receipts/build/build.json \
      --out "/scratch/q537-receipts/$q537_phase"
done

memra-gpu-run --gpus "$Q537_GPU0,$Q537_GPU1" --receipt /scratch/q537-leases/step-gguf -- \
  python3 research/modelplan-onboarding-537-20260919/qualify.py step-gguf \
    --models /scratch/q537-models --build-record /scratch/q537-receipts/build/build.json \
    --out /scratch/q537-receipts/step-gguf

memra-gpu-run --gpus "$Q537_GPU0,$Q537_GPU1,$Q537_GPU2" --receipt /scratch/q537-leases/step-fp8 -- \
  python3 research/modelplan-onboarding-537-20260919/qualify.py step-fp8 \
    --models /scratch/q537-models --build-record /scratch/q537-receipts/build/build.json \
    --out /scratch/q537-receipts/step-fp8
```

All model runs use the committed `research/e2e/prompts/board-2048.txt` prompt and 64 generated
tokens. This crosses Step's 512-token window. `run-gen` requires argmax MATCH; `run-spec`
requires every K=1..8 self-consistency arm to pass. The dense phase runs `kernel-check` with
ALL GREEN required. The separate Ornith HTTP smoke binds loopback port 15379, verifies the
listener belongs to its own server PID, checks a real completion, and terminates its owned
server. It stays in the wrapper-supervised process group for cancellation. It can use `--port`
if the coordinator assigns another free port.

The focused test checks the actual uploaded factor buffer, bit-identical decode with the
same factors supplied through config or tensor bytes, the full-head GGUF storage extent,
and non-vacuity against an omitted-factor diagnostic mutation. Its uniform Q8_0 synthetic
expert banks are not evidence for any checkpoint quantization format.

The runner stores raw stdout/stderr before reading verdicts, argv, exit status, source commit,
binary and log hashes, artifact hashes, physical lease metadata, GPU state, and other-card
compute activity. `PASS.json` is a phase receipt, never a support-state promotion. Preserve
these logs and complete any further model/serve/rewrite gates required by the combined lane.

## Preparation findings

The original strict factor guard expected 32 elements from the partial rotary consumer.
The banked official GGUF headers in `research/step37-bringup-20260802/raw/` explicitly record
`rope_freqs.weight [64] F32` for both trunk and MTP. The pack now declares that full-head
storage shape and validates/preserves either the compact consumer vector or the full-head
vector, rejecting short/intermediate extents and invalid values. All bytes are retained;
the kernel consumes its declared prefix. The regression fails before this correction in
[full-head-before.log](raw/full-head-before.log); the current CPU result is recorded separately
from the original PR evidence in [qualification-prep-cpu.log](raw/qualification-prep-cpu.log).
