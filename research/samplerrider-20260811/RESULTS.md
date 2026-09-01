# Sampler rider result

## Correctness

The decode epilogue now computes argmax for all `B` rows in one launch while preserving the
selected tokens byte-for-byte. The committed 5090 exactness battery is GREEN: the configured
`decode-batch-gate` matrix at B=1/2/4/8 and strict B=4 passed for greedy, sampled, and mixed-meta
cases; `kernel-check` reported 106 GREEN cells; `run-gen` reported argmax MATCH; and `run-spec`
passed K=1..8 self-consistency. Receipts are under `raw/exactness/`.

## Timing

Directional decode timing used the exactness model and its fixed 90-token prompt:

- Model: `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`
- Candidate: the `d838cecd` `run-gen` build, SHA-256 `b4a3a6c058a66234589865cda36d3c1b547b056c59ebb348adae5201d7a2744d`
- Baseline: `run-gen` built separately from `41033147`, SHA-256 `9ffe43eb62d52eda48e53828fcaf5631308efe5fbdc82d7bb692e47e373f9a29`
- Protocol: `MEMRA_FAST=1`, `MEMRA_NGEN=128`, `nice -n 15 ionice -c3`; one discarded warmup per arm, then N=5 separate launches interleaved candidate/baseline in one thermal window
- Candidate decode tok/s: **140.59, 140.05, 139.84, 139.70, 139.36**; median **139.84 tok/s**; min..max spread **1.23 tok/s (0.88%)**
- Baseline decode tok/s: **140.30, 139.90, 139.97, 139.56, 139.32**; median **139.90 tok/s**; min..max spread **0.98 tok/s (0.70%)**
- Median delta: **-0.04%** candidate versus baseline. Median decode wall time rounds to **0.915 s** for both arms.

The RTX 5090 Laptop GPU was stock and unlocked. The valid window began idle at 57 C, P8,
180 MHz graphics/SM, 405 MHz memory, and 9.89 W; it ended at 78 C, P0, 1905 MHz graphics/SM,
14001 MHz memory, and 171.76 W. Compute-process samples were empty before the window, between
every timed launch, and after the window. No concurrent GPU process appeared during the valid
window.

Raw log: `raw/timing/decode-ab-n5.log`.

## Verdict: NO-GO

The launch count is lower and exactness is clean, but the -0.04% median movement is far inside
the 0.70-0.88% per-arm run-to-run spread. The batched argmax is correct-but-flat for B=1 `run-gen`
decode wall time and does not justify promotion on this directional rider.
