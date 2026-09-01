# DSpark Phase 0 progress

Lane: `lane/cx-dspark0`

Rig: local RTX 5090

Opened: 2026-08-11

## Gate

Round-trip the Qwen3.5 9B model's own MTP head through the trained-head export path,
attach it as an external drafter, and compare in-engine acceptance against a
byte-verbatim extraction of the same head. The model, prompts, sampling settings,
and K must be identical. Parity is green only when acceptance differs by at most
2 percentage points and `run-spec` passes for K=1..8.

## Rules

- Guard every GPU command with `flock /tmp/memra-gpu.lock`.
- Capture raw stdout and stderr before parsing; do not infer failure causes.
- Record source, artifact, converter, runtime, prompt, and configuration hashes.
- Do not run `cargo fmt`.

## Status

- [x] Read the lane brief, Phase 0 plan, and project instructions.
- [x] Locate and hash the 9B target and native MTP artifacts.
- [x] Freeze the matched A/B protocol and prompts.
- [x] Produce the byte-verbatim reference drafter.
- [x] Round-trip the same tensors through the trained-head export path.
- [x] Compare tensor manifests before GPU execution.
- [x] Run matched in-engine acceptance A/B.
- [x] Run the `run-spec` K=1..8 exactness gate.
- [x] Record the final parity verdict in `RESULTS.md`.

## Current note

The target is
`/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`
(`sha256:52c9cceb190055e0...`). The two external arms are:

- `reference-byte-verbatim.gguf` (`sha256:fe6301c4ce6bc07d...`), extracted from
  the serving GGUF by `tools/extract_mtp_draft.py`.
- `converter-q8-full.gguf` (`sha256:806aac1725b9eaf5...`), exported from the
  pinned HF safetensors by hqmtp `export_frtrim.py` at commit `9b19cb99...`, with
  all 248,320 vocabulary ids retained in identity order.

All 15 shared MTP block tensors are byte-identical after the expected block-index
rename (`blk.32` to `blk.0`), including every Q8 matrix and every folded F32 norm.
The output projection is the one tensor with a dtype difference: reference Q6_K,
converter Q8_0; both are full-vocabulary, and converter `d2t` is the identity map.

Frozen A/B: greedy temperature 0, chat template on, K=3, `MEMRA_NGEN=128`, prompts
`research/e2e/prompts/{p1-code-short,p2-code-medium,p3-agentic-long-v3}.txt`.
Run the same command shape for each arm under the GPU flock, then run the full
K=1..8 self-consistency gate on both artifacts.

## Matched acceptance cells

| prompt | reference | converter | delta | exactness |
|---|---:|---:|---:|---|
| p1 code-short | 84/135 = 62.2% | 83/138 = 60.1% | -2.08 points | PASS / PASS |
| p2 code-medium | 93/111 = 83.8% | 93/111 = 83.8% | 0.00 points | PASS / PASS |
| p1+p2 aggregate | 177/246 = 72.0% | 176/249 = 70.7% | -1.27 points | PASS / PASS |
| p3 agentic-long | 86/123 = 69.9% | 86/123 = 69.9% | 0.00 points | PASS / PASS |
| frozen-suite aggregate | 263/369 = 71.27% | 262/372 = 70.43% | **-0.84 points** | PASS / PASS |

The p1 converter cell is 0.08 point beyond the literal two-point band using the
unrounded counts. The frozen p2/p3 cells remain required before the aggregate verdict;
no cause is assigned from one prompt. Both longer cells match exactly, and the frozen
suite aggregate is inside the Phase 0 band. The acceptance-parity arm is green; the
K=1..8 exactness battery passes on both artifacts. No local StudentSV checkpoint or
pre-exported StudentSV GGUF was present under `/data/projects/hqmtp` or `/data/ai-ml`,
so the plan's optional trained-student arm was not runnable on this machine.

An optional projection-matched control attempted to rebuild the exporter arm with Q6_K
`output.weight`, matching the reference's last differing tensor encoding. It did not
produce an artifact. The Python path stopped before artifact creation because gguf-py has
no Q6_K encoder. The first native llama.cpp path stopped before tensor conversion because
the minimal hqmtp metadata omitted `qwen35.rope.dimension_sections`. After the carrier was
revised to inherit missing Qwen3.5 metadata from the pinned serving GGUF, the native
quantizer parsed the carrier and then aborted before tensor conversion on its standalone
draft invariant: `n_layer_nextn < n_layer_all` with both values equal to one. All three
failures are captured verbatim. This optional encoding-matched control is therefore
inconclusive and does not change the completed Phase 0 attach-parity verdict.
