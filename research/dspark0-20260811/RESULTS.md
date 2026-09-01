# DSpark Phase 0 attach-parity result

Date: 2026-08-11

Lane: `lane/cx-dspark0`

Rig: local NVIDIA GeForce RTX 5090 Laptop GPU

## Verdict

**GREEN — Phase 2 pilot is unblocked.**

The Qwen3.5 9B model's own MTP head survived the hqmtp safetensors-to-GGUF
trained-head export route and attached through memra's external-drafter seam without
the historical 35–39% acceptance collapse. Across the frozen three-prompt K=3 suite,
the byte-verbatim reference accepted 263/369 drafted tokens (71.274%) and the
converter artifact accepted 262/372 (70.430%), a **-0.844 percentage-point delta**.
Both artifacts passed `run-spec` self-consistency for every K from 1 through 8.

Within the scope of this gate, the old converter-collapse mystery is therefore
localized to head content or prior artifact construction, not the current external
attach path. This result does not identify the defective content in any older
artifact.

## Frozen comparison

Both arms used:

- target model:
  `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`
- external attachment through `MEMRA_MTP_DRAFT`
- `MEMRA_CHAT=1`, `MEMRA_SPEC_TEMP=0`, `MEMRA_SPEC_K=3`, and
  `MEMRA_NGEN=128`
- prompts `p1-code-short`, `p2-code-medium`, and `p3-agentic-long-v3`
- release `run-spec` binary
- exclusive execution under `flock /tmp/memra-gpu.lock`

The two external artifacts were:

| arm | construction | SHA-256 |
|---|---|---|
| reference | `tools/extract_mtp_draft.py` byte extraction from the serving GGUF | `fe6301c4ce6bc07dd08603d7aa32cd119ee3d4161c71143de576742850b52cdc` |
| converter | pinned HF safetensors through hqmtp `export_frtrim.py`, retaining all 248,320 vocabulary ids in identity order | `806aac1725b9eaf5fab4b7d9049b559bc9e4da38e0a37f567a0fea482b772a83` |

The target SHA-256 is
`52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
The hqmtp source revision is
`9b19cb99cc2e6b2b149819be82f340b8d6b563cd`; source shard, converter,
prompt, runner, and binary hashes are retained in `raw/`.

## Tensor receipt

After the expected standalone-draft block rename (`blk.32` to `blk.0`), all 15
shared MTP block tensors have identical types, shapes, and payload SHA-256 hashes.
That includes all eight Q8_0 matrices and all seven folded F32 tensors.

The files are not wholly byte-identical. The reference carries a full-vocabulary
Q6_K `output.weight`; the converter carries a full-vocabulary Q8_0
`output.weight` plus an identity `draft_to_target` map. The reference also retains
carrier tensors that the external-draft loader does not consume. Thus the runtime
test is the decisive attach-path comparison; the tensor receipt shows that the MTP
block itself did not drift in conversion.

## Matched K=3 acceptance

| prompt | byte-verbatim reference | converter | exact delta | self-consistency |
|---|---:|---:|---:|---|
| p1 code-short | 84/135 = 62.2% | 83/138 = 60.1% | -2.077 points | PASS / PASS |
| p2 code-medium | 93/111 = 83.8% | 93/111 = 83.8% | 0.000 points | PASS / PASS |
| p3 agentic-long-v3 | 86/123 = 69.9% | 86/123 = 69.9% | 0.000 points | PASS / PASS |
| **frozen-suite aggregate** | **263/369 = 71.274%** | **262/372 = 70.430%** | **-0.844 points** | **PASS / PASS** |

The p1 cell alone is 0.077 point outside a literal two-point band when calculated
from unrounded counts; the displayed one-decimal rates differ by 2.1 points. This is
not hidden in the aggregate verdict. The two longer frozen cells match exactly, and
the predeclared suite aggregate is inside the Phase 0 parity band.

## K=1 through K=8 exactness

The exactness battery used the frozen p1 prompt and 128 generated tokens per K.

| K | reference acceptance | converter acceptance | result |
|---:|---:|---:|---|
| 1 | 61/67 = 91.0% | 61/67 = 91.0% | PASS / PASS |
| 2 | 77/102 = 75.5% | 76/104 = 73.1% | PASS / PASS |
| 3 | 84/135 = 62.2% | 83/138 = 60.1% | PASS / PASS |
| 4 | 91/152 = 59.9% | 90/156 = 57.7% | PASS / PASS |
| 5 | 92/185 = 49.7% | 91/190 = 47.9% | PASS / PASS |
| 6 | 92/222 = 41.4% | 91/228 = 39.9% | PASS / PASS |
| 7 | 94/245 = 38.4% | 93/252 = 36.9% | PASS / PASS |
| 8 | 94/280 = 33.6% | 93/288 = 32.3% | PASS / PASS |

Both logs terminate with `=== SELF-CONSISTENCY PASS ===`; every speculative output
was identical to the plain target output.

## Optional controls and limits

- No local StudentSV checkpoint or exported StudentSV GGUF was present under the
  searched hqmtp and model roots, so the plan's optional trained-student arm was not
  runnable on this machine.
- A non-gating attempt to match the output projection encoding exactly did not
  produce a Q6_K converter artifact. gguf-py lacks a Q6_K encoder; after routing
  through native llama.cpp and supplying the missing Qwen3.5 metadata, the native
  quantizer aborted before tensor conversion on
  `n_layer_nextn < n_layer_all` because the standalone draft carrier represents one
  NextN layer as one total layer. The raw logs quote each failure. This optional
  control is inconclusive and does not alter the successful required A/B.
- Each acceptance cell is a single deterministic run on the shared local rig. The
  timings are not an N-run thermal protocol and support no throughput claim.
- This lane changed no memra runtime or kernel source. It adds only the research
  runners, receipts, raw logs, and verdict, so no board or release surface moves.

## Evidence index

- artifact and source hashes: `raw/source-artifact-sha256.txt`
- runner, prompt, and binary hashes: `raw/protocol-sha256.txt`
- tensor headers and payload hashes: `raw/reference-header.json`,
  `raw/converter-header.json`, `raw/header-comparison.json`, and
  `raw/tensor-byte-hashes.txt`
- matched raw runs: `raw/parity-p{1,2,3}-{reference,converter}.log`
- K=1..8 raw runs: `raw/k1-8-{reference,converter}.log`
- optional Q6-control failures: `raw/converter-export-q6-*.log`
- build receipt: `raw/build-run-spec.log`
