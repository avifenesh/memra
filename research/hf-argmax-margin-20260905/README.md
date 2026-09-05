# Native HF input for the argmax-margin gate

Issue #203. The probe now opens HF safetensors directories through the existing native
tensor source and HF tokenizer. GGUF input retains its original loader and tokenizer.
No forward math, numerical threshold, or serving default changes.

The wrapper rejects missing explicit inputs, invalid thresholds, and missing or malformed
measured tables before canary injection. A review found that four-decimal table formatting
could erase a small explained margin; machine-consumed values now preserve f32 round-trip
precision. The shared row formatter has a CPU round-trip test, and the wrapper regression
includes the small explained-flip case. Five independent review passes are clean after
that correction.

CPU validation: nine Python test methods with HF/GGUF, negative-input, malformed-table,
and canary controls; two Rust tests for formatting and placement refusal; cargo check;
formatting and diff checks.
The wrapper tests run in hosted CI. Raw CPU outputs are adjacent.

Real-checkpoint execution passed for the GLM 4o6 case below. This gate compares prefill against serial decode
on identical teacher-forced prompt positions. It does not qualify sampled serving,
batched decode, speculative execution, or every hardware placement. A directory loading
successfully is not a checkpoint-parity receipt.

The serial decode cache now uses native pipeline placement, matching `run-gen`.
Sharded cross-device input is refused for trunks without a pipeline `forward_last`
dispatch. The hyper-connection trunk has that dispatch and remains eligible. This
restriction avoids silently replacing the measured prefill arithmetic. A targeted
review of the placement correction is clean.

## Real checkpoint receipt

`glm4o6-margin-table.txt` preserves all twelve measured tail-position rows from
`tiyuvta/GLM-5.3-Flash-NVFP4-4o6@b07bf78ff924a86df8361d4074150e0083f81778`,
source `b9749ae4e9b21ea771cd07a0a6f65a47a4a0a088`, on three RTX PRO 6000 Blackwell
cards. Eleven positions agreed; the remaining flip was explained by its top-two margin
being smaller than the measured configuration spread, within the unchanged one-flip budget.
Rust 1.97.1 and CUDA 13.0 built the probe. This is a numeric correctness receipt, not
evidence for sampled performance or every pipeline/batched path.

The recipe used `CUDA_VISIBLE_DEVICES=1,2,3`, `NVIDIA_TF32_OVERRIDE=0`,
`MEMRA_PP_STAGES=3`, `MEMRA_PP_SPLITS=15,30`, `MEMRA_PP_DEVICES=0,1,2`,
`MEMRA_MOE_RESIDENT_GB=98`, `MEMRA_PP_BF16=1`, `MEMRA_BF16_MMV=1`,
`MEMRA_MOE_GROUPED_PREFILL=1`, and `MEMRA_MLA_TC_PREFILL=1` under the canonical
GPU lock, with `research/e2e/prompts/board-2048.txt` and `--window 12`.
