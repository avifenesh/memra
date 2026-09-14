# Native HF input for the argmax-margin gate

Issue #203. The probe now opens HF safetensors directories through the existing native
tensor source and HF tokenizer. GGUF input retains its original loader and tokenizer.
No forward math, numerical threshold, or serving default changes.

The 2026-09-14 integration retains only issue #203's input/placement work and its
historical receipts. The proposed `sampled-instrument` push-hook policy is excluded;
the existing hardware-gate policy remains unchanged. Fresh hosted wrapper, formatter,
placement and compile checks are required after the rebase. No local CI or GPU gate
was run during that integration, and the archived results below are not new-head proof.

The wrapper rejects missing explicit inputs, invalid thresholds, and missing or malformed
measured tables before canary injection. A review found that four-decimal table formatting
could erase a small explained margin; machine-consumed values now preserve f32 round-trip
precision. The shared row formatter has a CPU round-trip test, and the wrapper regression
includes the small explained-flip case.

Historical CPU validation: nine Python test methods with HF/GGUF, negative-input, malformed-table,
and canary controls; two Rust tests for formatting and placement refusal; cargo check;
formatting and diff checks.
The wrapper and Rust formatter/placement tests run in hosted CI. Raw CPU outputs are adjacent.

Real-checkpoint execution passed for the GLM 4o6 case below. This gate compares prefill against serial decode
on identical teacher-forced prompt positions. It does not qualify sampled serving,
batched decode, speculative execution, or every hardware placement. A directory loading
successfully is not a checkpoint-parity receipt.

The serial decode cache now uses native pipeline placement, matching `run-gen`.
Sharded cross-device input is refused for trunks without a pipeline `forward_last`
dispatch. The hyper-connection trunk has that dispatch and remains eligible. This
restriction avoids silently replacing the measured prefill arithmetic.

## Real checkpoint receipt

`glm4o6-margin-table.txt` preserves all twelve measured tail-position rows from
`tiyuvta/GLM-5.3-Flash-NVFP4-4o6@b07bf78ff924a86df8361d4074150e0083f81778`,
source `b9749ae4e9b21ea771cd07a0a6f65a47a4a0a088`, on three RTX PRO 6000 Blackwell
cards. Eleven positions agreed; the remaining flip was explained by its top-two margin
being smaller than the measured configuration spread, within the unchanged one-flip budget.
This is a numeric correctness receipt, not evidence for sampled performance or every
pipeline/batched path. The measurement excerpt labels its toolchain Rust 1.97.1 / CUDA
13.0, while the separate local compile and formatter logs record
`/usr/local/cuda-13.1/bin/nvcc`. The excerpt has no accompanying compiler-version output
that resolves the measurement-build toolkit; its label is retained as historical
provenance, not asserted as independently verified. Any fresh model run must record
the actual compiler version and binary/source hashes together.

The recipe used `CUDA_VISIBLE_DEVICES=1,2,3`, `NVIDIA_TF32_OVERRIDE=0`,
`MEMRA_PP_STAGES=3`, `MEMRA_PP_SPLITS=15,30`, `MEMRA_PP_DEVICES=0,1,2`,
`MEMRA_MOE_RESIDENT_GB=98`, `MEMRA_PP_BF16=1`, `MEMRA_BF16_MMV=1`,
`MEMRA_MOE_GROUPED_PREFILL=1`, and `MEMRA_MLA_TC_PREFILL=1` under the canonical
GPU lock, with `research/e2e/prompts/board-2048.txt` and `--window 12`.
