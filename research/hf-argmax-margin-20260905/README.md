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
and canary controls; one Rust formatter test; cargo check; formatting and diff checks.
The wrapper tests run in hosted CI. Raw CPU outputs are adjacent.

Real-checkpoint execution is pending. This gate compares prefill against serial decode
on identical teacher-forced prompt positions. It does not qualify sampled serving,
batched decode, speculative execution, or every hardware placement. A directory loading
successfully is not a checkpoint-parity receipt.
