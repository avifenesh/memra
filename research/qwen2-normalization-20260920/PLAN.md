# Declared NFC normalization (#554)

Base: merged PR558 at fdb781362c8366e11db492cba14bfe1dc6bb6941. The loader currently omits tokenizer.json.normalizer and the normalized flag for added tokens. Preserve the declared normalization program, apply unnormalized added-token matching before normalization and normalized matching afterwards, and reject unsupported declarations explicitly.

The independent HF tokenizers0.22.2 oracle uses Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775. Raw inputs currently mismatch250/551 cases; normalized inputs match551/551 in both loaders/modes. The gzip baseline capsule is the exact approved source `67fce3a9` native record published in PR558, SHA256 dcbff73a25ea524fa2185f7474f7ebb954fd1635a9191358831c0e2537b709ca. It remains historical evidence, not qualification of this upcoming change.

An independent reviewer is capturing added-token ordering/normalization oracle cases before implementation-dependent tests. No normalizer should be inferred merely from a model family or from a GGUF lacking a declaration. Existing Qwen2 literal split, Qwen3.5 and GLM controls must remain unchanged for undeclared normalization.

Implementation and native validation are pending. The frozen tokenizer artifact directory is retained outside this worktree in the coordinator's tokenizer-baselines folder.
