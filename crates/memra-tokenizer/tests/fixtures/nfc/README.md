# Independent NFC input-program fixtures

Captured with Hugging Face tokenizers0.22.2 before the implementation, from the frozen Qwen2 fixture at fdb781362c8366e11db492cba14bfe1dc6bb6941. The31 variants contain315 encode cases; the special-mode matrix independently crosses special recognition and BOS/postprocessing controls. The22 imports include required identity/NFC, optional NFC-only Sequence wrappers and programs that this implementation must refuse.

Token IDs and import acceptance are the regression predicates. Stored offsets, decoded strings and masks are diagnostics, not added decoder/offset requirements. Default GGUF normalization stays identity; family names never enable NFC. The research record retains generator/oracle provenance and the separate real-vocabulary551-case capture.
