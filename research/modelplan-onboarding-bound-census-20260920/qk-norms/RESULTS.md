# Full-attention q/k norm convention

The Qwen3.5 bound-access regression reproduced a compiler/materializer mismatch before activation:
for a stored norm value of 0.25, the bound q/k view returned 0.25 while the existing native HF
loader correctly returned 1.25. The layer already declares the centered weight convention, but
the generic full-attention tensor contract hardcoded identity for q/k norms.

The contract now carries the block's declared norm weight transform into its full-attention q/k
requirements. The existing native numerical program is preserved. Positive controls verify that
ordinary Qwen3 norms remain 0.25 and pre-folded GGUF q/k weights retain identity transforms.
Step's isolated A/B text corrections are unaffected: B already explicitly declares its q/k fold.

The new centered-norm regression fails before the repair and passes after it. Both positive
controls pass, the GGUF/CLI full suite passes, and Clippy all targets passes with warnings denied.
Raw logs and source hashes are retained. No native run, universal activation, support promotion
or permission to reuse old binary/binding receipts is implied.
