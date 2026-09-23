# Bound Step access fixtures

These three CPU tests are #541 integration coverage, separate from the dependency-free Step
plan/schema corrections in `944516e44` and `ef131c82f`:

- Native FP8 stacked-bank views, private MTP norms/heads, attention gate names and all tested
  norm folds match the existing HF materializer; MTP remains dense.
- Wrong bank/head shapes fail during complete tiny-census binding.
- Absent private MTP heads retain the planned model-head path, and GGUF folds remain identity.

All three pass. Raw log and test-source hash are retained. The fixture is a synthetic text
checkpoint, not the complete official artifact, a CUDA run or vision qualification.
Universal activation still requires the scoped-census architecture decision, exact remaining
source/derived-view coverage, review and native qualification.
