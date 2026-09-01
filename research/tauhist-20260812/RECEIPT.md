# Tau histogram RTX 5090 reference receipt

Date: 2026-08-12

Lane: `lane/cx-tauhist`

Source: `85da378bcdcb80d951d38fb1f24cecdaa52cd168`

## Verdict

**PASS.** The Qwen3.5 9B NVFP4 artifact loaded its embedded MTP head (`nextn=1`), and
`run-spec` produced eight identical-to-target streams for K=1 through K=8. The independent
`run-gen` spot-check reported both prefill/decode argmax `MATCH` and batched-prime/tokenwise
argmax `MATCH`.

Tau here is accepted draft-prefix tokens divided by verification rounds. It excludes the target
bonus token. Position arrays are zero-based and carry both the offered denominator and accepted
count.

## Reference histogram

| K | rounds | accepted / drafted | tau | accepted by position | accept rate by position |
|---:|---:|---:|---:|---|---|
| 1 | 67 | 61 / 67 | 0.910448 | `[61]` | `[0.910]` |
| 2 | 51 | 77 / 102 | 1.509804 | `[45, 32]` | `[0.882, 0.627]` |
| 3 | 45 | 84 / 135 | 1.866667 | `[37, 26, 21]` | `[0.822, 0.578, 0.467]` |
| 4 | 38 | 91 / 152 | 2.394737 | `[34, 25, 21, 11]` | `[0.895, 0.658, 0.553, 0.289]` |
| 5 | 37 | 92 / 185 | 2.486486 | `[34, 24, 19, 11, 4]` | `[0.919, 0.649, 0.514, 0.297, 0.108]` |
| 6 | 37 | 92 / 222 | 2.486486 | `[34, 24, 18, 10, 4, 2]` | `[0.919, 0.649, 0.486, 0.270, 0.108, 0.054]` |
| 7 | 35 | 94 / 245 | 2.685714 | `[32, 23, 19, 11, 5, 2, 2]` | `[0.914, 0.657, 0.543, 0.314, 0.143, 0.057, 0.057]` |
| 8 | 35 | 94 / 280 | 2.685714 | `[32, 23, 19, 11, 4, 2, 2, 1]` | `[0.914, 0.657, 0.543, 0.314, 0.114, 0.057, 0.057, 0.029]` |

The offered array is `[rounds; K]` for every row because this fixed-K receipt did not enable a
draft confidence cutoff. Exact machine-readable counts and full-precision ratios are in
`histogram.json`.

## Provenance

- GPU: NVIDIA GeForce RTX 5090 Laptop GPU,
  `GPU-1a3cbffc-29df-926c-df5c-29b4c210ef5d`, driver 595.84.
- Model: `/data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf`,
  SHA-256 `52c9cceb190055e0591a9a30c21f7200572eaf3ff1c59f6e9a1eda838a8f39de`.
- `run-spec` SHA-256: `da005440c2faa8479246b0376acc07235cb47d3ff2e90c78df1ea84269e29707`.
- `run-gen` SHA-256: `b973bffdb0f2a730648b0b5a2db469014e7450a02947ecc628b42b1089b0a7a4`.
- Prompt: `research/e2e/prompts/p1-code-short.txt`, SHA-256
  `6e00d76296069277dc7717115f977aedcab502b610c95a042c63c30eefdb86b2`; chat-templated to
  37 tokens; `MEMRA_NGEN=128` for the K sweep and 32 for the argmax spot-check.
- Toolchain: rustc 1.97.1, cargo 1.97.1, sm_120a release build.

The K sweep was a single exactness run from 68 C to 82 C. A scheduled ColBERT index refresh held
1,392 MiB in a foreign CUDA context and showed 0% utilization at both command starts. This does not
affect the token-identity or counter receipt, but timings in the raw logs are deliberately **not**
scored or used as performance evidence.

## Commands and raw evidence

```sh
env -u MEMRA_SPEC_K -u MEMRA_PROMPT_DIR -u MEMRA_SPEC_TEMP -u MEMRA_SEED \
  -u MEMRA_MTP_DRAFT CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=128 \
  MEMRA_PROMPT_FILE=research/e2e/prompts/p1-code-short.txt MEMRA_CHAT=1 \
  MEMRA_SPEC_STATS=1 target/release/run-spec \
  /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf

env CUDA_VISIBLE_DEVICES=0 MEMRA_NGEN=32 \
  MEMRA_PROMPT_FILE=research/e2e/prompts/p1-code-short.txt MEMRA_CHAT=1 \
  target/release/run-gen \
  /data/ai-ml/hf-models/qwen35-9b-nvfp4-gguf/Qwen3.5-9B-NVFP4-MTP-GGUF.gguf
```

Both commands ran under `flock /tmp/memra-gpu.lock` and exited zero. Raw stdout/stderr and the
before/after GPU snapshots are retained in `raw/run-spec-k1-8.log` and
`raw/run-gen-argmax.log`. `raw/environment.log` pins hashes and toolchain details; the complete
crate test logs are `raw/cargo-test-memra-engine.log` and `raw/cargo-test-memra-server.log`.
