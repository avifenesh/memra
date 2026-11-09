# DeepSeek-V4.1-Flash

Support state: **NativeReference** for the `deepseek_v41` pack <!-- support: deepseek_v41-reference -->.
The plan runs in the reference executor on CPU and the tiny gate passes. Checkpoint parity,
rewrite and serve gates are pending, so nothing serves this model yet.
NativeQualified needs those gates first. <!-- support: none; not NativeQualified -->

## Recommended path

There is no serving path. What runs today is onboarding and the reference executor:

```bash
memra model inspect deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out inspect
memra model verify tiny --against deepseek_v41 --out tiny
```

`verify tiny` runs the main fixture (prefill past the window, then decode with DSpark drafts) and
two variants, a split below the window and a window of 128. Each is pinned to an oracle the
vendor `model.py` produced on the same checkpoint image, and memra matches all three to the bit.

## What the pack covers

- Source: `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`, 48
  safetensors shards, 96,085 header tensors. Dense weights FP8 e4m3 in 32x32 blocks, routed
  experts MXFP4, Engram tables FP8 rows. The census binds every tensor, the vision tower
  included, which the text plan does not execute.
- Architecture, read from `config.json` and the vendor `inference/model.py`: 40 trunk layers and
  3 DSpark draft blocks. CSA2 attention: a 128-token window on every layer, compressed KV at ratio
  2 (layers 2 to 19) and ratio 1 (20 to 39) shared within a layer group, a hierarchical indexer
  with a candidate pool produced at layer 20. Single-pass mHC over 4 streams, Engram at layers 1
  and 14, a 384-expert top-6 sqrtsoftplus MoE with one shared expert. Context 1,048,576 by YaRN.
- Chat dialect: `encoding_dsv41`, its own dialect (spaced DSML tags, a numeric reasoning budget,
  mid-conversation system turns, tool namespaces). It renders and parses byte for byte as the
  vendor's `encoding/encoding.py`; the serve surface refuses image parts, since memra has no V4.1
  vision program.
- Numerics: the reference reads every quantize-dequantize step from the plan and refuses by name
  any step or storage it does not implement.

## Constraint for a serving route

Below the 128-token window, the vendor's prefill and decode programs compute different numbers
for the same row. A prefill row lists its window entries first and its empty slots last, a decode
row lists the ring oldest first with the empty slots first, and the sparse-attention kernel rounds
probabilities to bf16 per block of 64 entries. At or past the window the two agree to the bit
(pinned on the tiny configs at windows 64 and 128). A serving route that primes, chunks or
splits a request below the window runs a different numeric program for those rows, and has to
fix the split or gate it.

## Deeper sources

- Onboarding receipts, the tiny gate and its oracle: `research/modelplan-onboarding-dsv41-20261109/`.
- The chat dialect and its fixtures: `research/modelplan-onboarding-dsv41-20261109/encoding/`.
- Support record: `deepseek_v41-reference` in `docs/support-records.toml`.
- Pack and plan: `crates/memra-gguf/src/model_packs/deepseek_v41/`; reference:
  `crates/memra-reference/src/deepseek_v41.rs`.

Nothing here transfers to DeepSeek-V4-Flash 0731 (`deepseek_v4`, [its card](deepseek-v4-flash.md)),
a different program.
