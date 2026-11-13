# DeepSeek-V4.1-Flash

DeepSeek's V4.1 Flash (the 1017 checkpoint) as a native ModelPlan pack: the reference executor on
CPU, and memra-server's own deepseek-v4.1 route on the GPU, which serves the checkpoint's own
program on one RTX PRO 6000 Blackwell.

| | |
|---|---|
| **Serving route** | memra-server's deepseek-v4.1 route (`dsv41_serve` over `dsv41_gpu::decode`) on one RTX PRO 6000 Blackwell Max-Q (sm_120a, 96 GB), the routed experts in pinned host memory; `verify checkpoint` and `verify serve` pass, served greedy answers equal the native runner's. A correctness route at 7.5 tok/s decode for one request, not a fast one |
| **Pack** | `deepseek_v41`, **NativeQualified** on that route (record `deepseek_v41-gpu-1x-pro6000`) and **NativeReference** on CPU (record `deepseek_v41-reference`) <!-- support: deepseek_v41-gpu-1x-pro6000,deepseek_v41-reference --> |
| **Artifact** | `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`, 48 safetensors shards, 96,085 header tensors |
| **Hardware** | one 96 GB RTX PRO 6000 Blackwell and about 290 GB of free host memory for the pinned expert store; CPU for the reference executor |
| **Input** | text in the plan; the serve surface refuses image parts |

## Serve it

```bash
MEMRA_MODELS=v41=/path/DeepSeek-V4.1-Flash-1017 \
MEMRA_DSV41_TOKEN_MAP=/path/engram_constants.token_map.i32 \
memra-server
```

- The Engram hash needs the vendor's compressed token map (`MEMRA_DSV41_TOKEN_MAP`, raw i32 per
  vocabulary id); memra does not derive it from the tokenizer yet.
- Chat requests render with `encoding_dsv41` and parse V4.1's DSML tool calls and reasoning.
  `reasoning_effort` takes `none` (chat mode), the levels, or the numeric budget `"1"`..`"100"`.
- Greedy, temperature, top-p, top-k, seeds, penalties and stop strings are served; `min_p`,
  `response_format` and image parts are not. `/v1/completions` takes `ignore_eos`
  ([API surfaces](../API-SURFACES.md)).
- DSpark drafting (`MEMRA_DSV41_DRAFT=1`) is off by default. The draft head and the several-token
  verification call are gated at the engine (the head on the vendor's floor, a verification row
  byte-identical to its one-token call, drafted greedy equal to plain greedy); the served route's
  gates with drafting on have not run. Receipts:
  `research/modelplan-onboarding-dsv41-20261109/dspark/`.
- `MEMRA_DSV41_DEVICES=a,b,...` names the cards of an N-rank split. Residency follows the cards:
  four ranks on four 96 GB cards hold the whole trunk, and the route then uses no host store.
  That arm has no serve-gate evidence yet; the record covers one card.

## What the route does

Every layer's own weights and the head stay on the card (9.9 GB). The 15,360 routed experts of the
40 trunk layers (289 GB) are page-locked in host memory, and the card keeps an LRU cache of expert
slots in its free memory (4,736 slots, 89 GB, with the route idle); the experts a router picks are
copied in on a copy stream. A slot holds the checkpoint's bytes as stored, so the route writes the
streamed one-card runner's numbers byte for byte. Each request is one prefill call, then one
decode call per token, the vendor's two programs; up to `MEMRA_MAX_SESSIONS` requests share each
step, each running alone through each block.

## Run the reference executor

From a source checkout (the release tarball does not ship the `memra` CLI):

```bash
cargo run -p memra-cli --bin memra -- model inspect \
  deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170 \
  --against deepseek_v41 --out inspect
cargo run -p memra-cli --bin memra -- model verify tiny --against deepseek_v41 --out tiny
```

`verify tiny` runs the main fixture (prefill past the window, then decode with DSpark drafts) and
two variants, a split below the window and a window of 128. Each is pinned to an oracle the
vendor `model.py` produced on the same checkpoint image, and memra matches all three to the bit.

## What the pack covers

- Storage: dense weights FP8 e4m3 in 32x32 blocks, routed
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

## Prefill and decode below the window

Below the 128-token window, the vendor's prefill and decode programs compute different numbers
for the same row. A prefill row lists its window entries first and its empty slots last, a decode
row lists the ring oldest first with the empty slots first, and the sparse-attention kernel rounds
probabilities to bf16 per block of 64 entries. At or past the window the two agree to the bit
(pinned on the tiny configs at windows 64 and 128). The route runs the vendor's two programs as
the vendor does, one prefill call per prompt and then one decode call per token, so it never
primes, chunks or splits a request below the window.

## Deeper sources

- [Serve receipts](../../research/modelplan-onboarding-dsv41-20261109/serve/README.md): the gates,
  the residency byte-identity runs, the decode rate.
- [Onboarding](../../research/modelplan-onboarding-dsv41-20261109/README.md): pack, census, tiny
  gate, encoding, GPU layers, CheckpointParity, the N-rank split and the decode loop.
- The chat dialect and its fixtures: `research/modelplan-onboarding-dsv41-20261109/encoding/`.
- Support records: `deepseek_v41-gpu-1x-pro6000` and `deepseek_v41-reference` in
  [support-records.toml](../support-records.toml).
- Pack and plan: `crates/memra-gguf/src/model_packs/deepseek_v41/`; reference:
  `crates/memra-reference/src/deepseek_v41.rs`; GPU route: `crates/memra-engine/src/dsv41_gpu.rs`
  and `crates/memra-server/src/dsv41_serve.rs`.

Nothing here transfers to DeepSeek-V4-Flash 0731 (`deepseek_v4`, [its card](deepseek-v4-flash.md)),
a different program.
