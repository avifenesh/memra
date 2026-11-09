# DeepSeek V4.1 Flash

| | Recommended use |
|---|---|
| **Status** | NativeQualified on one RTX PRO 6000 Blackwell (Max-Q, sm_120a) with the routed experts in pinned host memory <!-- support: deepseek_v41-gpu-1x-pro6000 --> |
| **Path** | The pinned safetensors checkpoint `deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e10`, its own program (bf16 activations, FP8 32x32 dense, MXFP4 experts, per-32 activation quantization), served by memra-server's deepseek-v4.1 route |
| **Hardware** | One 96 GB RTX PRO 6000 Blackwell and about 290 GB of free host memory for the pinned expert store |
| **Use this when** | You want the checkpoint's exact program served on one card: a correctness route at single-digit tokens per second, not a fast one |

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
  `response_format`, image parts and DSpark drafts are not.
- `MEMRA_DSV41_DEVICES=a,b,...` names the cards of an N-rank split. Residency follows the cards:
  four ranks on four 96 GB cards hold the whole trunk, and the route then uses no host store.
  That arm has no gate evidence yet; the record covers one card.

## What the route does

Every layer's own weights and the head stay on the card (9.9 GB). The 15,360 routed experts of the
40 trunk layers (289 GB) are page-locked in host memory, and the card keeps an LRU cache of expert
slots in its free memory (4,736 slots, 89 GB, with the route idle); the experts a router
picks are copied in on a copy stream. A slot holds the checkpoint's bytes as stored, so the route
writes the streamed one-card runner's numbers byte for byte. Each request is one prefill call,
then one decode call per token, the vendor's two programs; up to `MEMRA_MAX_SESSIONS` requests
share each step, each running alone through each block.

## Deeper sources

- [Serve receipts](../../research/modelplan-onboarding-dsv41-20261109/serve/README.md): the gates,
  the residency byte-identity runs, the decode rate.
- [Onboarding](../../research/modelplan-onboarding-dsv41-20261109/README.md): pack, census, tiny
  gate, encoding, GPU layers, CheckpointParity, the N-rank split and the decode loop.
- [Support record](../support-records.toml) `deepseek_v41-gpu-1x-pro6000`.
