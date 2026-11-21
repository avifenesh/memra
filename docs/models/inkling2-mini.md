# Inkling 2 Mini

Thinking Machines' Inkling 2 Mini as a native ModelPlan pack: the reference executor on CPU, and
memra-server's own inkling2 route on the GPU, which runs the checkpoint's own program, with the
weights as stored, on one 24 GB RTX 5090 Laptop GPU.

| | |
|---|---|
| **Serving route** | memra-server's inkling2 route (`inkling2_serve` over `inkling2_gpu`) on one RTX 5090 Laptop GPU (sm_120a, 24 GB): every request in the checkpoint's declared numeric class, several requests at once in batched decode calls whose rows equal their solo runs bit for bit. 41 tok/s for one request (TPOT 24 ms) and 24.6 tok/s over four at once, the one-row decode tuned and the batched decode not yet (2026-11-21) |
| **Pack** | `inkling2`, **NativeReference** (record `inkling2-reference`) <!-- support: inkling2-reference -->. CheckpointParity and Serve pass on the declared class's band (`docs/decisions/INKLING2-DECLARED-CHECKPOINT-PARITY.md`); RewriteParity is pending, so the state stays NativeReference |
| **Artifact** | `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`: one indexed safetensors shard and the drafter's `mtp.safetensors`, 1,170 census tensors bound before upload |
| **Hardware** | one 24 GB card of the 120a family: the weights take 17.6 GiB as stored, and the rest holds the requests' state |
| **Builds** | `sm_120a` (the routed experts' act-quant GEMMs are the 120a family's block-scaled MMA); a build without them refuses to load the declared class, by name |
| **Input** | text; image parts are refused (no vision program yet) |

## Serve it

```bash
MEMRA_MODELS=ink2=/path/Inkling-2-Mini MEMRA_CTX=8192 memra-server
```

- `MEMRA_CTX` is required: the checkpoint declares 1,048,576 positions, beyond what the route
  holds. With the f32 KV of this route, one RTX 5090 Laptop holds about 75,000 positions of
  full-attention cache in total at prefill chunks of 512; each request also takes 0.63 GiB of
  sliding-window rings and conv state, and the route admits a request only when the driver has
  its state free.
- Chat requests render the checkpoint's own template (the inkling2 dialect) and parse its replies
  by token id: thinking goes to `reasoning`, text to content, each tool-call message to one
  OpenAI `tool_calls` entry. A control token counts as a marker only when its id arrives.
- `reasoning_effort` takes the template's names (`none`, `minimal`, `low`, `medium`, `high`,
  `xhigh`, `max`) or a JSON number in [0.0, 0.99]. A number sent as a string is a name the
  template does not know, and is refused.
- Greedy, temperature, top-p, top-k, seeds, penalties and stop strings are served. Generation
  stops on the end-of-sampling id 200006. `min_p`, `response_format`, forced `tool_choice`,
  `parallel_tool_calls: false` and image parts are refused by name.
- The drafter (DSpark, `mtp.safetensors`) is bound by the tensor contract and not served yet.

## What the route does

Each request owns a sequence: the full-attention caches of the eight full-attention slots, the
rings of the 40 sliding-window slots and the conv histories of all 48 slots (24 weight sets, the
middle 12 run three times). Each step runs one prefill chunk of the oldest prefilling request,
then one decode token of every other running request in one batched call: each row is its own
sequence's one-token call at its own position, through shared launches. The batching gate
(`ink2-gpu --batch-check`) holds every decode row's logits to its solo run bit for bit, with
requests at different positions, joining and leaving mid-stream, and prefill chunks interleaved
with their peers' decode, in both numeric classes.

The numeric classes: the declared class (the default and the support target) quantizes the routed
experts' inputs as the checkpoint declares (NVFP4 four-over-six W4A4 on layers 0 to 19, MXFP8 W8A8
on 20 to 23) on an f32 stream outside the experts. The float-activation class (exact dequant on
f32 inputs) is a bring-up and oracle class, not served.

## Measured

On one RTX 5090 Laptop GPU, 2026-11-21, memra-server with `MEMRA_CTX=8192` and four sessions,
8 greedy streamed chat requests per level of 128 output tokens each, one boot, clocks full by the
telemetry rule (P0, 14001 MHz) with thermal slowdown on part of the window:

| Concurrency | TTFT p50 | TPOT p50 | Output tok/s |
|---|---|---|---|
| 1 | 41 ms | 24 ms | 41.0 |
| 2 | 315 ms | 139 ms | 14.2 |
| 4 | 533 ms | 162 ms | 24.6 |

Every request's answer at concurrency 2 and 4 equals its answer alone. One request alone is the
fastest case today: the one-row decode is tuned, the batched decode is not yet. p95, p99, ITL, E2E and
the conditions: `research/modelplan-onboarding-inkling2mini-20261121/serve/README.md`.

## Deeper sources

- Onboarding lane: `research/modelplan-onboarding-inkling2mini-20261121/README.md`
- GPU route and its checkpoint gates: `research/modelplan-onboarding-inkling2mini-20261121/gpu-route/README.md`
- Serve route, batching gate and serve receipts: `research/modelplan-onboarding-inkling2mini-20261121/serve/README.md`
- Act-quant kernels: `research/modelplan-onboarding-inkling2mini-20261121/actquant/README.md`
