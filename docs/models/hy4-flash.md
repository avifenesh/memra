# Hy4-flash

Tencent's Hy4-flash as a native ModelPlan pack, onboarded on its weight-only ModelOpt NVFP4
artifact: gated MLA on every layer (12 indexed layers with a pooled indexer, 30 sliding-window
layers), 4-stream hyper-connections, a latent MoE of 512 experts and the Engram n-gram memory.

| | |
|---|---|
| **Serving route** | memra-server's `hy_v4_flash` route (`hy4f_serve` over the GPU route `hy4f_route`) on one GB10 (DGX Spark, sm_121a): weights resident in their stored formats, a carried F32 cache per request, the Engram tables read from disk; `verify checkpoint` and `verify serve` pass at `d43aacce37` (the draft gate), served greedy answers equal the route's own. A correctness route: about 6 tok/s decode for one request undrafted, 9 with drafts |
| **Pack** | `hy_v4_flash` (aliases `hy-v4-flash`, `hy4-flash`), **NativeQualified** on that route (record `hy_v4_flash-gpu-1x-gb10`) and **NativeReference** on CPU (record `hy_v4_flash-reference`) <!-- support: hy_v4_flash-gpu-1x-gb10,hy_v4_flash-reference --> |
| **Artifact** | `Avifenesh/Hy4-flash-NVFP4@f9eac06cf022cfd3dc2e8ed423922466b2adcd87`, from `tencent/Hy4-flash@d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959`: NVFP4 routed experts, FP8 128x128 attention projections, FP8 Engram rows, BF16 and F32 elsewhere; 2,429 tensors, 161.08 GB |
| **Hardware** | one GB10 with 128 GB of unified memory: 87.5 GiB of weights resident, about 21 GB left for sessions; CPU for the reference gates |
| **Drafters** | the checkpoint's DFly2 drafter (5 blocks over the target's layers 25 to 41, MLP 9,216) drafts up to 5 tokens a round on the serve route, each round's depth priced by the draft gate from the drafter's own probabilities and the route's measured costs; the served tokens are the undrafted route's (`MEMRA_HY4F_DRAFT`, `0` turns it off; `MEMRA_HY4F_DRAFT_GATE=0` drafts the full 5 every round) |
| **Input** | text; the artifact's chat template, efforts and tool-call format; image and video parts are refused |

## Serve it

```bash
MEMRA_MODELS=hy4=/path/Hy4-flash-NVFP4 memra-server
```

- The served context is the largest session the memory book holds with the route idle: 509,886
  positions on a GB10 with the drafter resident, 532,944 without it (the config declares
  1,048,576). A request that cannot fit is refused by
  name; one that fits alone waits for running requests to finish.
- Long prompts are prefilled in chunks of `MEMRA_HY4F_PREFILL_CHUNK` tokens (default 2,048), one
  chunk a step beside the other requests' decode; the chunk size never changes a value.
- `reasoning_effort` takes `no_think` (or `none`), `high` (the template's default) and `max` (or
  `xhigh`); `low` and `medium` render `high`. Tool calls parse from the Hy4 call format.
- Greedy, temperature, top-p, top-k, seeds, penalties, stop strings and `ignore_eos` are served;
  `min_p`, `response_format` and image parts are not.
- DFly2 drafts are on by default: after every call the drafter proposes up to the artifact's 5
  tokens, the next call verifies them in one call, and the route keeps the drafts that equal its
  own choices, so every answer is the undrafted route's token for token, greedy or sampled. The
  draft gate verifies the depth with the most expected tokens a millisecond, from the drafter's
  probabilities scaled by how the request's drafts held. `MEMRA_HY4F_DRAFT=N` caps the drafts a
  round, `0` serves without the drafter.

## Gates

- Config, TokenizerTemplate and TensorCensus pass at `memra model inspect` on the pinned revision.
- TinyParity: `memra model verify tiny --against hy_v4_flash` on the artifact's config at hidden
  64, past the 512-token window and the indexer budget, with splits below each.
- CheckpointParity: `memra model verify checkpoint <artifact> --against hy_v4_flash --oracle
  <tapes> --native-runner hy4f-run` (f32, the source program) and `--native-runner hy4f-gpu`
  (class `hy-v4-flash-bf16x2-activations`, the oracle's picks applied) against the pinned
  transformers program in f64.
- Serve: `memra model verify serve <artifact> --against hy_v4_flash --native-runner memra-server`,
  and the served-answer gates: served greedy equal to the route's, teacher-forced positions equal
  to the verified tapes, chunked prefills bit-equal to whole ones, concurrent requests equal to
  solo, streaming, efforts, a tool call round trip, sampling, stops, cancellation, admission and a
  30,934-token needle.
- Drafts: the DFly2 drafter against the pinned vendor program in f64 (`memra-reference hy4f_dfly`
  on the tiny fixture and the artifact; `hy4f-spec gate-route` for the route's class, band 0.0398);
  `hy4f-spec gen` (drafted greedy equal to plain greedy, ids and every logits row); the served
  gates above with drafts on; every answer of that gate equal to a drafts-off server's
  (`draft_identity.py`); and the chat turns with a refused draft round and a refused ring backup
  equal to the drafts-off answers (`MEMRA_HY4F_DRAFT_FAULT`, `turns_identity.py`).

## Limits

- Untuned: about 6 tok/s decode without drafts and 205 tok/s prefill on a 2,104-token prompt for
  one request; with drafts and the draft gate, six sampled chat turns read 1.3 to 2.2 times the
  undrafted speed (9.1 against 5.8 tok/s over the six; 0.42 to 0.73 of the verified drafts
  hold), and the 2,104-token prompt decodes at 9.05 tok/s (step 11). Requests run one call at a time, so throughput does
  not grow with concurrency.
- No prefix reuse, constrained decoding or vision; drafts are greedy for sampled requests too
  (drawing them with the request's sampler kept fewer, step 11).
- One GB10 only; the record covers no other device.

## Deeper sources

- The lane: [`research/modelplan-onboarding-hy4flash-20261119/`](../../research/modelplan-onboarding-hy4flash-20261119/)
  (pins, semantics, each step's oracle and gate; step 9 is the serve route).
- Pack: `crates/memra-gguf/src/model_packs/hy_v4_flash/`; reference:
  `crates/memra-reference/src/hy_v4_flash.rs` and `hy_v4_flash_trunk.rs`; checkpoint runner:
  `crates/memra-engine/src/hy4f_gpu.rs`, `src/bin/hy4f_run.rs`, `cu/hy4f_gpu.cu`; GPU route:
  `src/hy4f_route.rs`, `src/bin/hy4f_gpu.rs`, `cu/hy4f_route.cu`
  ([activation format](../decisions/HY4F-GPU-ACTIVATION-FORMAT.md)); serve route:
  `crates/memra-server/src/hy4f_serve.rs`; the DFly2 drafter: `crates/memra-reference/src/hy_v4_flash_dfly.rs`,
  `crates/memra-engine/src/hy4f_route/dfly.rs`, `src/bin/hy4f_spec.rs` (step 10 of the lane).
