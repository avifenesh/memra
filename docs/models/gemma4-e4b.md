# Gemma-4 E4B

Google's Gemma 4 E4B from the official QAT Q4_0 GGUF: the dense Gemma 4 program plus per-layer
input embeddings and KV sharing. Its engine cells pass on one RTX 5090 Laptop; `memra-server`
serving is not gated.

| | |
|---|---|
| **Serving route** | none gated. `memra-server` loads E4B in its eager-only class, and the served drafter attach skips it. Engine cells (load, run-gen, calibrated argmax, `gemma-gate` speculative K=1..8, verify, graph) passed on 2026-11-04, one RTX 5090 Laptop. Standing gate: none |
| **Pack** | `gemma4_e4b`, NativeReference on the tiny gate (record `gemma4_e4b-reference`; Config, TokenizerTemplate and TensorCensus also pass on this GGUF) <!-- support: gemma4_e4b-reference --> |
| **Artifact** | `gemma-4-E4B_q4_0-it.gguf` (5,154,941,280 bytes) from `google/gemma-4-E4B-it-qat-q4_0-gguf`; drafter `gemma-4-E4B-it-assistant.Q8_0.gguf` (98,653,280 bytes) |
| **Drafters** | the Gemma assistant drafter, through `gemma-gate` only |
| **Hardware measured** | RTX 5090 Laptop |
| **Input** | text |

## What the pack covers

The last 18 layers read the caches of layers 22 and 23 (KV sharing), and each layer takes a
256-wide per-layer input. The `gemma4_e4b` pack types both in the plan, and its global layers carry
their own V projection. The pack matches this geometry only (42 layers, 18 of them KV-shared,
per-layer inputs 256 wide); another checkpoint with per-layer inputs gets no pack until it has its
own record. CheckpointParity, RewriteParity and Serve are pending.

E4B runs its own programs: whole-prompt prime, eager and dc decode, a graph-exec loop, and
`gemma-gate`'s batched verify and assistant-drafter speculative decode. It has no batched decode
and no pipeline split; a load with `MEMRA_PP_STAGES` set refuses by plan operation.

## Run it

No cookbook block. The engine gates run as `model inspect <gguf> --against gemma4_e4b --out <dir>`
(the `memra` CLI, `cargo run -p memra-cli --bin memra --`), `run-gen`,
`tools/argmax-margin-gate.sh <gguf>` and `gemma-gate` with `MEMRA_DRAFT=<assistant>`;
the exact commands per cell are in the
[record](../../research/modelplan-onboarding-e4b-20261104/RESULTS.md).

## Measured

RTX 5090 Laptop (driver 615.71.09), 2026-11-04, `sm_120a`, one run per cell. The engine cells were
re-run on `fbdd631927`; the pipeline-refusal and serve cells come from the lane's first run at
`09348940b8`.

- Load: 666 tensors bound by the pack contract.
- `run-gen` argmax: prefill/decode and batched prime MATCH; calibrated argmax PASS with 0 flips.
- `gemma-gate` speculative K=1..8: stream agreement 64/64 at every K; acceptance 0.684 at K=1,
  0.407 at K=8.
- Batched verify PASS; graph door 256/256 tokens identical to eager.
- `memra-server`: the plain surface passes serve-smoke 7/7, and a greedy chat repeated three times
  concurrently matches its solo run.

Plain decode: 199.9 tok/s on the RTX 5090 Laptop (the 2026-07-15 Gemma re-audit;
[Gemma-4 history table](../history/PERFORMANCE.md#gemma-4-qat-q4_0)). Full table:
[`research/modelplan-onboarding-e4b-20261104/`](../../research/modelplan-onboarding-e4b-20261104/RESULTS.md).

## Limits

- The serve-smoke gemma4 chat arm fails: on a thinking-off chat E4B writes a thought after the
  template's closed thought channel and ends it with a stray `<channel|>`, so the thought and the
  tag reached `content`. The gemma4 parsers now route text before an unopened `<channel|>` to
  `reasoning` ([decision](../decisions/GEMMA-UNOPENED-THOUGHT.md)); the arm has not been re-run on
  it. At `reasoning_effort: low` E4B answered with no thought, so the thinking-on check fails too.
- The cache-metering cell does not apply: the eager-only class takes no prefix snapshot.
- `run-spec` needs an embedded MTP head, so the assistant drafter is gated by `gemma-gate` stream
  agreement instead, the same check the 31B runs in `tools/local-ci.sh`.

## Deeper sources

- Record and raw logs: [`research/modelplan-onboarding-e4b-20261104/`](../../research/modelplan-onboarding-e4b-20261104/RESULTS.md).
- Pack: `crates/memra-gguf/src/model_packs/gemma4_e4b/`.
- The full gate table as first written: [history](../history/MODELS.md#gemma-4-e4b-card-qualification-record-2026-11-04).
