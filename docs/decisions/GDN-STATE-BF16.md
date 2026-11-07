# bf16 GDN recurrent state (2026-11-07)

**Status:** rejected on the quality gate in both forms; the default stays fp32 on every device.
Implementations and receipts live on research branches, not on main:
`research/gdn-state-bf16-storage` (form 1,
[`research/gdn-state-bf16-storage-20261107/`](https://github.com/avifenesh/memra/tree/research/gdn-state-bf16-storage/research/gdn-state-bf16-storage-20261107)) and
`research/gdn-state-bf16-decode` (form 2,
[`research/gdn-state-bf16-decode-20261107/`](https://github.com/avifenesh/memra/tree/research/gdn-state-bf16-decode/research/gdn-state-bf16-decode-20261107)).

## Question

Qwen3.5-family hybrids (Qwen3.8-27B: 48 Gated DeltaNet layers) carry a recurrent state of
48 x 128 x 128 elements per layer: 144 MiB per session in fp32, read and written on every decode
step. SGLang (`--mamba-ssm-dtype bfloat16`) and vLLM store it in bf16. A rounding-only research
arm (`research/batch-spec-20261101/GDN-STATE-BF16.md`) rounded each store to bf16 at fp32 storage,
scored 160/164 and 97/100 on the long-context panel and cut the GB10 decode tick 6.6-8.9% at
16-32 sessions, but it ran two numeric programs (a T-row verify against T decode steps, a split
prime against the unsplit one). Can memra take the decode win with both exactness laws intact and
the panel in band?

## Form 1: bf16 storage everywhere

One storage class per process; every GDN state buffer half size; the state rounded after every
token and after every 32-token WY chunk of the prime (both laws need it once the prime stores bf16).

## Form 2: the decode class

The prime, every capture and every prefix entry stay fp32, main's program bit for bit. A
decode-class model rounds a layer's state to bf16 once, when its session starts decoding
(`Engine::gdn_ensure_decode`, `gdn_state_to_bf16`); decode and verify then store bf16 in the first
half of the fp32 buffers and round after every token (form 1's scan kernels). A prime that resumes
on a decoded session widens back exactly. decode == verify holds because both start from the same
rounded state and run the same per-token rounding scan; restore == unrestored holds because the
prime is fp32 (a split and restored prime equals the unsplit one, as on main) and the rounding is a
pure function of that state on both paths. Resident buffers are fp32, so form 1's capacity gain is
gone; decode traffic, verify rings, column checkpoints and stash slabs stay halved. Per model,
`MEMRA_GDN_STATE_BF16=1` on an eligible plan.

## Measurements (Qwen3.8-27B GGUF NVFP4+Q5_K, masked sxc32768 MTP draft, GB10)

| measurement | fp32 | form 1 | form 2 |
|---|---:|---:|---:|
| decode tick 16 x 2048 (A B B A A B medians) | 122.37 / 122.45 ms | 110.49 ms (-9.7%) | 110.60 ms (-9.7%) |
| decode tick 32 x 2048 | 246.62 / 246.21 ms | 223.14 ms (-9.5%) | 223.63 ms (-9.2%) |
| decode tick 8 x 2048 | 90.88 / 90.40 ms | 84.98 ms (-6.5%) | 84.61 ms (-6.4%) |
| decode tick 1 x 2048 | 73.50 / 72.98 ms | 72.78 ms (-1.0%) | 72.22 ms (-1.0%) |
| decode tick 16 x 12288 | 162.89 / 162.12 ms | 150.93 ms (-7.3%) | 150.84 ms (-7.0%) |
| cold TTFT 32K / 128K / 258K (A B B A medians) | 35.98 / 181.99 / 455.23 s; 32.27 / 167.68 / 426.44 s | 36.24 / 182.22 / 455.12 s | 32.22 / 167.24 / 426.98 s |
| MemAvailable floor at 256K | 45.95 GiB; 43.68 and 46.11 GiB | 47.87 GiB (+1.92) | 46.41 and 46.42 GiB (within the fp32 runs' 2.4 GiB spread) |
| long HumanEval, same-binary fp32 control 158 | 158 / 164 | 155 / 164 (twice) | 155 / 164 (twice) |
| long GSM8K | 97 / 100 | 97 / 100 (twice) | 97 / 100 (twice) |

fp32 columns: form 1's binary / form 2's binary (same fp32 program: identical decode-tick token
fingerprints and identical failed panel tasks). Both forms keep both exactness laws: run-spec K=1..8
identical to plain, the packed verify equal to chained rows, capture/restore cold-exact; form 2 also
passes a model-level restore == unrestored check through its decode-start rounding.

## Decision

Rejected under the gate ("the quality panel is the gate; it must stay in band"). Both forms score
155/164 on long HumanEval against a 156-159 band and a same-binary fp32 158, deterministically (the
same nine failures per form in every run, spread 0). The decode win is real (-9.7% per tick at 16
sessions on GB10) and does not override the gate. Nothing lands on main.

Form 2 answers the question form 1 left: removing every prime rounding (about 4,100 per
131K-token prime) recovered no task. The loss is in the per-token rounding of decode and verify,
which both forms share and which verify == decode requires. The arm that scored 160 rounded once
per verify launch, several times less often in speculative serving, and gave up that law to do it.

## Rejected alternatives

- **Store-only rounding** (the research arm): 160/164, but verify and decode are two programs and
  restored sessions diverge from unrestored ones. Not shippable.
- **bf16 on the RTX 5090**: form 1 measured -0.2% at 1 session to -1.7% at 16 per decode tick on the
  card, with no quality receipt for its served models. The 5090 stays fp32.
- **A half-size resident state with an fp32 prime scratch**: the carry between prime calls of a long
  prompt must stay fp32, so a priming session would hold the half state and a full fp32 carry at
  once; the 256K floor would fall below the fp32 class during the prime.

## What would change the answer

> Measured 2026-11-08 ([GDN-STATE-FP16.md](GDN-STATE-FP16.md)): fp16 holds the state's range
> (largest element 644x below its maximum), and its decode class scored 156/164 in both runs against
> the same-binary fp32 158, so it was rejected on the same gate. It lost one task fewer than bf16.


- **A finer 16-bit state format with per-token rounding**: fp16 keeps 10 mantissa bits against bf16's
  7, so each per-token rounding is about 8x smaller at the same bytes per step. It needs a range
  receipt for the GDN state (the largest state magnitude over long sessions against fp16's 65504 and
  its subnormal floor) before a panel.
- Rounding less often than once per token is not an option for an exact engine: plain decode runs
  one token per launch, so any coarser rounding in the verify makes verify and decode two programs.
