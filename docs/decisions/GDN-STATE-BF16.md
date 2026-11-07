# bf16 storage for the GDN recurrent state (2026-11-07)

**Status:** rejected on the quality gate; the default stays fp32 on every device. The
implementation and its receipts live on branch `research/gdn-state-bf16-storage`
([`research/gdn-state-bf16-storage-20261107/`](../../research/gdn-state-bf16-storage-20261107/)),
not on main.

## Question

Qwen3.5-family hybrids (Qwen3.8-27B: 48 Gated DeltaNet layers) carry a recurrent state of
48 x 128 x 128 elements per layer: 144 MiB per session in fp32, read and written on every decode
step, and the state class of every kernel, snapshot and prefix-cache entry. SGLang
(`--mamba-ssm-dtype bfloat16`) and vLLM store it in bf16. A rounding-only research arm
(`research/batch-spec-20261101/GDN-STATE-BF16.md`) rounded each store to bf16 at fp32 storage,
scored 160/164 and 97/100 on the long-context panel and cut the GB10 decode tick 6.6-8.9% at
16-32 sessions. Should memra store the state as bf16 for real, and where?

## What was built and measured

- **One storage class per process**, decided by the first engine (`MEMRA_GDN_STATE_BF16=0/1` or a
  device default). Every GDN state kernel takes `st16`; every GDN state buffer is sized in f32
  words of the class: resident state and twin, verify rings and checkpoints, prefix-cache entries
  (recurrent part 149.6 to 77.6 MiB), host staging, handoff layout. KDA and qwen4_exp stay fp32.
- **Rounding at every step boundary** (each token in the sequential and packed scans, each WY chunk
  in the chunked prime), because store-only rounding breaks two laws: a T-row verify stops equaling
  T decode steps (two numeric programs inside one request, since the spec gate switches between
  them), and a prime split at a chunk boundary stops equaling the unsplit prime (a restored session
  no longer continues like the unrestored one). With per-step rounding both laws hold: run-spec
  K=1..8 identical to plain, and capture/restore cold-exact, in both classes.

| measurement (Qwen3.8-27B GGUF, masked MTP draft) | fp32 | bf16 |
|---|---:|---:|
| GB10 decode tick, 16 x 2048 (A B B A A B medians) | 122.37 ms | 110.49 ms (-9.7%) |
| GB10 decode tick, 32 x 2048 | 246.62 ms | 223.14 ms (-9.5%) |
| GB10 decode tick, 1 x 2048 | 73.50 ms | 72.78 ms (-1.0%) |
| GB10 cold TTFT 32K / 128K / 258K (A B B A medians) | 35.98 / 181.99 / 455.23 s | 36.24 / 182.22 / 455.12 s |
| GB10 MemAvailable floor at 256K | 45.95 GiB | 47.87 GiB |
| RTX 5090 Laptop decode tick, 16 x 2048 | 121.74 ms | 119.70 ms (-1.7%) |
| GB10 long-context panel, long HumanEval (same binary) | 158 / 164 | 155 / 164 (twice) |
| GB10 long-context panel, long GSM8K (same binary) | see receipts | 97 / 100 (twice) |

The panel is deterministic on one binary and class: both bf16 runs failed the same nine HumanEval
tasks. Against the same-binary fp32 control the discordant tasks are 4 fp32-only and 1 bf16-only
(two-sided exact p = 0.375): not significant on its own, but outside the 156-159 band the gate uses.

## Decision

Rejected under the gate ("the quality panel is the gate; it must stay in band"): bf16 storage
with exact verify and restore scores 155/164 on long HumanEval against a band of 156-159 and a
same-binary fp32 158. The decode win is real (-9.7% per tick at 16 sessions on GB10) but does not
override the gate. Nothing lands on main.

## Rejected alternatives

- **Store-only rounding** (the research arm). It scored 160/164, but it keeps the state fp32
  across the rows of one verify launch and the chunks of one prime call, so verify and decode are
  two programs and restored sessions diverge from unrestored ones. Not shippable.
- **A bf16 class through the first half of each fp32 buffer** (the research arm's storage form):
  half the bytes per scan, no capacity change, same laws problem.
- **bf16 on the RTX 5090.** Measured -0.2% at 1 session to -1.7% at 16 per decode tick, with no
  bf16 quality receipt for the card's served models. Not worth a numeric change there either way.

## What would change the answer

- **An fp32 prime carry**: keep the state fp32 through every prime call and capture (prefix entries
  fp32, so prime-path numerics equal the fp32 class exactly), round it to bf16 once at the
  prime-to-decode transition, and keep per-token bf16 storage in decode and verify, where the speed
  comes from. The decode tick win stays; the prefix-entry capacity win goes; the prime adds no
  rounding at all. That is a new lever with its own gates (mixed-class prefix entries, pooled
  forwards that mix priming and decoding sessions).
- **A re-measured band**: several benign same-binary numeric variants of the fp32 class would show
  whether 155 is inside the panel's real spread on this build.
