# val256 validation progress — 2026-08-09

Lane: `lane/cx-val256`

Base: `4f80a0882f2c36572a386c6dc29831069c435ca9`

Rig: box1 rented cloud pair (`<rented-box-ip>`, 2x RTX PRO 6000)

Status: complete; final verdict is FAIL. See `RESULTS.md`.

## Block checklist

- [x] Block 1: affinity at `MEMRA_CTX=262144` — **FAIL**, receipt committed; no fix attempted
- [x] Block 2: admission at true 256k — **FAIL**, receipt captured; no fix attempted
- [x] Block 3: one requested-128k-session capacity row — receipt captured, N=1
- [x] `RESULTS.md` complete with raw evidence references

## Inbox checks

- Start / before Block 1: `~/.lanectl/inbox/cx-val256.md` was absent.
- Final Block 1 attempt: the same inbox path was still absent.
- Before Block 2: the same inbox path was still absent.
- Before Block 3: the same inbox path was still absent.

## Block 1 outcome

- The recorded workload gate passed: 8/8 rewritten sequential turns, 37,823–47,175 prompt
  tokens, PP-2 policy-selected K=0.
- One affinity-ON replay completed with 8 `plain_affinity_rewinds`; resumed sequential TTFT
  stayed 4.131–4.244 s while the cached boundary advanced 37,817–45,833 tokens.
- The first affinity-OFF replay completed three full-prime turns at 101.988, 105.790, and
  109.930 s, then turn 3 returned `cache alloc failed: DriverError(CUDA_ERROR_OUT_OF_MEMORY,
  "out of memory")`.
- Compute-app capture at failure: 87,530 MiB on GPU 0 and 97,036 MiB on GPU 1. Final metrics
  reported 3 parked continuation entries, 0 admission VRAM defers, and 0 step-OOM parks.
- Required N=3 A/B slope and three-server determinism checks are therefore incomplete. Per lane
  instruction, the failure is recorded and this lane will not attempt a runtime fix or retry.

## Block 2 outcome

- Forward order (`8k`, two `256k` parks, then c=4 requested-`128k`) completed all 7 requests;
  final metrics reported 0 step-OOM parks.
- Reclaim did precede the first defer: the server evicted both parked sessions and raised effective
  free memory from 9,346 MB to 31,727 MB before logging the first VRAM defer.
- The request-cost contract failed: the server logged only `ctx=262144`, `83520 B/token`, and
  `cost=21894MB`; it never logged distinct costs for requested caps 8,192 and 131,072. The forward
  arm accumulated 423 defer polling events while serializing the four-request burst.
- Inverse order (one requested-`256k` first, then c=4 requested-`8k`) also completed all 5 requests
  with 0 step-OOM parks, but the small burst was charged the same 21,894 MB and accumulated 423
  defer polling events. TTFB serialized from 0.350 to 3.948 s, so small requests were over-gated.
- This is a validation failure, not a harness failure. Per lane instruction, no runtime change or
  alternate setting will be attempted here.

## Block 3 outcome

- N=1, one exclusive lock, continuous one-second `nvidia-smi` sampling. Twenty-four concurrent
  requests each advertised `max_ctx=131072` to a `MEMRA_CTX=262144` PP-2 server.
- The first VRAM defer reported 1 active session: effective free 31,727 MB was below the admission
  requirement of 21,894 MB cost plus 21,894 MB reserve. This is the observed admission capacity.
- The cost line again reported effective `ctx=262144`, not 131,072. The row therefore describes
  current overcharged requested-128k behavior after the Block 2 failure; it is not evidence for a
  correctly scaled 128k-session capacity claim.
- All 24 requests completed, final metrics reported 0 step-OOM parks and 19,573 defer polling
  events, and no CUDA/OOM/panic line was captured. Request wall times spanned 1.277–29.021 s.
- Peak simultaneous `nvidia-smi` sample: GPU 0 66,705 MiB used / 30,548 MiB free at 34 C; GPU 1
  77,683 MiB used / 19,570 MiB free at 35 C; combined used memory 144,388 MiB.

## Discipline

- One bounded remote GPU-lock hold per block.
- Long runs detached and tee'd.
- Raw logs and JSONL retained next to the summary.
- Any product/runtime failure is captured and reported; no code fix in this lane.
- No origin push, tag, `rustup`, or `nsys`.
