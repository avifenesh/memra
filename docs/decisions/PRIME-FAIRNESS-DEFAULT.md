# Prime fairness: the owned prime walker yields by default (2026-09-22)

**Status:** decided ON (memra#521). `MEMRA_PRIME_YIELD` defaults to ON; `MEMRA_PRIME_YIELD=0` is the
rollback seam with `decide-by: 2026-10-06` for deleting the seam. Landed with the serving-shape gate
`tools/prime-fairness-gate.py` in `tools/local-ci.sh`. Receipts: `research/prime-fairness-default-20260922/`.

## Question

Since 2026-09-08 the speculative prime routes (GDN MTP prime, DFlash, GLM plain and spec) own a
`PrimeWalker` that freezes the prime's chunk tape once and can advance one chunk per worker tick
(`research/prefill-fairness-20260908/SEAM.md`). The door stayed OFF: OFF drains the same tape in one
call, which is the shape of the 2026-09-05 incident (a 135k-token cold prefill in one worker tick,
`tick_max_ms 92542`, every queued request from three other tenants timing out for 11.5 h while every
monitor stayed green). The question was whether yielding becomes the naked default on every walker
route, on the owner's per-hardware rule: numbers from the local RTX 5090 and from an RTX PRO 6000
before a default moves.

## Measured

Three independent RTX 5090 receipts on the MTP route preceded this decision:
`research/prefill-fairness-20260908/MTP.md` (Ornith 35B, 127k cold prime beside 20 small arrivals:
small p95 15.125 s OFF, 0.977 s ON, long TTFT +1.756 s), `research/requal-5090-20260909/`
(10.141 s to 4.308 s) and the 2026-09-11 Ornith receipt (darklanes#641, `research/orn-5090-20260911/`:
eight cold 4,228-token peers beside one 104k prime, small p95 19.72/19.85 s OFF, 7.01/6.97 s ON at
chunk 4096, 4.78/4.79 s at chunk 1024, long prime +54%). DFlash: exactness and admitted-peer gates
pass, mixed p95 60.452 s to 48.042 s, still admission-limited (`DFLASH.md`).

The decision cell (this lane): `tools/prime-fairness-gate.py` on Qwen3.5-9B NVFP4 MTP GGUF, spec
route pinned (`MEMRA_SPEC_GATE_LOW=64 HIGH=65`), one 131k cold prime beside two cold 2k peers and one
4k cache hit, greedy, both arms interleaved by boot, three reps per rig. The first gate version sent
synthetic token ids; review round 1 showed those prompts end at EOS almost at once, so the byte
clause compared two characters. The gate now sends calibrated natural text (135,470 / 2,071 /
2,140 / 4,111 tokens on the 9B) whose tail asks each prompt for its own case number and plant word,
refuses a run whose requests generate fewer than 16 tokens or whose three cold prompts do not produce
three distinct outputs (round 2: a shared task line had made every request echo the same tokens);
the 5090 cell was re-run on the final version, the PRO 6000 cell below is the id version (its timing
clauses are unchanged by the prompt content; its byte clause compared the short outputs).

| rig (3 reps, arms interleaved by boot) | peers p95 first event OFF / ON (s) | peers max OFF / ON (s) | `tick_max_ms` OFF / ON | long prime first event OFF / ON (s) | bytes identical | yields ON |
|---|---|---|---|---|---|---|
| local RTX 5090, prompt-derived gate (final; each request answers with its own case number and plant word) | 52.88 / 1.52 | 52.88 / 1.52 | 54375 / 2430 | 54.08 / 56.39 | yes (four distinct outputs, each one sha across 6 boots) | 426 |
| local RTX 5090, shared-task text version (superseded: every request echoed the same 32 tokens, round 2) | 55.02 / 1.43 | 55.02 / 1.43 | 56479 / 2458 | 55.82 / 55.94 | yes (every request one sha across 6 boots, full outputs) | 426 |
| local RTX 5090 (9950X host), id-prompt gate version | 49.64 / 1.87 | 49.64 / 1.87 | 51199 / 2309 | 50.77 / 52.45 | yes (every request one sha across 6 boots) | 189 |
| rented RTX PRO 6000 Blackwell WS (Core Ultra 9 285K host, driver 595.71.05) | 14.51 / 0.53 | 14.51 / 0.53 | 16338 / 870 | 16.33 / 17.26 | yes (every request one sha across 6 boots) | 189 |

Both rigs: PASS on every clause. The long prime's own first event pays about 3% on the 5090 and
about 6% on the PRO 6000 (its 131k prime runs in 32 chunks of 4,096 and each chunk now costs one
extra scheduler pass and the peers' quanta). The peers' wait falls from the length of the whole
prime to about one chunk. `tick_max_ms` is the whole prime OFF and one chunk ON on both cards.

## Decided

- ON is the default on every walker route. The mechanism is structural: a yield returns the worker
  to its peers between two chunks of a tape that is frozen before the first chunk, so the bytes of
  every request are the ones the OFF arm produces (the gate's V1 asserts it on every run).
- `MEMRA_PRIME_CHUNK` stays at its 4096 default. It is the quantum: 1024 halves the peers' wait
  again and costs the long prime more. A latency-tuned launcher sets 1024; the engine default is a
  memory knob and stays.
- The OFF arm survives only as the `=0` rollback seam for two weeks. If nobody uses it by
  2026-10-06 the seam is deleted and the arm is the naked default.

## Rejected

- Keeping the shared default OFF and enabling per launcher (the 2026-09-09 proposal). Rejected
  because the incident class is the default configuration, and the routes the door does not reach
  are unaffected by it either way: the serial plain-trunk prime is bounded by `MEMRA_PREFILL_TICK`
  per tick; E4B and dsv4 prime monolithically regardless and belong to memra#535 P3/P4.
- Flipping `MEMRA_PRIME_CHUNK` to 1024 with it. Rejected here: the chunk is a memory knob on the
  serial trunk and a per-launcher latency choice; the receipts show the direction at both sizes.

## What the gate found beside the decision

A peer whose route differs between arms (spec K=3 on one arm, plain K=0 by concurrency demotion on
the other) produced different greedy bytes on a synthetic near-tie prompt, while the same prompt
solo on the spec route and solo on the plain route agree. The gate therefore pins the route, and the
divergence is memra#641 with its raw receipts; it is recorded in the lane README, not decided
here.
