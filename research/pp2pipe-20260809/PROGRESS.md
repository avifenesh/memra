# pp2pipe — PP-2 prime serve-trial increment

Branch: `lane/pp2pipe`
Base: `8e8c93af`
Rig: box1 `<rented-box-ip>`, 2x RTX PRO 6000, PP-2 devices 0/1 under
`/tmp/memra-gpu.lock`.

## Mission and stop line

Move exact 4096-token streaming TTFT below 10 seconds tonight with the smallest
correct PP-2 prime change. Anything not on that direct path is deferred. If a
subset clears the target, run the required gates and stop rather than broadening
the mechanism.

Inherited standing evidence is recorded in
`research/pipeprime-20260808/PROGRESS.md`; that lane's requested `RESULTS.md` is
absent in this checkout. Its final box2 receipts report:

- pp512 / pp2048 / pp4096 naked pipeline: 330.0 / 401.8 / 417.6 tok/s, N=5;
- exact 4096-token streaming TTFT: 11.009 s p50, N=5;
- unsplit / serial split / pipelined split bit-identical, with live-overlap teeth;
- model-backed kernel-check green, PP-2 run-gen MATCH, run-spec K=1..8 PASS,
  and chunk/tick invariance plus canaries green.

The original anatomy bill remains the mechanism context: the unsplit prime
walker leaves device 1 idle, pays a 22% stage-1 peer-read tax, and spends 28% in
tokenwise MoE dispatch. The inherited pipeprime implementation already claims
stage-local walkers, per-device prime slabs/caches, and chunk overlap, so this
increment will first verify the current code and box1 runtime rather than repeat
that work.

## Pre-registered order

1. Audit the inherited walker, engine/cache ownership, and serve tick geometry.
2. Measure one bounded box1 baseline if the inherited receipt cannot identify the
   remaining >1.009 s directly.
3. Implement only the smallest identified critical-path change.
4. Run split-vs-unsplit prime identity, model kernel-check, PP-2 run-gen,
   run-spec K=1..8, chunkinv35/tickinv35, and all required canaries.
5. Run interleaved A/B in one lock hold, N=5, retain raw logs, and report medians
   with the thermal regime.

Success is TTFT p50 below 10 seconds with all gates green. A performance result
without the exactness battery is not a result.

## Coordination state

`~/.lanectl/inbox/pp2pipe.md` was absent at lane start (2026-08-09); the inbox
directory contained no alternate pp2pipe entry. It will be checked at every
bounded work block as requested.

## Increment 1 — integrated-tip audit changes the implementation verdict

Base `8e8c93af` already contains all four parts of the Lever-B cut:

- stage-local prime ranges run through `PpNRt::engine(stage, primary)`;
- each engine owns its own MoE cache and per-device prime slab;
- `prime-split-gate.sh` compares unsplit, serial split, and pipelined split in
  one process with bit and liveness checks;
- two stage-owned host walkers overlap adjacent chunks while the boundary
  slots retain their TX-waits-RX reuse edge.

It also contains the later dynamic-microchunk, grouped-prefill, and solo-fresh
outer-prefill merges. An ancestor same-rig serving receipt reported 5.992 s p50
for a rendered 4107-token turn (N=5), but that run used grouped prefill as its
then-default policy. The integrated tip now documents grouped prefill as opt-in
after the 5090 transfer gate rejected its default. The old 5.992 s receipt is
therefore not current naked serving truth; this lane must reproduce the target
with today's defaults before applying the stop rule.

The two cited upstream laws were refreshed from the official GitHub API on
2026-08-09. SGLang PR #33666 was merged 2026-08-06 and charges PP state from
the heaviest stage slice so capacity remains uniform. TensorRT-LLM PR #16170
was merged 2026-08-06 and drains pending relay sends before a potentially
blocking forward, while treating a missing relay as a loud error. The current
memra walker matches the relevant resource and ordering contracts.

Implementation verdict: **no engine change unless the current box1 trial
regresses**. The causal A/B is the bit-identical Lever-B reference itself:
`MEMRA_PRIME_PP=0` unsplit versus the naked PP-2 pipeline. The arms alternate
inside one lock hold, with one warmup and N=5 measured streaming requests each.

## Increment 2 — current-tip target-rig battery

Release binaries for commit `cf1b5c06` built on box1 with CUDA 13.2 and
auto-detected sm_120a. The engine gate binaries built in 3m34s and
`memra-server` in a further 20.47s; final build rc was zero. The measured server
SHA-256 is
`9dfd62a171d76c94088202349c713fb8d635159f22204cb017b0cc338e6b91df`.

One exclusive lock hold ran from 11:43:32Z through 12:01:08Z:

| gate | result |
|---|---|
| model-backed `kernel-check` | `ALL GREEN`; two unrelated Qwen3.6-only sections explicitly skipped because that model is absent on box1 |
| prime split identity | unsplit / serial split / dynamic pipeline bit-identical for logits, hidden stack, seed, and eight continuation steps |
| prime split liveness | auto: 8 split chunks and 7 overlaps; chunk 513: 10 split chunks and 9 overlaps |
| prime split canary | exact bits and split counts retained; overlap forced to 0/0, correctly detected RED |
| `chunkinv35` / canary | exact at 4096/513/512/256/64 / historical SWA seam diverged |
| `tickinv35` / canary | exact at budgets 0/1024/513/512/256/64 and splits 64/256/512 / call-local seam diverged in every nonzero/split arm |
| PP-2 `run-gen` | prefill/decode and batched-prime/tokenwise argmax 6776 MATCH |
| PP-2 `run-spec` | K=1..8 self-consistency PASS; pinned 14/17 then 15-token acceptance retained |

Both acceptance boots independently proved the residency-flip contract:
device 0 selected RESIDENT for its 45.72 GB expert slice and device 1 selected
RESIDENT for its 55.35 GB slice, each against about 94.9 GB of per-card expert
budget. The hold began at 26/27 C with 0 MiB and released at 34/35 C with 0 MiB.

## Increment 3 — interleaved serving verdict

Current policy check: `MEMRA_MOE_GROUPED` is off by default. This receipt is
therefore Lever B with the integrated dynamic schedule and solo-fresh outer
prefill, not the opt-in grouped expert arm.

Protocol: rendered 4k turn (4107 prompt tokens), streaming first-visible-token
TTFT, spec off, unique cold cache salts, trunk plus MTP draft, one warmup per
server boot, then one measured request. Five pairs alternated arm order
`U/P, P/U, U/P, P/U, U/P` inside one lock hold from 12:02:22Z to 12:14:57Z.

| arm | measured TTFT samples, sorted | p50 | range |
|---|---|---:|---:|
| unsplit (`MEMRA_PRIME_PP=0`) | 27.798, 27.805, 27.814, 27.862, 27.919 s | **27.814 s** | 27.798-27.919 s |
| naked PP-2 pipeline | 9.766, 9.768, 9.771, 9.772, 9.814 s | **9.771 s** | 9.766-9.814 s |

The pipeline is 2.847x faster than the same-window unsplit reference and removes
64.87% of client TTFT. Its median clears the owner stop line by 0.229 s. Every
measured request reported exactly 4107 prompt tokens; the combined JSONL has 10
warmup rows and 10 measured rows.

GPU0 snapshot temperatures spanned 27-36 C and GPU1 28-39 C. Every inter-arm
stop returned both cards to 0 MiB, as did final lock release. Server-log fault
scan found no CUDA error, illegal address, OOM, panic, Xid event, request error,
or server death.

**Stop verdict:** target met on the current naked policy with all required gates
green. Do not add another engine mechanism tonight. Follow-up work that does not
move this serve-trial line remains parked.

Primary receipts:

- `raw/box1/build/build-20260809T113844Z.log`
- `raw/box1/gates/gates-summary-20260809T114332Z.log` plus per-gate/probe logs
- `raw/box1/ttft/ttft-ab-summary-20260809T120149Z.log`
- `raw/box1/ttft/ttft-ab-client-20260809T120149Z.jsonl`
- `raw/box1/ttft/server-{pipe,unsplit}-p{1..5}-20260809T120149Z.log`
