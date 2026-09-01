# PP2PIPE SERVE-TRIAL RESULT — box1, 2026-08-09

## Verdict

**TARGET MET.** The current naked PP-2 pipeline serves the rendered 4k turn
(4107 prompt tokens) at **9.771 s TTFT p50, N=5**, below the owner's 10-second
line by 0.229 s. The same-window unsplit reference is 27.814 s p50, so Lever B
is 2.847x faster and removes 64.87% of client TTFT.

No runtime change was needed. Base `8e8c93af` already contains the full
stage-local resident pipeline plus the later dynamic-microchunk and solo-fresh
outer-prefill increments. The deadline rule therefore stops this lane at
verification, harness, and evidence rather than adding another mechanism.

## Interleaved N=5 receipt

Box1, 2x RTX PRO 6000 Blackwell, PP devices 0/1. Step-3.7-Flash IQ4_XS shards
plus the Q8_0 MTP draft; spec off; streaming first visible token; unique cold
cache salts. Each server boot ran one excluded warmup and one measured request.
Arm order alternated `U/P, P/U, U/P, P/U, U/P` inside one exclusive lock hold.

`MEMRA_MOE_GROUPED` is off by default on this tip, so these numbers do not use
the opt-in grouped expert path.

| arm | p50 TTFT | measured range | N |
|---|---:|---:|---:|
| unsplit, `MEMRA_PRIME_PP=0` | 27.814 s | 27.798-27.919 s | 5 |
| naked PP-2 pipeline | **9.771 s** | 9.766-9.814 s | 5 |

The hold ran 12:02:22Z-12:14:57Z. Snapshot temperatures were 27-36 C on GPU0
and 28-39 C on GPU1. Both cards were 0 MiB at entry, between every arm, and at
release. The 10 server logs contain no CUDA fault, illegal address, OOM, panic,
Xid event, request error, or server death.

## Required gate battery

One preceding box1 lock hold passed every required production row and every
canary:

| surface | verdict |
|---|---|
| release build | PASS, CUDA 13.2, sm_120a auto-detected |
| `kernel-check` | ALL GREEN on the available Step-backed and synthetic sections |
| prime split identity | unsplit / serial / pipeline bit-identical; live split and overlap counters |
| prime split canary | PASS; overlap removal detected while split and exact bits stayed live |
| `chunkinv35` / canary | PASS / teeth |
| `tickinv35` / canary | PASS / teeth |
| PP-2 `run-gen` | argmax 6776 MATCH in both comparisons |
| PP-2 `run-spec` | SELF-CONSISTENCY PASS for K=1..8 |

The acceptance logs also show per-stage expert residency: 45.72 GB on device 0
and 55.35 GB on device 1 both selected RESIDENT against their own per-card
budgets. This directly proves the residency-flip part of Lever B on today's rig.

## Receipts

- `raw/box1/build/build-20260809T113844Z.log`
- `raw/box1/gates/gates-summary-20260809T114332Z.log` and its per-gate raw logs
- `raw/box1/ttft/ttft-ab-summary-20260809T120149Z.log`
- `raw/box1/ttft/ttft-ab-client-20260809T120149Z.jsonl`
- per-pair client JSONL and server logs under `raw/box1/ttft/`

The branch was not pushed. No release tag was created.
