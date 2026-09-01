# Deep-load curve progress

Date: 2026-08-12
Lane: `lane/cx-loaddepth`
Starting head: `8b2ba8c88`
Rig: box1 (`ubuntu@<rented-box-ip>`), 2x RTX PRO 6000 Server Edition

## Status

COMPLETE — exactness is green through c=24 and the N=3 deep-load curve is
frozen. The first non-rising width is c=20; c=16 is the revenue optimum at
171.419 aggregate tok/s (14.811M tokens/day).

## Frozen execution order

1. Reconstruct the merged Step-3.7 PP-2 server recipe from
   `research/dualpp2-20260811/box1-soak-fixed.sh`, using the naked arm with no
   dual/overlap environment overrides.
2. Queue behind `cx-specpp2fix`, then hold `/tmp/memra-gpu.lock` once for the
   complete campaign.
3. Run `qos_probe` golden-identity gates at c=12,16,20,24 in ascending order;
   stop at the first divergence or error.
4. For the clean width range, measure three interleaved 128-token sustained
   windows at c=8,10,12,16,20,24, retaining aggregate tok/s, TTFT p50/p99,
   step p99, admission counters, thermal state, and raw JSONL/logs.
5. Report the first throughput/15-second-TTFT knee, the revenue-optimal width,
   and its tokens/day in `RESULTS.md`.

## Guardrails

- Measurement-only lane; code changes are permitted only if a hard cap blocks
  the requested measurement.
- Exactness precedes timing at every new width; a failing width is never timed.
- One exclusive box1 lock hold covers the whole scored campaign.
- Source must be `~/memra-dualpp-flip` at `e94699eba` or a later merged-default
  commit, with the exact source revision recorded.
- No merge, tag, push, perf-board edit, formatting pass, or hook bypass.
- Raw evidence and the N=3/thermal regime are part of the deliverable.

## Log

- 2026-08-12T01:26:14+03:00 — Read the owner brief and steering inbox, confirmed
  a clean dedicated worktree on `lane/cx-loaddepth` at `8b2ba8c88`, and opened
  the lane before remote measurement work.
- 2026-08-12T01:34:59+03:00 — Recovered the exact merged-default flip battery
  from box1 and verified the live clean checkout at `0edc57b9c` descends from
  `e94699eba`; its only later commits contain evidence/docs, so the runtime tree
  and staged server binary remain the scored flip. `fuser` and `lsof` reported
  no holder for `/tmp/memra-gpu.lock`. Frozen a research-only driver with one
  lock hold, ascending exactness stop gates, fresh naked servers per perf point,
  same-width discarded warmups, three rotated width orders, 250 ms thermal
  sampling, streaming content TTFT, worker step p99, and admission deltas.
- 2026-08-12T01:37:12+03:00 — The first launch was externally stopped with
  exit 143 by the transient user-service lifecycle while hashing inputs. It
  never reached `SERVER_START`, no GPU compute app appeared, and its receipts
  are preserved under `raw-aborted-user-manager-stop-20260811T223712Z/` rather
  than mixed into scored evidence.
- 2026-08-12T01:38:39+03:00 — The scored foreground campaign acquired the
  exclusive box1 lock as PID 912106. Exactness passed at c=12,16,20,24: 72/72
  completions matched `21b8293f`, with zero errors or divergences.
- 2026-08-12T01:59:53+03:00 — `LOADDEPTH_ALL_PASS`. All 18 N=3 points and
  270/270 scored requests completed exactly 34,560 output tokens. Every width
  recorded zero session defers, VRAM defers, OOM parks, sampled queue depth,
  and dual-slot collisions. Medians rise from c8 156.741 to c16 171.419 tok/s,
  fall at c20 to 164.911, and recover only to 169.005 at c24; scored p99 TTFT
  remains below 1.523 seconds throughout.
- 2026-08-12T02:03:00+03:00 — Retrieved 175 scored raw files plus the excluded
  preflight receipts. Both SHA-256 manifests verify locally, the reducer output
  is byte-identical to box1 (`b35659a3...`), all JSON/JSONL parses, and the
  scored server-log failure scan is clean. No runtime, board, merge, tag, push,
  or formatting change was made.
