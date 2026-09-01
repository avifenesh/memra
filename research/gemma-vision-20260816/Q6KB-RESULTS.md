# Q6_K batched-tier arc — results (lane/gemma-q6kb, 2026-08-17)

Coordinator-funded follow-up to ZOO-FUSION verdict 3. Question: the shipping trunk
(shipQ6K-downQ6K) pays a Q6_K m=8 matvec running 862 GB/s where the Q8_0 tier gets
1.28-1.47 TB/s. Receipts: gap-receipts/q6kb/ + /data/memra/evidence/gemma-fused2-ab
(prof-q6kb-decode, prof-r2check, prof-q8down-c8, prof-kqrp, kqf-* cells).

## Diagnosis chain (each step banked)

1. Decode-dominant c8 re-profile at ship HEAD: `qmatvec_q6_K_mmvq_b8` (ffn_down) =
   18.6% of decode GPU time, med 88 µs = 862 GB/s. Head's `_b8_r2` = 1.34 TB/s.
2. Codec-vs-shape discrimination on the Q8_0-down artifact: `q8_0_mmvq_b8_rp` on the
   SAME down shape = 80 µs = 1.47 TB/s — **Q6_K down costs MORE absolute time than
   Q8_0 despite 36% fewer bytes**. It's the codec, and it's the v2 bank's c8 dent
   mechanism at kernel level.
3. r2 probe (`MEMRA_KQ_BV=r2`): **FLAT** — kernel med 88→89.9 µs, serving 236.8→236.0.
   Recorded kill; the winners-table rule holds on this shape.
4. Root cause: 210-byte q6_K superblocks are NEVER 16B-aligned in GGUF layout — 16
   word-loads per 32-group where q5_K's `_il` (176B, aligned) rides LDG.128. The house
   fix already exists: the H100 K-quant coalescing mirrors (`build_q6k_rp4` +
   `_b8_rp` twins, 2026-08-01) — behind a Hopper-only default. Third instance of the
   pattern (Q8RP, PP_F16, now KQRP).

## Shipped: capacity-keyed KQRP default

`MEMRA_KQRP` unset now admits the q4_K/q6_K split-plane mirrors iff free VRAM covers
the admissible mass + 8 GiB (env priority both ways; 24 GB rigs refuse by
construction — verified boot). Kernel: `_b8_rp` med 88 → **66 µs (1.15 TB/s, −25%)**.

Gates (hygiene rules: invocations banked in gap-receipts/q6kb/): greedy tokens
IDENTICAL mirror on/off (m=1 `_rp` singles are layout-only, rp law); calibrated
argmax-margin-gate PASS both arms (banked logs); dflash acceptance EXACT (0.573,
82/143, agreement 128/128); m=1 plain 65.5 → 66.8 (+2%, the singles ride `_rp` too);
fresh-boot output samples on every serving boot.

## Certification cells (Japan GPU1 @450W, interleaved, dead-flat)

| cell | KQRP off | KQRP on (capacity default) | delta |
|---|---|---|---|
| c8 cold agg (×5) | 234.8 (234.8–235.3) | **246.6 (246.4–246.8)** | **+5.0%** |
| c8 cold per-stream | 33.29 | 35.16 | +5.6% |
| c16 cached agg (×3) | 232.2 | **243.7** | **+5.0%** |
| c16 cached per-stream | 16.99 | 17.98 | +5.8% |

The v2 bank's −5.9% downQ6K c8 dent is fully recovered at the decode tier (this arc)
on top of the prefill-wall fix (ZOO-FUSION §2b). Ship lane to re-bank c8/c16 per the
coordinator — posted to LANE-STATUS.md; not re-banking the ladder here.

## Sized, not taken

- Q6_K `_il`-style issue reduction on the GGUF-layout base kernel: superseded — the rp
  mirror achieves alignment structurally; the base kernel only serves no-mirror rigs
  (24GB, where the 31B trunk doesn't fit anyway).
- The remaining 1.15 vs 1.47 TB/s gap on the mirrored kernel (~22%): the two-stream
  6-bit unpack cost proper. A `_vl`-class or unpack-restructure arm could chase it
  (+~2% c8); next-tier EV after the ship re-bank.
