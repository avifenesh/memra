# matmul_nvfp4_fused2 — results (lane/gemma-fused2, 2026-08-16)

Arm: `qmatvec_nvfp4_mmvq_fused2_rp` + `matmul_nvfp4_fused2` (m==1, `MEMRA_NVFP4_FUSED2=0`
rollback seam), chained after the q4-fused miss at the gemma4 dense decode sites (SWA
trio = fused2(q,k) + Q8_0 v single; global = fused2(q,k), V:=K; ffn + shared gate/up).
Motivation: gemma4 NVFP4mix keeps attn_v/ffn_down at Q8_0, so the all-NVFP4 fused3
never matches and decode fell to per-tensor singles (NVFP4 55.2 < Q4_0 72.4 c1).

## Kernel-check pin (fused2-pin bin) — BIT-IDENTICAL

Real loader (rp repack), fused2 vs two `matmul_pre` singles at `f32::to_bits` equality:
- Q38 NVFP4 (local 5090): **12/12 pinned, 0 FAIL** (attn 12288/1024 + ffn 17408/17408).
- gemma-4-31B NVFP4mix (Japan): **20/20 pinned, 0 FAIL** (both attn geometries:
  SWA 16384/2048, global 8192/4096 + ffn 21504/21504).

## Gates
- Engine + server suites: 267/267 (server), engine ok.
- Greedy byte-identity, gemma4 NVFP4mix, seam OFF vs ON: **BYTE-IDENTICAL** over 8 steps.

## Serving A/B (interleaved, Japan GPU1 @450W, memra-server c1, NVFP4mix)
| arm | decode p50 tok/s |
|---|---|
| MEMRA_NVFP4_FUSED2=0 | 57.2 (flat, n=2) |
| MEMRA_NVFP4_FUSED2=1 | 58.2 (flat, n=2) |

**Delta: +1.8%.** Real, dead-flat within arm, bit-identical. (2 of 3 planned reps per
arm produced valid cells; the flat within-arm spread makes the small-n call sound. Raw:
fused2-ab-cells.jsonl.)

Note: `run-gen`'s raw greedy loop reported 332 tok/s dead-flat both arms — a DIFFERENT
measurement class that does not traverse the eager/dc dispatch sites fused2 wires (it
takes the graph/slotted path, which the directive kept out of scope). The serving cell
above is the correct measurement.

## Verdict — arm is correct and default-safe; it is NOT the 55→72 gap-closer

fused2 fuses only the uniformly-NVFP4 PAIRS. The Q4_0 advantage the gap reflects comes
from `matmul_q4_fused3` fusing the FULL q/k/v trio (Q4_0 keeps v at Q4_0, so all three
join one launch), whereas NVFP4mix's v and ffn_down are Q8_0 singles that fused2 cannot
absorb. So +1.8% is the ceiling this arm can reach; the remaining ~55→72 gap lives in
(a) the Q8_0 v/down singles themselves and (b) whatever else the eager gemma4 dense
step pays that the Q4_0-fused path avoids — a separate diagnosis, not this kernel.

Recommend: default-on is harmless (bit-identical, +1.8%) but not worth a release on its
own. Fold it into the batched-decode / gap-diagnosis lane rather than shipping solo. The
gap-closer is elsewhere; fused2 removes one of its contributors and proves the NVFP4
pair-fusion primitive exists for any future all-NVFP4 gemma pair.
