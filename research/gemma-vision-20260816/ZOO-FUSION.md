# Zoo-fusion arc — per-op budgets, shipped fusions, and the honest ceiling (lane/gemma-zoofusion, 2026-08-17)

Owner question: "why 61 and not 71?" — answered with profiles, two ships, and arithmetic.
Baselines re-taken on ship HEAD; shipping trunk = shipQ6K-downQ6K per owner acceptance.
Receipts: gap-receipts/ + /data/memra/evidence/gemma-fused2-ab (prof-zoo-*, prof-serving,
prof-c8b, cells).

## 1. The re-profiled budgets (µs/token unless noted)

**c1 window (eager arm, embdQ6K trunk, 16612 µs = 60.2 tok/s):** matvecs 13984 (84%),
zoo 1246 (7.5%: exit-fold 337, entry-fold 320, gelu 271, rope+append 225, quantize 72),
attention 824 (5%), launch gap 558 (3.4%). The old "rms×121" zoo is GONE — pn-fold
collapsed it; the two remaining fold kernels are already maximally fused (two full-row
reductions each, 5.3–5.7 µs).

**c1 serving:** the batched lane's gemma4 arm is default-on since its merge — serving
c1 rides decode_step_batch's B=1 tier, NOT the eager arm. Measured routing A/B:
eager 61.48 vs batched 61.33 (+0.2% eager) — parity; batched default stays (its own
lane's gates own that call).

**c8 serving (downQ6K trunk, the owner's dent):** `qmatvec_gemm_q6_K` = **30.2% of all
GPU time at 3.46 ms per layer-call** (the known "Q6_K has no MMQ arm" dequant-GEMM
prefill wall — now on THE shipping trunk's ffn_down), ttft 1.38 s. Decode-side Q6_K
m=8 (q6_K_mmvq_b8) runs ~840 GB/s vs Q8_0's 1.26 TB/s — real but second-order next to
the prefill wall.

## 2. Shipped

**(a) Eager qkv norm+rope+KV-append fold** (host-len twin of the dc fold, shared inlined
body — twins cannot drift). Bit-identical to the pair by construction; token-identity
seam 0/1 GREEN on 5090 (gemma4-12B) + Japan (NVFP4mix-Q6K, Q4_0). Window 60.2→60.5
(+0.5%); serving-neutral (serving rides the batched arm, which has its own folds) —
ships as eager-path parity. `MEMRA_QKV_APPEND=0` (the dc seam) reverts.

**(b) Capacity-keyed f16 prefill mirrors — the arc's prize.** The house f16 Lt lane
already admits Q6_K (round 47) but defaulted Hopper-only. With MEMRA_PP_F16/MEMRA_Q4F16
unset, the gemma dense walk now admits mirrors iff free VRAM ≥ admissible f16 mass +
8 GiB (env keeps priority both ways; 24 GB rigs refuse by construction — verified 5090
boot builds none; the Q8RP-hijack class is dead per abf155e8 and this lane's full-phase
gate law applied).

Probe (downQ6K, c8 ×3 + c1, dead-flat): **agg 172.5 → 235.8 (+37%)**, **ttft 1.379 →
0.408 s (−70%)**, c1 agg 56.3 → 61.5 (+9%), decode bits untouched (64.03/64.05).
Gates (CORRECTED 2026-08-17, ship-lane merge diligence — the original cert line here
overstated a smoke as a gate; standing rule applied: a skipped gate is never written
as PASS, and every cert line carries its banked invocation):
- What was originally run: a run-gen boot probe (`CUDA_VISIBLE_DEVICES=1 MEMRA_CHAT=1
  MEMRA_NGEN=32 run-gen <shipQ6K-downQ6K> --prompt "Explain binary search briefly."`)
  whose internal single-position assert did not fire — a smoke on a 28-token prompt,
  NOT the calibrated argmax gate. The calibrated gate had NOT been run.
- The calibrated gate, run and banked (gap-receipts/argmax-gate/banked-invocations.txt;
  raw logs /data/memra/evidence/gemma-fused2-ab/argmax-gate/): the ship agent's build
  of argmax-margin-probe showed flips=2 at pre-merge HEAD invariant of silicon and
  mirror arm — both flips at margins (0.135, 0.199) inside the measured config spread
  (up to 3.03) — the documented near-tie class tripping a flip budget calibrated for
  the thin-near-tie qwen class. Fixed by a gemma-4-31B calibration row (3 explained
  flips per 12-window, derived from the banked margin distribution p10 0.293 / p50
  2.28, NOT loosened-to-green; every flip must still be individually margin-explained).
  Re-run at HEAD: `tools/argmax-margin-gate.sh <shipQ6K-downQ6K>` **PASS on both
  mirror arms** (flips=2, bad=0) and `--canary` teeth intact (injected wide-margin
  flip rejected). Ship-lane attribution receipts:
  /data/memra/evidence/gemma-ship-20260817/zoofusion/.
- Output samples healthy every boot; **dflash acceptance on the shipping trunk EXACTLY
  unmoved (0.573 = 82/143 both arms, agreement 128/128)** — invocation banked in §2's
  battery (gemma-gate + MEMRA_SPEC_DFLASH + banked accept5 IDS, f16 on/off arms) —
  prefill numeric class change (f16 vs int8, argmax-gated per round-45 law) costs zero
  draft quality on the banked tape.
Certification cells: see table below (final-*.log carries per-boot output samples).

## 3. Recorded / not shipped

- **CUDA graphs stay dead** (re-confirmed −12% both artifacts in this lane's earlier gates).
- **fa combine-q8 emit for the eager arm**: kernel exists (E4B wave 5b), fa_decode +
  rows_w already plumb `q8_out`; the hot sub-window path (fa_decode_kvmod→_view) needs
  the param through 2 wrapper layers + 3 exit rewires. Sized: kills the 60/tok
  standalone quantize (~72 µs + slots ≈ +0.5% eager-only — serving unaffected). Next-arm.
- **Q6_K m=8 decode kernel (varlen playbook)**: 840 GB/s vs Q8_0's 1.26 TB/s in the
  batched tier — worth ~+1% c8 agg AFTER the prefill wall fix; re-rank once the f16
  cells land (the wall dominated the dent).
- **Mixed nvfp4+q8 trio kernel**: stays dead (launch-slot-only win, PDL already covers).
- **Launch gap (~558 µs c1)**: host submission cost of ~730 launches/tok; graphs lose,
  megakernels falsified (FFN slab #7). Structural until launch count drops by class.

## 4. The honest ceiling arithmetic (owner's "why not 71")

downQ6K trunk bytes/token ≈ **18.6 GiB** (census-derived: NVFP4 mass 11.13 + Q8_0 v
1.09 + Q6_K down ~5.30 + Q6_K embd/head 1.08). At the measured matvec plateau
(1430–1520 GiB/s = 86–91% of the card's 1669 GiB/s peak):

- matvec floor ≈ 12.6–13.4 ms/token
- + attention 0.8 ms + folds/gelu 0.9 ms + gap 0.55 ms ≈ **14.9–15.7 ms → 64–67 tok/s**
- measured plain: 64.0–65.5 ✓ — the engine is at 93–97% of its own realistic ceiling
  on this trunk. **c1 ≈ 71 is NOT reachable by kernel work on 18.6 GiB/token**; it
  needs ~16.5 GiB/token (the recipe lane's v/down ladder) plus the residual shavings
  above. The compounding math: bytes to 16.5 GiB → floor ~11.5 ms → ~+8% → ~69–70 with
  the current non-matvec budget. That is the path; nothing else on the menu reaches it.

## 5. Certification cells (shipQ6K-downQ6K, Japan GPU1 @450W, interleaved, fresh-boot
output-sample gate every boot, dead-flat)

| cell | f16 mirrors OFF | f16 mirrors ON (capacity default) | delta |
|---|---|---|---|
| c8 agg tok/s (×5) | 173.5 (171.7–174.2) | **236.0 (234.4–236.3)** | **+36.0%** |
| c8 ttft p50 | 1.609 s | **0.431 s** | **−73%** |
| c8 per-stream decode p50 | 30.50 | **33.33** | **+9.3%** |
| c1 agg tok/s (×3) | 56.2 | **61.5** | **+9.4%** (all ttft) |
| c1 decode p50 | 64.03 | 64.06 | untouched ✓ |

NOTE (protocol honesty): these absolutes are cold-prompt/uncached cells — not
comparable to the recipe lane's prefix-cached c8 bank (255.8/271.7); the DELTA is the
certified claim. The owner's downQ6K c8 dent (−5.9%) is not merely clawed back — its
dominant mechanism was the Q6_K prefill wall, and with it removed downQ6K wins every
width on this protocol. The decode-side Q6_K m=8 kernel gap (~840 GB/s vs 1.26 TB/s)
remains the sized follow-up (verdict §3) if the batched lane's cached-c8 protocol still
shows a residual dent on the fixed build.

Merge notes: (1) acceptance re-banked unmoved (0.573) on the shipping trunk — the
recipe lane's f16-off banks remain valid as f16-off references; (2) the batched lane's
serve-stream gate should re-run at merge as usual (shared kernels bit-identical, f16
changes prefill class only, argmax-gated).
