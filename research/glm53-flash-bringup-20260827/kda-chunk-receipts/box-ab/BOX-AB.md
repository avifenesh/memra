# Chunked KDA prefill scan — box A/B (2026-08-29/30 window)

Pre-registration: `../README.md` (battery section). Executed as registered; the knee sweep
was conditional on the lever paying and did not run (verdict below states why).

## Provenance

- Engine: `5e33d338fbaf` = merge of `lane/glm5-kda-chunk-scan` (`ef94a78b04`) onto the
  bringup head `bac42f759d` (L1 flip + L2 lane). Branch `ab-l3-20260830`, pushed to origin.
- Binary `d93afe4785...` rebuilt in 4m57s on the box; binary-newer-than-sources PASS
  (receipt line in `l3-run.public.log`). Prompts `de57a7a471...` = L2's real prompt set.
- Serving shape: L2's 3-card PP (`serve.sh`: PP_STAGES=3, SPLITS=15,30, RESIDENT_GB=98,
  SLOTS=16, PREFIX_CACHE_MB=0, TF32 off, ctx 8192), 4x RTX PRO 6000 Blackwell Server 96 GB
  box class, cards 0 MiB before and after.
- BASE ENV both arms = L2's **arm C**: `MEMRA_MOE_GROUPED_PREFILL=1 MEMRA_BF16_MMV=1
  MEMRA_PP_BF16=1`. Arms one flag apart: k0 = base, k1 = base + `MEMRA_KDA_CHUNKED=1`.
- Arm identity per boot (the receipts law): every boot logs `[kda-chunked] flag=on/off`;
  k1 boots log `execute` 170x = 34 KDA layers x 5 rows (warmup + 3 greedy + sampled), and
  every boot re-proves the base: `gpf_execute=42`, `mmv_resident=148`, `bf16tc flag=on`.
- Structural receipt: the prime hands KDA the WHOLE prompt per layer call (execute lines
  `t=4626/5547/6467`, `nc` up to 102 at C=64), not 4096-token chunks — every engaged call
  was deep in the chunked regime.

## Real-weights band (MEMRA_KDA_DIFF=1 boot, 170 calls, T in {427, 4626, 5547, 6467})

Worst over all calls (per-element rel = |d| / max(|x|,|y|,1e-3), the gdn_scan_diff stat):
out max_abs 5.2e-6 / max_rel 8.7e-4; state max_abs 2.1e-4 / max_rel 1.2e-2. Wider than the
small-fixture band (real 64-head weights, up to 102 chunk-boundary carries per call); output
absolute error stays in the 1e-5 class of the core's O(1) scale, and the acceptance
authorities below (first-token argmax, drift depth) hold. Full lines: `kda-diff-lines.txt`.

## The table (interleaved x5 fresh boots, greedy TTFD seconds, mean [all 5])

| row | k0 (base C) | k1 (+KDA_CHUNKED) | delta |
|---|---|---|---|
| A4630 | 6.844 [6.82 6.95 6.83 6.81 6.81] | 6.846 [6.82 6.85 6.88 6.85 6.83] | +0.03% |
| B5550 | 8.165 [8.15 8.18 8.15 8.21 8.13] | 8.253 [8.19 8.49 8.28 8.15 8.16] | +1.1% (one 8.49 outlier; medians 8.15 vs 8.19) |
| C6470 | 9.488 [9.50 9.43 9.42 9.67 9.40] | 9.438 [9.43 9.45 9.45 9.44 9.43] | -0.5% |
| A4630 sampled twin | 6.632, prefill 697.6 tok/s | 6.655, prefill 695.1 tok/s | +0.3% |

Prefill 672-685 tok/s both arms on greedy rows; decode 24.6-26.1 both arms (32-token rows,
content-dependent). Sampled twin: 200s, full prefill speed, both arms; no spec route is
armed in this shape, so the vendor-default receipt is the sampled row itself plus the arm's
engagement lines (there is no K>0 counter to quote).

## Correctness on the flip candidate

- First-token argmax: IDENTICAL both arms, 5/5 on all three prompts ("The"). No flipped
  position, so the 8-draw census trigger never fired (census.py stood armed).
- Full-32-token greedy shas differ between arms but are STABLE 5/5 WITHIN each arm:
  deterministic band-class trajectory drift, the accepted GDN-A4 precedent class. Depth:
  A4630 and C6470 agree 55+ chars into the output; B5550 diverges at char 21 (word ~5,
  "actively-growing" vs "mature").

## Verdict: the lever does not pay. Recommend HOLD (flag stays DEFAULT OFF).

At 4.6-6.5k prompts on top of L1+L2 (arm C), the chunked KDA scan moves TTFD by less than
run-to-run noise (+0.03% / +1.1% / -0.5%). That bounds the sequential KDA scan's share of
the remaining wall at these widths to ~<=1% (~<=100 ms of 9.4 s at 6.5k), even though the
scan runs whole-prompt-serial (t=6467) per layer. The ~5.9 s residual at 6.5k belongs to the
other terms (L4 host-sync diet, remaining MoE/trunk work, L5 prefix cache), not the KDA
scan. The knee sweep (MIN_T x CHUNK) was skipped per pre-registration: no chunk-size choice
turns a noise-level term into a win.

L3 closes ATTRIBUTED-NEGATIVE at current serving widths: the kernel work is landed, gated,
and shelf-ready behind the default-OFF flag with its band receipts; the flip condition would
be a future regime where the scan term is material (much larger single-call t with the rest
of the wall already collapsed). Owner call: accept HOLD or direct otherwise; no self-flip.

## Box hygiene

Final server stopped PID-verified (pid checked against `/proc/<pid>/cmdline` before TERM),
all four cards 0 MiB after, no memra-server processes left. `~/gpf-ab/` and the drafter
clone untouched. Raw (unredacted) rows and log banked privately; this directory carries the
public twins (output text heads redacted, same discipline as L2's `.public.json` rows).
