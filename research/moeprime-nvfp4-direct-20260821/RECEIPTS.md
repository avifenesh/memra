# moeprime — prime-wall increments after the v0.99.0 cells (2026-08-21)

Owner: keep tuning until memra beats vLLM on the realistic cells. Post-v0.99.0 prime
anatomy (box3, one RTX PRO 6000 Blackwell, ornith15 14.7k prompt): gdn_linear 433ms /
moe 637ms / attn_full 273ms; naked pp14715 = 12,036 tok/s.

## Increment 1: NVFP4 direct sk tile loader (f16g dequant-workspace pass dies)

The v0.99.0 NVFP4 f16g admission ran the workspace path (537MB f16 dequant write + GEMM
read per projection per layer). The kq-direct precedent (research/q4k-expert-prefill §5:
dequant passes = 41.8% of f16g GPU time) extends cleanly: NVFP4's 36B block holds 4
UE4M3-scaled sub-blocks of 16 values, so the direct loaders' 16-value per-thread window
IS one sub-block — never crosses a scale boundary. `kq_fetch`/`kq_store` NVFP4 arms are
the workspace kernel's exact DAG (UE4M3 scale decode + pre-converted-float mxfp4
codebook in shared, the IQ4_XS trick).

- kernel-check `f16g-kq-direct [nvfp4 synth]`: **byte-identical (maxdiff=0)** across all
  4 visitor forms (hybrid/all-128/all-32-deep/all-32-legacy), random UE4M3 scales.
- pp14715 A/B (direct naked vs MEMRA_F16G_DIRECT=0): 11,773/12,107 vs 10,977 = **+7-10%**.
- Prime MoE stage 637 -> 519.8 ms (anatomy receipt).
- Cachecell: session 20.89 -> 19.77s (cold turn 3.98 -> 3.55s), sharedc8 661.8 -> 682.8.

## Increment 2: GDN K4/K5 mma pair — naked default ON for sm_120a builds

The pair was qualified on 90a (bf16 HMMA m16n8k16 = sm_80-class PTX; only the wgmma nest
is Hopper-gated via MEMRA_K45_REAL at __CUDA_ARCH__==900) and left env-opt-in elsewhere —
no Blackwell verdict existed. Measured (interleaved, both orders):

- box3 PRO 6000, ornith15 pp14715: flag ON 12,716/12,825/12,945/12,957 vs OFF
  12,007-12,114 = **+5-8%**.
- local 5090, q38-27b pp6435: ON 1,427/1,446 vs naked 1,397/1,429 = +1-2%, no loss.

Default flip per the per-hardware doctrine: `gdn_mma_default_on()` = hopper build OR
sm_120a build. **Defect found & fixed while flipping:** MEMRA_GDN_MMA had THREE read
sites (gdn_mma_enabled, the k123 pre-work, gdn_scan_chunked's dispatch); flipping only
two armed the mma pre-work (wb16/kb16 mirrors) while the scan took the scalar route —
measured as a 0.8% LOSS vs clean OFF. All three sites now share the one const helper.
Also closed: MEMRA_GDN_WGMMA=1 on a non-Hopper build could reach the EMPTY-bodied wgmma
kernel once the mma default flipped — gdn_wgmma_on/`_pre` are now hard-gated to
cfg!(memra_hopper_mma).

- Naked verify (box3, interleaved): naked 12,595/12,945 vs MEMRA_GDN_MMA=0
  12,029/12,103.
- Gates on the naked default, both rigs: kernel-check ALL GREEN (85 cells box3, 107
  local — incl. the gdn mma/f32 pinned-config cells), run-gen argmax MATCH, run-spec
  K=1..8 PASS, argmax-margin-gate PASS on the 14.7k prompt (flips=1 bad=0, the known
  near-tie), serve-smoke 0 failed on both rigs (local incl. Q35 mixed c=4 + spec +
  gemma4 arms).

## Prime wall ledger (pp14715, ornith15, box3)

| state | tok/s |
|---|---|
| campaign start (per-pair _em experts) | 2,331 |
| v0.99.0 (f16g NVFP4 workspace) | 12,036 |
| + NVFP4 direct loader | 12,107 |
| + GDN mma naked | **12,595–12,945** |

Anatomy after both: moe ~520ms, gdn_linear ~360ms (mma), attn_full ~273ms.
