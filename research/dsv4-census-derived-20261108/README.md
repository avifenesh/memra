# DSv4 full-token census: counts derived from the loaded program (2026-11-08)

Verdict: the census pins in `dsv4_compose_cadence_dense_gate` (total kernels 2741/3140/3240,
2026-09-08) and `dsv4_hc_dot_split_gate` (standalone RMSNorm 129/171/191 and RoPE 150,
2026-09-09) were stale, not a regression. `Dsv4Gpu::full_token_forward_census_for_gate` now
counts the full-token replay graph from the loaded program. On a 2x RTX PRO 6000 Blackwell
Max-Q pair both gates fail on main at the old pins and pass with the derived counts. The
derivation matches every graph it was checked against: four programs (compose candidate and
control, HC split on and off), every forward variant, both ranks.

Scope: DeepSeek-V4-Flash NVFP4, TP/EP placement with exact attention TP2, the f32x
small-kernel-diet program, sampled full-token replay. Untimed pass/fail cells, one run each. No
performance claim and no native qualification.

## Conditions

- 2x RTX PRO 6000 Blackwell Max-Q, 250 W limit left as found, CUDA 13.4, sm_120a release
  build, rust 1.97.1. Peak temperatures 88 C and 82 C, fans at most 70 %: no thermal stop.
- The pair cells' environment: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0,1
  MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_EXPERT_ARM=native MEMRA_DSV4_DENSE_ARM=fp8
  MEMRA_DSV4_EP=pair MEMRA_DSV4_GROUPED_ROUTE=device MEMRA_DSV4_VERIFY_TOPK=device
  MEMRA_DSV4_PREFILL_MOE=reference MEMRA_DSV4_DOTS_ARM=f32x MEMRA_DSV4_DRAFTER=off
  MEMRA_MOE_F16G=2 MEMRA_F16G_SK=32 MEMRA_DSV4_SAMPLER=device`, the gates run as
  `<gate> <model> <tape> <new-dir> [--defaults]`.
- `before-main/`: main 958ae32fed. `after-fix1/`: 958ae32fed plus the change as first built
  (patch sha256 d39a00a1a316...). It differs from ee524c6dd8 only in the census loop's form,
  `for il in 0..n_trunk` with `let cache = &caches[il]` where the commit iterates
  `caches.iter().enumerate()` (clippy `needless_range_loop`); the counts are the same code.
  Each directory has the raw gate logs, the cell status lines, the binaries' sha256 and the
  1 s telemetry.

## Runs

| cell | before (main) | after |
|---|---|---|
| compose cadence `--defaults` | rc 101: 1680 total-kernel mismatches, every other check passed. Read 2546/2945/3045 for slots 0/2/3 against 2741/3140/3240 | rc 0: want 2546/2945/3045 |
| HC dot split `--defaults` | rc 101 at the RMSNorm pin, rank 0 segment 0: read 0 against 129 | rc 0: kernels 2248/2647/2747, RMSNorm 0/42/62, RoPE 0 |
| compose cadence A/B (both arms) | rc 101: 2288 mismatches; the control arm reads 3045 against 3240 | rc 0: candidate 2546/2945/3045, control 3045 |

The HC gate's RoPE pin (150 per forward variant) was stale too: it is the assert after the
RMSNorm one, and the graphs hold no `dsv4_rope_kernel` node.

## Derivation

Per rank and forward graph; both ranks hold the same forward graphs. The trunk has 43 layers: 2
window layers (ratio 0), 21 ratio-4 layers (indexer, position-split C4 store) and 20 ratio-128
layers. Terms are the helper's, in walk order (`decode_step_tp_ep`, `tp_ep_trunk_walk_rows`).
The compose gate freezes `MEMRA_DSV4_DENSE_FAST=0` and `MEMRA_DSV4_HC_DOT_SPLIT=0`; the HC gate
runs the defaults (dense fast on, HC split S16).

| term | counted per | compose program | HC split program |
|---|---|---|---|
| replay input, embedding, HC repeat | graph | 3 | 3 |
| attention HC entry | layer | 4: HC dots, small HC, RMSNorm, bf16 pack | 2: split partials, fused finish (norm and pack inside) |
| wq_a + wkv | layer | 2 | 1 pair launch |
| q norm pack | layer | 1 | 1 |
| wq_b | layer | 1; 2 on ratio-4 (the indexer's wq_b) | 1; a pair on ratio-4 |
| q head RMS + RoPE | layer | 1 | 1 |
| latent norm + RoPE + QAT | layer | 1 | 1 |
| indexer q chain + weights | ratio-4 layer | 2 | 2 |
| compressor append | compressor | 5: snapshot, kv dots, gate dots, two row appends | 4: kv/gate pair |
| compressor emission | emitting compressor | indexer 10, ratio-4 main 9, ratio-128 5 | same |
| index list | layer | 1; 3 on ratio-4 (indices, score, top-k) | same |
| C4 split gather | ratio-4 layer | 1 | 1 |
| sink attention | layer | 2 | 2 |
| o inverse RoPE + pack | layer | 1 | 1 |
| wo_a | layer | 4 (the rank's groups) | 4 |
| attention TP2 gathers | layer | 2 | 2 |
| wo_b | layer | 2: bf16 pack, GEMV | 2 |
| attention hc_post | layer | 1 | 1 |
| FFN HC entry | layer | 3 (no pack) | 2 |
| router | layer | 2: dots, route | 2 |
| routed experts | layer | 11: x QAT, three route kernels, input gather, fused gate/up, h QAT, h gather, M1 down, scale, slot scatter | 11 |
| expert reduce | layer | 1 | 1 |
| shared expert + hc_post | layer | 9: combine, cvt, gate and up, SwiGLU, cvt, down, add, hc_post | 8: gate/up pair |

The emission is pool, RMSNorm, RoPE at the block, the FP4 rotation (Hadamard and QAT, 2) or the
FP8 QAT (1), the store (the C4 split store or a row copy) and, for the overlapping ratio-4
compressors, four shifts of the two pending planes: 3 + 2 + 1 + 4 = 10 for the indexer,
3 + 1 + 1 + 4 = 9 for the ratio-4 attention compressor, 3 + 1 + 1 = 5 at ratio 128.

| graph | compose program | HC split program |
|---|---|---|
| window / ratio-4 / ratio-128 layer | 49 / 65 / 54 | 44 / 57 / 48 |
| ordinary: 3 + 2 x window + 21 x ratio-4 + 20 x ratio-128 | 2546 | 2248 |
| C4: + 21 x (10 + 9) | 2945 | 2647 |
| C4 + C128: + 20 x 5 | 3045 | 2747 |
| control (no cadence: one forward graph in which every compressor emits) | 3045 | |
| RMSNorm in ordinary / C4 / C4 + C128 | 86 / 128 / 148: the two HC entries per layer, plus one per emitting compressor | 0 / 42 / 62 |
| RoPE (`dsv4_rope_kernel`) | 0 | 0 |

The q RoPE runs in the head-RMS launch, the latent's in the latent norm launch, the indexer q's
in its chain, the output's inverse with its pack, and the compressors' as
`dsv4_rope_at_kernel`. The commit graph holds the head's final norm on both ranks (the
vocab-parallel head), `full_token_commit_rmsnorm_for_gate`.

`census-compose-on.txt`, `census-compose-off.txt` and `census-hcdot-on.txt` are the per-symbol
kernel censuses of the after run's qualification dumps (`sym_census.py <dump-dir>`), the
counts each term above was read against.

## Why the pins drifted

The pins predate the small-kernel diet as the f32x program (8d423ced37, 2026-09-23), the
two-launch sink attention (aea8c69337, 2026-09-23), exact attention TP2 as the TP/EP program
with its two gathers (9d19b5014d, 2026-09-25) and the #710 launch fusions (q head RMS with
RoPE and the latent norm, RoPE and QAT in 70f46bece3, 2026-09-27; the indexer q chain, the
inverse RoPE with its pack, the paired GEMVs). The 2026-11-07 note put the compose gap (195 in
every variant) on the paired kernels and the fused HC finish; the compose gate freezes dense
fast and the HC split off, so its graphs hold neither. The 195 is the net of the changes above;
this lane did not decompose it per change.

## Limits

The helper models the program these gates run and refuses the rest: another placement, the
fused one-token MoE pair (`moe_fused_eligible`; the gates turn the route and mirror validation
off and pin the fused gate/up, which rules it out), an unfused gate/up or another down tail, a program without the
small-kernel diet, a pending small-component cell. A gate that needs one of those adds its
terms first.
