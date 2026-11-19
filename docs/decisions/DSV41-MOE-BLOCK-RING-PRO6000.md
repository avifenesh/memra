# DeepSeek-V4.1: the grouped small-row body on a block-staged ring on the RTX PRO 6000 (2026-11-18)

> Since 2026-11-18 both rings are instances of the shared skeleton's `bs_small_moe_kernel` (`bs::Warp6` was
> `v41_moe_small_kernel`, `bs::Block2` was `v41_moe_small_blk_kernel`), the same SASS kernel for kernel:
> [BS-SMALL-SKELETON](BS-SMALL-SKELETON.md).

## Decision

On an RTX PRO 6000 Blackwell (sm_120 with 188 SMs) the routed experts' small-row calls (tiles of at most 8
rows: every decode, verify and batched call of a four-rank split up to 64 rows) run
`v41_moe_small_blk_kernel` instead of `v41_moe_small_kernel`. Both are the same program element for element,
so the choice is speed only. Every other card keeps the six-stage warp ring with its rule unchanged (the RTX
5090 and GB10 were not measured for this change). `memra_v41_moe_small_ring` lets the kernel gate force
either ring; nothing reads an environment variable.

## Why the warp ring was slow on this card

memra#1324: on four PRO 6000 ranks the routed experts at 4 to 10 rows ran 16 to 19% behind the pre-CUTLASS
kernels (`gemv_f4s_tiles` + `swiglu_f4s_tiles`, base `8031c7ef90`). Those kernels split K over warps, a
different numeric program, so they cannot come back. The warp ring gives each warp six stages of 4 K blocks
(64 bytes of a weight row a stage) and its own copy of the tile's activation rows, sized for the plan's tile
rows. At 4 rows that is 46.5 KB a block (two blocks an SM, 432 or 480 blocks on 376 slots); from 9 rows the
plan's tiles are 8 rows high though most hold one, and the block takes 60.7 KB (one block an SM, four warps).

## What was chosen

Two stages of 2 KB of weights a warp (16 columns x 8 K blocks, or 8 columns x 16 K blocks while that launch is
at most one wave of four blocks an SM), the activation rows staged once a block, a block barrier a stage. At the
shapes served calls take a block is 19.5 to 23.9 KB: four blocks an SM. The 8-column shape at 6 to 8 tile rows
takes 27.1 KB (three an SM); served calls choose 8 columns only at 3 rows or fewer, so only the kernel gate runs it.

## The measurement

Four RTX PRO 6000 Blackwell Max-Q (power limits 266-281 W, read at the gates step before the A/Bs and kept in no
receipt), 40 layers, four ranks, device-push, 2026-11-18,
nothing else on the box. A B B A B A A B, four runs an arm, steady calls past 0.5 s of load, ms; the range is
B / A of the four adjacent pairs of runs. B is `4e80f97800`. The decode cell's token tape is the same in all
eight runs of both A/Bs (`cd38e1b772b75d5a`).

| call | A = main `0db086f87a` | B | B / A | A = base `8031c7ef90` | B | B / A |
|---|---|---|---|---|---|---|
| one-row decode | 9.695 | 9.320 | 0.961 (0.960-0.962) | 9.915 | 9.313 | 0.939 (0.939-0.941) |
| 4 rows | 13.037 | 11.834 | 0.908 (0.907-0.908) | 13.027 | 11.828 | 0.908 (0.907-0.910) |
| 8 rows | 16.839 | 14.713 | 0.874 (0.873-0.874) | 16.886 | 14.716 | 0.871 (0.871-0.872) |
| 9 rows | 19.144 | 16.960 | 0.886 (0.885-0.887) | 18.975 | 16.953 | 0.893 (0.892-0.894) |
| 10 rows | 20.822 | 18.708 | 0.898 (0.898-0.902) | 19.574 | 18.721 | 0.956 (0.955-0.958) |
| 12 rows | 21.957 | 19.773 | 0.901 (0.900-0.903) | 20.651 | 19.761 | 0.957 (0.955-0.959) |
| 16 rows | 24.803 | 22.307 | 0.899 (0.897-0.902) | 23.975 | 22.291 | 0.930 (0.929-0.931) |

The routed experts' kernels a layer a rank in one nsys trace of the same block (6 rounds, the traced
prompt rows of memra#1324's trace; attribution, not timing), us:

| rows | base `gemv_f4s_tiles` + `swiglu_f4s_tiles` | main `v41_moe_small_kernel` x2 | B `v41_moe_small_blk_kernel` x2 |
|---|---|---|---|
| 4 | 77.3 | 92.0 | 62.2 |
| 9 | 142.9 | 167.3 | 112.3 |
| 10 | 148.1 | 171.7 | 117.2 |

On one card at the four-rank share's calls (random routings and pooled routings, 64 a cell), the block
ring's w1 + w3, SwiGLU and w2 take 0.66-0.77 of the base kernels' time at 3 to 64 rows and 0.81-0.92 at
1 and 2 rows; the warp ring took 0.91-1.27.

Byte identity: every probe variant against the warp ring on 64 routings a cell (0 differences);
`dsv41-gate kernels` 1089 of 1089 with every small-row check on both rings (40 on the block ring, two
cases at w2's shape with a short last stage); `dsv41-gate tp` on four ranks (needle capture, layers
1,2,14,20,21,24, two decode steps) 916 of 916 (default), 914 of 914 (peer-pull) and 914 of 914
(host-canonical), whose 180 layer hashes a transport equal main's binary's on the same command; on two
ranks (cards 0 and 1) 480 of 480, 478 of 478 and 478 of 478, the 108 layer hashes of each equal main's default.
One card: the layer gate (161 of 161), the decode gates (gsm, straddle126, capital) and `memra model verify`
checkpoint and serve (verified=Serve) pass on a full-binary build of the change.

Receipts: `research/modelplan-onboarding-dsv41-20261109/perf/receipts/moe1324/` (probe, gates, both
A/Bs, the trace).

## What was rejected

On the probe, each against the shipped shape at the same widths (random and pooled routings):

- **The pre-CUTLASS kernels.** They split K over warps and sum the parts: a different numeric program.
- **The warp ring with other launch shapes.** Three or four stages, 8 or 16 K blocks a stage, two or eight
  warps a block: the best of them (two stages, 8 columns x 16 K blocks, two warps) is level at one row
  and 1.04-1.38 from 2 rows. Its per-warp copy of the activations takes the shared memory that the extra
  blocks need.
- **Three stages on the block ring:** 1.01-1.28 at every width (fewer blocks an SM).
- **4 KB of weights a warp a stage** (16 columns x 16 K blocks): 0.95-0.98 at 8 to 10 rows on random
  routings but 1.10-1.17 on pooled ones, the traced prompt's kind. **1 KB** (8 x 8): 1.04-1.17 from 4 rows.
  8 columns x 32 K blocks is slower at every width.
- **Changing other cards.** The RTX 5090 and GB10 were not measured; their rule and kernel are unchanged.
