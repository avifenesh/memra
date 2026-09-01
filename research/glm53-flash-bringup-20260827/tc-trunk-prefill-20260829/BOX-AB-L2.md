# L2 box A/B: the three-arm run (2026-08-29/30, owner window)

Owner-granted window on the rented 4x RTX PRO 6000 Blackwell Server 96 GB box (600 W all
cards). Protocol exactly as `AB-PLAN.md` (pre-registered); raw evidence in `box-ab/`
(`l2-run.log`, `rows-*.public.json`, `summary.txt`, `prof-moe-split.txt`,
`2card-and-prof-provenance.txt`; `text_head` fragments stripped before banking, output shas
retained; prompt text stays on-box).

## Provenance

- Engine `bac42f759d40343e74e9bfd17f4581b875f2bc28` = origin/lane/glm5-tc-trunk-prefill
  (cb6600490) merged with origin/lane/glm53-flash-bringup (which now carries
  `dd7f1d11d MEMRA_MOE_GROUPED_PREFILL: DEFAULT ON, by owner acceptance 2026-08-29`),
  plus cherry-pick `6f720883c` (build.rs rerun-if-changed on .git/HEAD).
- Binary `aab0a0a85999a0bb...cc509aa10`, rebuilt in-window, BINARY-NEWER-THAN-SOURCES: PASS
  (find -newer over crates returned nothing). `git log -1` in the run header.
- Artifact `/root/models/glm53-nvfp4` (byte-verified per box state doc). Prompts
  `prompts.json` sha `de57a7a471f9b163...74b53e46`, the BOX-AB set (measured prompt_tokens
  4626/5547/6467 + 427 warmup).
- Placement: PP3 cards 0/1/2, full expert residency, `MEMRA_PREFIX_CACHE_MB=0`,
  `MEMRA_CTX=8192`, `MEMRA_MAX_SESSIONS=4`, TF32 off, `reasoning_effort` low,
  `MEMRA_MOE_GROUPED_PREFILL=1` pinned in every arm (also now the family default).
- Interleaved x5, fresh boot per arm, order a,b,c repeated; smoke boot first (not counted).

## Arms

- **A** baseline (L1 on): no MMV, no PP_BF16.
- **B** = A + `MEMRA_BF16_MMV=1` (bf16-resident trunk).
- **C** = B + `MEMRA_PP_BF16=1` (the tensor-core door).

## The table (median [min..max], x5)

| row | arm | TTFD s | prefill tok/s | decode tok/s |
|---|---|---|---|---|
| A4630 greedy | A | 7.51 [7.42..7.74] | 616.2 [597.5..623.3] | 21.34 [19.41..21.44] |
| A4630 greedy | B | 7.46 [7.43..7.80] | 620.5 [593.0..622.7] | 26.06 [25.73..26.11] |
| A4630 greedy | **C** | **6.81 [6.79..7.11]** | **678.9 [650.9..681.6]** | 26.14 [25.05..26.22] |
| B5550 greedy | A | 8.86 [8.85..8.89] | 626.0 [623.9..627.1] | 22.00 [20.73..22.13] |
| B5550 greedy | B | 8.89 [8.86..9.00] | 623.7 [616.4..625.7] | 26.05 [25.97..26.12] |
| B5550 greedy | **C** | **8.17 [8.13..8.35]** | **679.0 [664.3..682.6]** | 26.15 [26.00..26.17] |
| C6470 greedy | A | 10.43 [10.24..10.48] | 619.8 [617.1..631.3] | 21.95 [20.70..22.03] |
| C6470 greedy | B | 10.26 [10.24..10.62] | 630.3 [609.2..631.6] | 25.98 [25.92..26.08] |
| C6470 greedy | **C** | **9.41 [9.38..9.42]** | **687.3 [686.3..689.5]** | 26.06 [25.56..26.12] |
| A4630 sampled (vendor default) | A | 7.23 [7.21..7.24] | 640.3 | 21.50 |
| A4630 sampled (vendor default) | B | 7.25 [7.24..7.26] | 638.4 | 25.34 |
| A4630 sampled (vendor default) | **C** | **6.62 [6.61..6.63]** | **698.8 [697.4..699.5]** | 25.50 |

**The door (C vs B): TTFD -8.7% / -8.1% / -8.3%, non-overlapping x5 spreads at every length,
sampled twin the fastest row (698.8 tok/s median). Residency (B vs A): prefill neutral,
decode +18-22% (21.3-22.0 to 26.0-26.1 tok/s, non-overlapping), VRAM -10.0 GiB.**

The per-token reading of the door: 0.13-0.14 ms/token removed, consistent across lengths.
That is SMALLER than PREFILL-GAP section 1.2's 0.3-0.5 ms/token arithmetic for the f32 trunk
term; the corrected attribution (this receipt supersedes the estimate) is that the f32 trunk
GEMM term on this card class was roughly a third of the prior arithmetic, and the KDA
sequential scan (L3) owns most of the post-L2 residual.

## Argmax gate: 30 of 30 MATCH, no census needed

Greedy first token, 5 boots x 3 prompts per arm: `The` everywhere, in all three arms.
B-vs-A MATCH 3/3 prompts, C-vs-B MATCH 3/3 prompts. Every arm is boot-deterministic (ONE
32-token output sha per prompt per arm across all 5 boots). The 8-draw census tool was
staged (`census.py` on-box) and never needed: there was no flip to census. Bonus receipt:
arm A and arm B produce BYTE-IDENTICAL full 32-token greedy text on C6470
(`b68df3f95bfc103c` both).

## Decode stop condition (C vs B): PASS

26.06/26.05/25.98 (B) vs 26.14/26.15/26.06 (C): overlapping spreads, no decode movement
from the door, exactly as the m >= 16 predicate requires.

## Engagement receipts

- `[bf16-mmv] RESIDENT`: 148 per boot in B and C, 0 in A. Census (from the prof boot's
  log): 34 x `kda_q/k/v/out` (33.5M each) + 12 x `indexer.attn_q_b` (6.3M) + embed/head
  members.
- `[bf16-tc] flag=off` printed in every B boot, `flag=on` in every C boot (the announce
  added by this lane, both arms).
- `[bf16-tc] ENGAGED`: 12 distinct shapes per C boot, three projection classes
  (n=8192 k=4096 the KDA q/k/v; n=4096 k=8192 the KDA out; n=4096 k=1536 the indexer q_b)
  x four prompt lengths (427/4626/5547/6467; chunks=1 per prompt on this walk, as BOX-AB
  found). `DECLINED`: **0**.
- `[moe-grouped-prefill] execute`: 42/42 MoE layers in every boot of every arm.
- Boot identity: fresh PID + `readlink /proc/pid/exe` + binary sha per boot in the run log.

## VRAM per arm (after load, MiB, cards 0/1/2; card 3 untouched)

| arm | dev0 | dev1 | dev2 | total delta vs A |
|---|---|---|---|---|
| A | 54,547 | 65,651 | 68,085 | - |
| B, C | 51,443 | 62,771 | 64,021 | **-10,048 MiB (~9.8 GiB freed)** |

B also LOADS faster (~64-66 s vs ~86-104 s): the load-time host dequant of the trunk is gone.

## THE 2-CARD BOOT: SUCCESS (owner-priority check)

Under arm C (`serve2.sh`: `MEMRA_PP_STAGES=2 MEMRA_PP_SPLITS=24 MEMRA_PP_DEVICES=0,1`,
cards 2/3 never touched), the boot the merged head refuses at f32 (misses by 650 MiB at
split 24, BOX-AB) now LOADS AND SERVES:

- Load: dev0 88,659 MiB / dev1 88,949 MiB of 97,887 (about 9.0 / 8.7 GiB free where f32
  had 650 MiB); both stages `-> RESIDENT` (97.84 GB and 89.69 GB experts vs ~100 GB free).
- Session admission proven, not assumed: under the full probe the cards reached
  95,797 / 95,479 MiB with zero OOM, all five rows 200.
- The greedy cell: TTFD 7.14 / 8.01 / 9.29 s, prefill 648 / 692 / 696 tok/s, decode
  26.1-26.3 tok/s, and the 32-token greedy output shas are **byte-identical to the 3-card
  arm C** on all three prompts (`0908061a3fdaddab` / `3ad5d2889683a115` /
  `dfab5cc3dfbfda25`): cross-placement PP3-vs-PP2 greedy identity.
- Sampled vendor-default twin healthy: 6.54 s TTFD, 707 tok/s prefill.

This is the serving-economics receipt: the model serves from TWO 96 GB cards at the same
speed as three, with the bf16-resident trunk as the enabler. Rows in
`box-ab/rows-2card-c.public.json`.

## Profile pass (arm C): the honest partial

- The nsys half is BLOCKED on this box: the newest packaged Nsight Systems for this image
  (2024.2.3, cuda-12-5 repo) records NO CUDA kernel events against the CUDA 13.1 driver
  ("does not contain CUDA kernel data"); `profile-prime-phases.sh`'s phase buckets need a
  newer nsys CLI. Named follow-up for the L3 window (which needs the same attribution).
- Banked fallback (`prof-moe-split.txt`, `MEMRA_PRIME_PROF=1`, arm C): grouped MoE per
  layer at t=4626 is router 0.0 + gemm_gu ~13.9 + down_scatter ~7.6 + shared ~1.4 =
  ~22.9 ms, x42 = **~0.96 s of the 6.8 s TTFD**. The remaining ~5.9 s is the KDA scan +
  attention/kpool + mHC + glue + the small f32 trunk residue: the L3 term dominates the
  residual, confirming the plan's L3-next sequencing.

## Verdict vs the pre-registered flip condition

| condition | result |
|---|---|
| TTFD improves C-vs-B at every length, x5 non-overlapping | **PASS** (-8.1% to -8.7%) |
| sampled vendor-default twin healthy | **PASS** (fastest row) |
| engagement receipts both flags, both ways | **PASS** (148/0 RESIDENT; 12 ENGAGED / 0 DECLINED; flag announces in all arms) |
| first-token argmax gate, C-vs-B and B-vs-A, real prompts | **PASS 30/30, no flips, census not needed** |
| decode C-vs-B within noise | **PASS** |

**No default flipped in this window.** The bundle goes to the owner for the accept/hold
call on: (1) pinning `MEMRA_BF16_MMV=1 MEMRA_PP_BF16=1` in the glm5 serving env, (2) any
family-default flip (its own PR, FLAGS rows updated with these measured rows), (3) whether
the 2-card placement becomes a serving option (economics: same speed, two cards).

## Box state at window close (L2 share)

All memra-server processes stopped (PID-verified via the pidfile guard, pgrep clean), VRAM
0 MiB on all four cards. `~/memra` at `bac42f759d` (branch `ab-l2-20260829`), binary
`aab0a0a859...` in target/release. `~/l2-ab/` kept on-box (rows, serve logs, prompts copy,
census.py unused, serve2.sh for the 2-card shape); `~/gpf-ab/` untouched; the pinned
drafter clone and `/root/models/*` untouched; `/tmp/prime-prompt.json` deleted. Box handed
to the L3 lane next (same window tail).
