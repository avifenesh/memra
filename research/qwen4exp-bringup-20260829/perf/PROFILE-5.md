# qwen4_exp decode PROFILE-5 — the MTP spec-decode round (2026-08-30)

Lane: mtp-spec (spec/MTP-SPEC.md carries the semantics, gates, and per-battery
receipts; this file is the perf ledger in the PROFILE series). Box: sbox-eval
Frankfurt, 2× RTX PRO 6000 Blackwell 96 GB; artifact ~/data/q48fn-nvfp4 (NVFP4
mint + BF16 mtp graft). Baselines from PROFILE-4: plain single-card 14.5 ms/token
(69.16 tok/s), plain TP2 12.9 ms (77.27) — the 90 tok/s owner line was NOT crossable
by shaves and PROFILE-4 named MTP/spec as the multiplier. It is.

## Headline (mtp7 final battery, single-card route, real prompt, greedy instrument)

| | ms/token | tok/s | vs plain single (same-run arm) | vs plain TP2 12.9 (PROFILE-4) |
|---|---|---|---|---|
| plain single-card (same-run interleaved arm, 256-token window) | 14.86 | 67.28 | 1.0× | 0.87× |
| **spec K=5, interleaved ×5, 256 tokens/arm** | **8.37** | **119.50** | **1.78×** | **1.55×** |

Ladder single-run best: K=5 8.22 ms/token (121.6 tok/s). Decode-timing same run:
14.3 ms (plain baseline intact with the draft resident). Sampled probe (vendor
defaults): ENGAGED 54/58 rounds, hist 4,6,12,2,7,27 (27 full-K rounds), accepted
199/290. Receipts: spec/mtp7/.

Byte-identity held on EVERY receipt: verify-bit-gate 24/24 rows bit-identical
(plain-decode rows vs verify-chunk rows on the same fed tokens), spec-gate token-for-
token equality on all 4 real prompts, rep-0 A/B chains identical. The greedy chains
here are the INSTRUMENT; the sampled probe (vendor defaults 1.0/0.95/20) ships its own
engagement receipt per the serving law.

## The round's ledger (interleaved A/B or ladder, real prompt, per battery)

| step | change | measured | receipt |
|---|---|---|---|
| mtp2 | first correct spec loop (per-token verify rows, bit-identical) | spec 11.1-12.9 ms/token on real prompts (vs plain 14.4) but 18.5 on the AB; ladder/AB were run on the goldens probe whose greedy chain DEGENERATES (loop law) — perf rows invalidated, lesson banked | spec/mtp2 |
| mtp3 | `set_verify_mt` — weight-shared verify: `qmatvec_bf16w_mt_f32` on trunk linears + MoE verify-column MERGE (gufuse tok_map) | vmt OFF 11.03 → ON **8.44 ms/token** (90.7 → 118.5 tok/s, 1.31×), interleaved ×5, k=4, real prompt | spec/mtp3/ab-spec-k4-* |
| mtp4 | hc-diet MT stages (stage0/1/3_mt — read-gate weights read once) | AB 8.24 (121.3); ladder best K=5 8.02 (124.7) | spec/mtp4 |
| mtp5 | draft cuts: lm_head reads ONE row unless gates ask; draft prefill MoE via per-expert executor | AB 8.16 (122.5); ladder K=5 7.93 (**126.1**) | spec/mtp5 |
| mtp6 | sel warp packing (4 warps/block) | **NEGATIVE**: plain arm 14.38 → 15.13, verify sel flat → REVERTED (receipts kept) | spec/mtp6 |
| mtp7 | final battery at the shipped defaults | plain 14.86 vs spec K=5 **8.37 ms/token (119.5 tok/s)**, interleaved ×5 over 256 tokens/arm, rep-0 chains byte-identical | spec/mtp7 |

## K ladder (256 tokens, real prompt, mtp6/mtp7 class)

K=5 is the knee: accept ~0.84-0.86, mean accept len ~5.1-5.3, 121-126 tok/s.
K>6 loses to acceptance decay (K=8: 0.66-0.70, 104-113); K<4 wastes the verify
amortization. Full tables: spec/mtp*/spec-ladder-*.tsv.

## Where the milliseconds sit (spec-profile k=5, sync-bounded, SHARES are the signal)

Per round (≈5.2 committed tokens): verify ≈ 0.80 share, draft ≈ 0.19.
Verify rocks: moe.sel_grouped (the merged grouped launch — ~60 slots' expert bytes are
irreducible reads; NOT block-slot-limited, packing measured flat), hyper.read (mt
stages), gdn.proj (mt matvec), gdn.conv_scan (per-column step launches + snapshots).
Draft rocks: the K−1 chain steps' FULL-VOCAB lm_head (~0.85 ms each — the FR-Spec trim
lever), the eager per-step launch overhead.

## Verdict vs the 200 tok/s owner target: NOT crossed — ~126 vs 200 (1.6× gap)

What the round DID cross: the 90 tok/s line that PROFILE-4 declared unreachable for
plain decode — spec K=5 lands 119.5 tok/s interleaved (121.6 ladder; 126 on the
100-token mtp5 window) single-card, 1.78× plain single and 1.55× the TP2 plain route,
with byte-identity receipts at every step.

The honest residual, in measured order:

1. **FR-Spec draft-head trim** (DRAFT-REGIME law): the draft's chain-step head is
   ~3.4 ms/round at K=5 (~0.65 ms/token); a ranks-derived 32k trim is the regime's
   established ~8× cut on that section (~+10-12 tok/s) AND unlocks higher K (the K=8
   arm's mean accept len 6.2-6.7 becomes profitable when drafts are cheap): projected
   ~150-165. Needs the own-gen ranks pipeline on THIS artifact (chat template on,
   class coverage) — a lane of its own, not a quick fix (law 1 is not relaxable).
2. **Draft + verify segment graphs**: both sides run eager (~60 launches/draft-step,
   ~500+/chunk); the trunk's segment-graph pattern applies to both. Est. 1-3 ms/round.
3. **Verify sel restructure** (split-K / two-stage reduce): the merged grouped launch
   sits ~2× over its bytes floor and is the largest verify slice; the PROFILE-4
   residual already named this class. Warp packing alone is NOT it (mtp6 negative).
4. **TP2-route verify**: decode_step_tp2 is t==1-wired (per-card halves, P2P joins,
   segment graphs); a t-generic TP2 verify is a parallel build of the half-kernel set.
   Upper bound from the plain ratio (14.5/12.9 = 1.12×) applied to the verify share:
   ≤ ~9%. Only worth it after 1-3.

Stacked projection if 1-3 land: round ≈ 26-30 ms at K=6-8 with mean len 5.5-6.5 →
~180-230 tok/s. 200 is inside the projection band but NOT demonstrated; the claim
stays at the measured 119.5-126 until the next lane banks its own receipts.

## Rule gates at final HEAD (mtp8)

tp2-gate re-run at the lane's final commit: **24/24 argmax, worst rel 3.018e-5, PASS**
— identical class to the round-4 banked baseline (the lane's kernel touches are
bit-neutral on the TP2 route). Fresh TP2 plain decode-timing same run: 12.6 ms/token
(79.5 tok/s) — single-card spec K=5 beats the TP2 plain route 1.51× on the same day's
clock. Receipts: spec/mtp8/.

## VRAM (residency receipt)

Trunk NVFP4 + DeviceBf16 draft bank co-resident on card 0: 95,283 / 97,887 MiB
post-load; verify stash (K=5: ~650 MB; K=8: ~1 GiB) + logits/workspace fit the
headroom at bench contexts. Long-context serving would move the draft bank to card 1
(free) — noted, unmeasured.
