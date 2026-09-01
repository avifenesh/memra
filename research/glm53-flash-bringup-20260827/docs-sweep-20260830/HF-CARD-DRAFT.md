# HF card correction draft: Avifenesh/GLM-5.3-Flash-NVFP4 (docs-sweep, 2026-08-30)

**DRAFT FOR THE OWNER. Nothing here is published; the owner pushes HF changes.**

Scope: the published card's "Context, and where the practical wall actually is" section
(mirrored locally at `research/glm53-flash-bringup-20260827/hf-card/README.md:227-257`)
is stale in two load-bearing ways:

1. It pins the 1M wall to a monolithic-prefill transient ("Under a monolithic prefill
   where `t = t_kv = N`, that plane is **N squared bytes**") and reads as if that wall
   is where the artifact stands today. The monolithic prime WAS the wall; the shipped
   chunk schedule removed it (`hyper_prime_ranges` delegates the mHC prime to
   `prime_chunk_ranges`, single-engine since 08-28 and the ppN twin fixed in the 1M-demo
   lane at 93927b1fac, bit-identity gated by glm5-hyper-ppn-gate stages=4 cross-device).
2. "We have not gated or run this artifact anywhere near that context" and "If you need
   long context on this model today, that is unsolved work, not a flag" is falsified by
   the 1M demo receipts: a real **1,035,357-token** prompt primed through the serving
   surface and decoded to EOS (`research/glm53-flash-bringup-20260827/1m-demo-20260829/LANE.md`,
   branch lane/glm53-1m-demo, head 315a1080fb).

The mint-innocence point the section makes (the wall was never a quantization artifact)
stays true and stays in the draft; what changes is where the wall is and what has been
demonstrated.

---

## Replacement section (drop-in for "## Context, and where the practical wall actually is")

## Context: 1,035,357 tokens demonstrated, and where the remaining wall actually is

The upstream checkpoint declares `max_position_embeddings: 1048576`. We have run this
artifact at that scale, once, deliberately, and the receipts are dated bench measurements
on named hardware: not a hosted offering, and not a latency claim.

**Demonstrated (2026-08-29, 4x RTX PRO 6000 Blackwell 96 GB, memra PP4 pipeline,
`MEMRA_PP_SPLITS=13,26,39`):** a real 1,035,357-token prompt (Gutenberg prose, sha-banked,
token count from the server's own usage, `cached_tokens=0`) primed through the serving
surface inside the model's 1,048,576 window, then decoded greedy to EOS with a coherent
cross-book answer. Prefill 161.28 tok/s (6,419.8 s wall, 107 minutes); the vendor-default
sampled twin primed the same prompt at 161.26 tok/s (two independent full primes agreeing
to 0.01%) and also answered coherently. Error census of the serve log: 0.

Depth behaves, with no cliff:

| prompt tokens | prefill tok/s | decode tok/s (greedy, steady) |
|---|---|---|
| 15,766 | 172.7 | 24.5 |
| 128,566 | 171.5 | 22.8 |
| 257,775 | 169.6 | 21.1 |
| 525,616 | 165.8 | 18.9 |
| 1,035,357 | 161.28 | 15.7 |

Chunked prefill throughput is depth-flat to 1M (-6.6% from 16k), and greedy decode decays
smoothly (1.75x over three orders of magnitude of depth); the sampled twin sits within
0.6% of greedy at 1M. In-context retrieval held at depth (at 131k the answer cited a
detail from a salon debate inside War and Peace).

**What the earlier revision of this card called the wall, the DSA score-plane transient
of a monolithic prefill (`N^2` bytes per MLA layer per call), is fixed, not by more
VRAM but by the chunk schedule it predicted:** memra's mHC prime now walks bounded chunks
(monolithic rollback preserved behind `MEMRA_PRIME_CHUNK=0`), so the per-call plane is
`chunk_rows x (N / 4)` f32 rather than `N^2` bytes. That was never a quantization
artifact and its fix is not one either: an FP8 or BF16 copy of this model has the same
arithmetic on both sides of the fix.

Honest boundaries, so this is not read as a serving claim:

- **1M is demonstrated on 4 cards, not fewer.** A 3-card full-expert-residency
  configuration of this artifact fails the same prime (CUDA out-of-memory in the DSA
  k-pool selection at a 97.2 GiB per-card peak), so the depth ceiling of smaller
  placements sits well below 1M, and the only demonstrated 1M configuration is the
  4-card pipeline with the expert cache capped to leave the tail stage room.
- **Time-to-first-token at 1M is 107 minutes on that baseline.** Long-context capacity
  is solved; long-context *latency* is engineering in progress (prefill has since moved
  independently on other configurations of the same engine, e.g. a tensor-core MLA
  prefill path measured at 1629-2255 tok/s at short depths, but no 1M number exists on
  that path and none is claimed here).
- Everything above is greedy-plus-sampled-twin bench measurement on named hardware,
  published with its receipts in the memra repo
  (`research/glm53-flash-bringup-20260827/1m-demo-20260829/`).

---

## DFlash2 attribution line (CONDITIONAL: include ONLY if a served configuration draws drafts from DFlash2)

> The inco approval (owner holds WRITTEN APPROVAL from the DFlash2 owners, 2026-08-30,
> for use beyond probe/eval) requires a visible DFlash 2 mention IF it serves. As of this
> draft, `MEMRA_GLM5_DFLASH` is DEFAULT OFF (the 3way decision: DFlash2 is the drafter of
> record but the untuned loop's best arm is 0.988x plain, so spec does not serve). Do not
> include this block until that changes.

Placeholder text for that day:

    Speculative decoding for this model uses **DFlash 2**
    ([incoai/GLM-5.3-Flash-DFlash2](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2),
    revision `dc77ff1c`), a block-diffusion drafter by the DFlash team at Inco AI, used
    with the authors' permission. Measured on our serving shape it banks 1.907 accepted
    tokens/cycle at K=3 greedy (1.889 vendor-default sampled) against the target's own
    outputs, 66/66 served byte-identity vs plain decoding.

---

## Receipts index for whoever edits the live card

| claim | receipt |
|---|---|
| 1,035,357 tokens primed + decoded to EOS, greedy 161.28 tok/s prefill / sampled twin 161.26 (0.01% agreement), error census 0 | `1m-demo-20260829/LANE.md` (branch lane/glm53-1m-demo @ 315a1080fb), phase7-R1M.json / phase7-R1M-V.json |
| depth ladder 16k/131k/262k/524k/1M rows | same LANE.md result table (phase3/phase4/phase7 receipts) |
| monolithic ppN prime OOM from ~32k + gridDim.y 65,535 ceiling; fixed by walking the shipped chunk schedule; bit-identity stages=4 cross-device | same LANE.md "walls" §2, 04-phase2.txt, 07-phase3.txt |
| per-card VRAM peaks at 1M: 81,945 / 80,121 / 80,089 / 94,905 MiB | same LANE.md, phase7-vram.csv |
| 3-card resident shape is NOT a 1M config (kpool OOM, 97,242 MiB peak) | `research/glm5-prefix-latent-20260830/box-window/WINDOW-STATUS.md` "1M serving-config receipt, box B" |
| MLA-TC prefill 1629-2255 tok/s (short-depth, 3-card shape, default ON) | `research/glm5-prefix-latent-20260830/box-window/` mla-tc-ab arms; FLAGS.md `MEMRA_MLA_TC_PREFILL` row |
| DFlash2 drafter-of-record + 0.988x best-arm no-flip | `research/glm53-flash-bringup-20260827/3way-decision-20260830/LANE.md` |
