# Serving-card-class gate re-run, and the real-width identity check (partial)

Box: second bench box, 2x **RTX PRO 6000 Blackwell Server Edition** 96 GB, driver 595.91.07,
kernel `6.8.0-1061`, `NVIDIA_TF32_OVERRIDE=0`, release build of `5ac464acd1`.

## 1. LAW:card-keyed-full-pins — SATISFIED on the serving card class

The whole gate suite (4 host-only + 5 GPU) re-run on the card class glm5_next actually serves on.
Receipt: `gate-pro6000.txt`. **9 passed, 0 failed.**

The result is not merely green, it is **identical to the 5090 run to every printed digit**:

| quantity | rig 5090 | RTX PRO 6000 Blackwell Server |
|---|---|---|
| fused vs reference, worst row | 4.822e-3 on `decode step 2` | 4.822e-3 on `decode step 2` |
| unfused control, worst row | 4.822e-3 | 4.822e-3 |
| two arms, every row | 0.000e0, 0/N bits differ | 0.000e0, 0/N bits differ |
| fused dispatches | 51 of 89 | 51 of 89 |
| SLRU fused | hits=498 misses=645 staged=2,972,160 B | hits=498 misses=645 staged=2,972,160 B |
| SLRU unfused | hits=414 misses=387 staged=1,783,296 B | hits=414 misses=387 staged=1,783,296 B |
| post-for-pre-clamp | 7.563e-2 | 7.563e-2 |
| plain-swiglu | 8.615e-1 | 8.615e-1 |
| shared-expert-dropped | 1.297e-1 | 1.297e-1 |
| softmax-for-sigmoid | 5.301e-2 | 5.301e-2 |
| macro-plane-dropped | 6.244e-2 | 6.244e-2 |

Two readings, and the second is the load-bearing one:

- The fused arm is **card-class invariant** on this fixture — no cuBLASLt algorithm change, no
  warp-scheduling difference, nothing that moved a bit. The 2.1x tolerance headroom measured on
  the 5090 was the concern that made this run mandatory; it did not move at all.
- **The tolerance headroom was never the risk it looked like.** The concern was that the
  reference-parity floor might drift on a different card class and eat the 2.1x margin. It did
  not drift by one digit, which says the floor is the q8_1 activation quantization (a property of
  the arithmetic, not of the card) exactly as `TOL`'s comment claims.

## 2. Real-width bit-identity — ONE ARM BANKED, THE COMPARISON IS NOT DONE

Artifact: `~/models/glm53-nvfp4`, 178 GB on disk, the real 288-expert/n_used=8 geometry.
Env: `MEMRA_ST_PINNED=1 MEMRA_BF16_MMV=1 MEMRA_MOE_RESIDENT=0 MEMRA_MOE_SLOTS=1000`,
`reasoning_effort` pinned `low`, 96 tokens, prompts `p5`/`p7` from the banked real-prompt pool.
1000 slots is a ~41x margin over the 24-block per-token working set, chosen so the fused arm is
not fighting the fall-through this lane already measured at a thin margin.

`MEMRA_MOE_FUSED_EPI=0` (control) ran and is banked in `identity-IDOFF.txt`:

```
p5 greedy   out_sha 9753578f056888fe  completion_tokens 96
p5 sampled  out_sha 004d5d76932fd62e  completion_tokens 96   (vendor default t=1.0/top_p=0.95, seed 20260828)
p7 greedy   out_sha ede42d3f695f17c6  completion_tokens 96
[moe-fused-epi] snapshot dispatches=0                          <- the OFF arm took the arm zero times
[moe-cache]     snapshot hits=614632 misses=425624 hit_rate=59.085 slots=1000
```

**The `=1` arm never ran.** Its boot was in flight when the box rebooted at 14:28:49
(`BOX-INCIDENT-20260828.md`), and the lane then stood down because the PP lane declared the box
exclusive for an incidence study.

So real-width identity is **NOT yet claimed**, and the shas above are only half a comparison.
Recording them is still worth it: they are the control side, taken on the real artifact with a
known env, and they are what the `=1` arm has to reproduce byte for byte.

**Pre-committed protocol for the resumed run, so the kernel boundary cannot confound it.** The
control above ran on kernel `6.8.0-1061`; the box is now on `6.8.0-1063`. A kernel
boundary inside a byte-identity comparison is not acceptable evidence, so when the window opens
the order is: **re-run `=0` on the current kernel first, then `=1`, then compare** — never the
banked 1061 shas against a 1063 `=1` arm. If the re-run `=0` shas differ from the three above,
that is itself a finding (a kernel/driver-level numeric change) and gets reported rather than
absorbed.

Bar for the resumed check:
- greedy AND seeded-sampled `out_sha` byte-identical between `=0` and `=1` on the same kernel;
- `epi_per_tok` ~ 42.0 (one dispatch per MoE layer per token) in the `=1` arm, from the new
  `[moe-fused-epi]` counter — anything well under that means the arm fell closed and the
  comparison is measuring the sequential loop against itself.
- A sha mismatch at real width is a **stop-and-investigate**, not a tolerance question: the
  fixture proved bit-identity at width 128/64 only, and the fused chain's equivalence to the
  sequential one is a per-reduction-order claim.

## 3. Not yet done, and blocked on a box window

- the interleaved x5 A/B (A4BEST +- the flag, p5 greedy + p5 vendor-default sampled + p7 greedy),
  which is what the FLAGS row names as the flip condition;
- the staged-bytes delta at serving slot count (14000), which is the receipt for this lane's
  finding 3 (the fused arm admits all blocks up front and staged 1.19 MB more at a three-slot
  margin);
- engagement at the serving slot count — 51/89 was measured at fixture width with a margin of
  three, and the flip condition should be stated against the regime we would actually ship.
- **LAW:multiturn-cache-twin**: a serving-default flip additionally needs the 8-turn
  larger-prompt cache-on/off twin, not just steady cells. Not started, and named here so the
  flip decision is not surprised at review.

## 4. Staged and ready for the window (`ab-arm.sh`)

`ab-arm.sh <tag> <0|1>` runs ONE boot of the A/B: idle-check, then `cell-epi.sh` with the
A4BEST base env (`MEMRA_ST_PINNED=1 MEMRA_BF16_MMV=1 MEMRA_MOE_RESIDENT=0
MEMRA_MOE_SLOTS=14000`) plus the flag, then the banked instrument — p5 greedy, p5
vendor-default sampled, p7 greedy, 192 tokens x 4 reps each — and prints the engagement and
cache lines.

Three things it settles that the fixture could not:

- **the flip condition itself**, on the sampled row, because a default flip is justified by the
  product shape and never by the instrument;
- **engagement at the serving slot count**. 51/89 was measured at fixture width with a margin of
  three slots over a nine-block working set. At 14000 slots against 24 blocks the margin is
  ~583x, and `epi_per_tok` should sit at 42.0 (one dispatch per MoE layer per token). If it does
  not, the flip condition is not met no matter what the tok/s says;
- **finding 3 in the serving regime**. `MB_per_tok` and `miss_per_tok` medians on the `=0` and
  `=1` arms are the direct receipt for whether the admit-all-first order still costs extra
  staging when slots comfortably exceed `3*n_used`, or whether that cost was purely an artifact
  of the thin fixture margin.

Interleave: alternate boots `x5` per arm, never the two arms from one boot, idle-checked before
every boot. Still outstanding after that: the `LAW:multiturn-cache-twin` 8-turn larger-prompt
cache-on/off twin, which a serving-default flip needs and which this lane has not written.

## 5. Box vacated for the PP lane's residency cell (2026-08-28)

The coordinator called the box for the PP lane's keystone cell — real 190.7 GB artifact,
residency actually achieved across both cards, which is the measurement the whole 90 tok/s
roadmap hangs on and which needs both cards and an exclusive box. This lane yielded immediately.

Stop was PID-verified, never `pkill`: the wrapper was TERMed first so it could not launch another
probe, then the server only after `/proc/<pid>/cmdline` was read back and confirmed to be
`memra-epi/target/release/memra-server`. Server logged `drain complete in 0.0s` and
`GPU worker shutdown complete`. Confirmed after: **card 0 = 0 MiB, card 1 = 0 MiB, no compute
apps, no process of this lane left.** Artifact and scripts left in place as asked.

Two facts the aborted arm banked before it was stopped, both worth keeping:

- **Card pinning works.** `CUDA_VISIBLE_DEVICES=1` put the whole server on card 1 (13,907 MiB)
  with card 0 at 0 MiB, which validates the `cell-epi.sh` env-precedence fix (defaults before
  `"$@"` so a caller can override; previously the hardcoded `CUDA_VISIBLE_DEVICES=0,1` silently
  won and no caller could pin a card).
- **13,907 MiB is the BF16 trunk to the megabyte**, independently reproducing
  `ATTRIBUTION.txt`'s "13,907 MiB with MEMRA_BF16_MMV=1" on a different box and a different
  kernel. A small thing, but it is a free cross-box confirmation of a load-bearing constant.

No identity rows were produced: the arm was still in warm-up. Real-width bit-identity remains
UNCLAIMED, with the protocol in section 2 unchanged and pre-committed.

### A guard bug worth writing down, because it happened twice in one session

Both `idle-check.sh` and the identity arm's exclusivity guard were first written as
`ps ... | grep PATTERN`, and both **aborted on their own command line** — the pattern is
literally inside the pipeline that searches for it. The identity arm refused to start twice on
nothing but itself before the check was rewritten to walk `/proc/[0-9]*/cmdline` and skip its own
pid, its parent, and any cmdline containing `grep`/`ps`/its own name.

This is the house's "loud failures fail quietly" class inverted: a guard that fires when it
should not is not safe-by-default, it is a guard nobody will trust for long, and the second time
it cried wolf the temptation was to delete it. A process-existence check must inspect `/proc`,
never a pipeline that contains the string it is hunting.
