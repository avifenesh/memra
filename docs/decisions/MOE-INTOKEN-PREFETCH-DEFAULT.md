# The MoE in-token expert prefetch is a per-card default (2026-11-04)

**Chosen:** on the RTX 5090 class and the full-power RTX PRO 6000 Blackwell Workstation Edition,
the legacy MoE slot cache stages the next routed expert's cache misses on the copy stream while
the current expert computes (`MoeSlotCache::prefetch_source`). The card class is read from the
device name once at engine construction (`moe_cache::in_token_prefetch_for_device`, keyed as the
VMM grow placement is). The in-token arm of `MEMRA_MOE_PREFETCH=1` is deleted.

**Rejected:** staging each expert only when its kernels need it, on those two card classes. It
stays the program on the RTX PRO 6000 Max-Q and Server Edition variants and on every other card,
which have no cell of their own.

**Measurement that settled it:** `research/spill-c-20260919/DAY59.md`, pre-registered before any
cell. One build per card for both arms, `Qwen3.6-35B-A3B-UD-IQ4_XS`. Correctness gates per card:
G1 run-gen tapes at a pressure shape (`MEMRA_MOE_RESIDENT=0`, 9986 slots) and a heavy one (512
slots), one tape across both arms; G2 run-spec K=1..8 self-consistency; G3 seven greedy served
requests byte-equal across an OFF and a PF boot. Timing: gen-only decode, 20 runs per shape in
both orders, N=10 per arm, 250 ms telemetry.

| Card | Gates | Pressure shape OFF -> PF | Naked shape OFF -> PF |
| --- | --- | ---: | ---: |
| RTX PRO 6000 Blackwell Workstation Edition, 600 W (`pro-single-day61/`) | G1, G2, G3 PASS | 0.310 -> 0.255 s, `pf_wins` | 0.113 -> 0.113 s, `pf_flat` |
| RTX 5090 Laptop (`rtx5090-day59/`) | G1, G2, G3 PASS | 0.412 -> 0.364 s, `pf_wins` | 0.216 -> 0.214 s, `pf_flat` |

The pre-registered rule makes PF a card's naked default when its gates pass, the pressure shape
reads `pf_wins` and the naked shape does not read `pf_loses`. Both cards meet it. The naked shape
is flat because every expert stays resident there, so there is no miss to prefetch. The owner's
2026-11-04 ruling makes positive-receipt winners defaults keyed to the receipt's card, with the
switch removed. The finding that reached the owner said the 5090 cells were still owed; the
tracked receipt shows they ran and passed on 2026-09-26, so the 5090 class is keyed on the same
footing as the PRO 6000.

**Scope:** the 5090 receipt is the laptop card and is applied to the RTX 5090 class, as the VMM
grow placement applies its laptop receipt. The PRO 6000 receipt covers the full-power Workstation
card only. The CPU-expert companion's prediction worker, which reads the same variable as a depth,
was not part of the cell and is not decided here.
