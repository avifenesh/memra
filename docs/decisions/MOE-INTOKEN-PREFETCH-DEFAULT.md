# The MoE in-token expert prefetch is a per-card default (2026-11-04)

**Chosen:** on the RTX 5090 Laptop GPU and the full-power RTX PRO 6000 Blackwell Workstation
Edition, the legacy MoE slot cache stages the next routed expert's cache misses on the copy stream
while the current expert computes (`MoeSlotCache::prefetch_source`). The card is read from the
device name once at engine construction (`moe_cache::in_token_prefetch_for_device`), and the slot
cache's boot line names the arm and its source (`[moe-cache] in-token prefetch: on (card
rtx5090-laptop-receipt)`). The in-token arm of `MEMRA_MOE_PREFETCH=1` is deleted.

**Rejected:** staging each expert only when its kernels need it, on those two cards. It stays the
program on the desktop RTX 5090, the RTX PRO 6000 Max-Q and Server Edition variants and every other
card, which have no cell of their own.

**Measurement that settled it:** `research/spill-c-20260919/DAY59.md`, pre-registered before any
cell. One build per card for both arms, `Qwen3.6-35B-A3B-UD-IQ4_XS`. Correctness gates per card:
G1 run-gen tapes at a pressure shape (`MEMRA_MOE_RESIDENT=0`, 9986 slots) and a heavy one (512
slots), one tape across both arms; G2 run-spec K=1..8 self-consistency; G3 seven greedy served
requests byte-equal across an OFF and a PF boot. Timing: gen-only decode, 20 runs per shape in
both orders, N=10 per arm, 250 ms telemetry.

| Card | Gates | Pressure shape OFF -> PF | Naked shape OFF -> PF |
| --- | --- | ---: | ---: |
| RTX PRO 6000 Blackwell Workstation Edition, 600 W (`pro-single-day61/`) | G1, G2, G3 PASS | 0.310 -> 0.255 s, `pf_wins` | 0.113 -> 0.113 s, `pf_flat` |
| RTX 5090 Laptop GPU, 175 W limit (`rtx5090-day59/`) | G1, G2, G3 PASS | 0.412 -> 0.364 s, `pf_wins` | 0.216 -> 0.214 s, `pf_flat` |

The pre-registered rule makes PF a card's naked default when its gates pass, the pressure shape
reads `pf_wins` and the naked shape does not read `pf_loses`. Both cards meet it. The naked shape
is flat because every expert stays resident there, so there is no miss to prefetch. The owner's
2026-11-04 ruling makes positive-receipt winners defaults keyed to the receipt's card, with the
switch removed.

**Scope:** each arm is the card its cell ran on. The 5090 cell ran on the laptop card (its
telemetry reads a 175 W power limit and a 1627 MHz SM median), so the desktop RTX 5090 (170 SMs,
32 GB, a different bus-to-compute ratio) keeps the prefetch off until its own cell. The PRO 6000
receipt covers the full-power Workstation card only. The CPU-expert companion's prediction worker,
which reads the same variable as a depth, was not part of the cell and is not decided here.

**Open for the owner:** his 2026-11-04 answer named this door "on the PRO 6000". The finding that
reached him said the 5090 cells were still owed; the tracked receipt shows they ran and passed on
2026-09-26, so the laptop arm rests on the general rule (a positive receipt keys its own card).
Removing it is a one-line change in `in_token_prefetch_for_device` if he wants the PRO 6000 only.

**Owed:** the cell covers one artifact, and the default reaches every MoE family on the legacy
slot-cache spill path on those two cards. The same G1 tape and G3 served byte-equality cells on a
second MoE family close that gap, for example a mixed-layout Hy3 spill, whose experts take the
metadata-aware slot-cache path with the same prefetch hook.
