# moe-coactivation-20260902: can GLM-5.3-Flash's routed experts be split across two cards by co-activation?

Owner question (2026-09-02): 288 routed experts per MoE layer, 8 selected per token. If experts that
fire together sit on one card and the always-active ones are replicated on both, how often would a
token's 8 experts land on a single card? This lane ships the instrument, not the answer: the answer
is a box run on real routing data. Branch `lane/moe-coactivation-20260902` (base
`lane/glm5-b200-int2-20260902`), no GPU here.

Prior art this builds next to, not over: `MEMRA_MOE_TRACE`/`MEMRA_MOE_WEIGHT_TRACE` (text, per
forward call, DIVERT dispatch to the host-routed path) feed `tools/build_expert_placement_map.py`
(the `memra-ep-map-v1` mint, `research/ep-placement-map-20260831/REPORT.md`). Neither carries a
replication budget, neither scores P(single card) per token, and neither can see the device-table
decode arm. The dump here records the served arms and changes no dispatch decision.

## Deliverables

1. `MEMRA_MOE_SEL_DUMP=<path>` (default OFF, `docs/FLAGS.md` §4 row; module
   `crates/memra-engine/src/moe_sel_dump.rs`). One binary record per (routed token, MoE layer) for
   prime (grouped prefill, the CSR's own host selection) and decode (the pinned `[sel, w]` readback
   and the `vrows_dev` device-table arm, read back with one DtoH per layer-call).
   Format `memra-moe-sel-v1`, LE, no header: `u8 layer, u8 n_sel, n_sel x (u16 expert, f32 weight)`,
   50 bytes per glm5_next record. Zero cost unset (one `OnceLock` read per entry point). The
   device-routed TP walks refuse an armed dump by name.
2. `tools/moe_coact.py` (numpy only): per layer and pooled, activation frequency (min/median/max,
   Gini, top-16 share), the 288x288 co-activation matrix (`--save-coact`), and for R in
   {0, 16, 32, 64} replicated experts the best 2-way partition (spectral start + Kernighan-Lin,
   replicated set by frequency or by cut contribution, best kept) scored on HELD-OUT tokens:
   P(all n_sel experts on one card), mean cards touched per token, per-card share of token-expert
   pairs, against the random-halves baseline (mean over seeds) and the engine's contiguous even
   split. Markdown to stdout (`--out-md`), numbers to `--out-json`.

## Box invocation (the spawning session runs this on the 2x B200 pair)

```sh
# 1. Same launch and posture pins as the lane's serving boots (MEMRA_MOE_RESIDENT_GB=130 per
#    device, MEMRA_ST_PINNED unset, spec gate pins), plus ONLY this env. Do NOT set
#    MEMRA_MOE_TRACE / MEMRA_MOE_WEIGHT_TRACE / MEMRA_MOE_STATS in the same boot: they divert
#    dispatch to the host-routed path and the dump would then measure a different program.
MEMRA_MOE_SEL_DUMP=/root/out-coact/sel-$(date -u +%H%M%S).bin  <the lane's server launch>
# boot receipt: stderr line "[moe-sel-dump] armed path=..." (absent = the door did not arm)

# 2. Vendor-default sampled requests (no sampling params), 256-token answers, in this order:
#    /root/prompts/code.txt, /root/prompts/prose.txt, /root/prompts/digits.txt (decode-shaped),
#    then the Gutenberg corpus slices of 28,000 chars (~6.9k tok) and 170,000 chars (~42k tok)
#    (prime-shaped: one grouped-prefill record per token per MoE layer; expect ~50 B x tokens x
#    MoE layers, on the order of 100 MB for the 170k slice). Wait >1 s after the last response
#    before stopping the server (the sink flushes at most once per second), then SIGTERM it.
#    Optional second boot with max_tokens=1 on the two slices to get a prime-only file.

# 3. Analyze (seconds per file), one report per dump; the pooled table is the verdict input:
python3 tools/moe_coact.py /root/out-coact/sel-*.bin --experts 288 \
    --out-md research/moe-coactivation-20260902/report-<prompts>.md \
    --out-json research/moe-coactivation-20260902/report-<prompts>.json \
    --save-coact research/moe-coactivation-20260902/coact-<prompts>.npz
```

Commit the dump files' sha256 and sizes next to the reports (the dumps themselves go to R2, not
git). A report whose per-layer table shows fewer layers than the model's MoE count means an arm
declined: read the boot stderr before interpreting anything.

## Interpretation thresholds (proposed; the held-out pooled row at each R)

The measured control is the naive even split: `naive EP peer-touch ~99.3%` (tp2-battery, quoted in
`LAW:coactivation-expert-placement`), i.e. P(single card) ~0.7%, which is exactly the independent-
routing value 2 x 0.5^8 = 0.78%. Any partition is judged against that and against the random-halves
column the tool prints on the same tokens.

| pooled P(single card), held-out | verdict |
|---|---|
| >= 50% at R <= 32, cards/token <= 1.5, per-card load within 40/60 | GO: open the EP-2 lane. Most tokens skip the peer hop entirely; the hop cost (peer read of the hidden row + a second-card launch chain + the combine) is paid on a minority of tokens, so per-token expert bytes halve per card while the latency tail stays near the single-card walk. |
| 20% to 50% at R = 64, or >= 50% only with load worse than 40/60 | EXPLORE: the structure exists but is not enough for a hop-free design. Price it: expected hops per token = (cards/token - 1); the EP-2 lane is worth opening only if hop cost x that number is below the per-token expert-read time the split saves, measured on the pair (a one-day cell, no engine change). A lopsided load means one card idles; replication cannot fix that, a bandwidth-parallel split (both cards always work) is the design to evaluate instead. |
| < 20% at R = 64 | NO-GO for the co-activation split: routing is too close to independent for a placement to remove hops. The only EP-2 design left is the always-both-cards split, which competes with PP-2 on bandwidth alone and is a different lane. |

Two more reads the table must pass before a GO: the in-sample column must not exceed the held-out
column by more than ~10 points (otherwise the partition memorized the prompt, re-run with a
different prompt set), and the prime-only and decode-heavy files must agree in verdict (a split
that holds during prefill and fails during decode is a prefill-only optimization, and prefill is not
where the hop cost bites).

## Self-test (synthetic, on this rig, no GPU; NOT a measurement of the model)

Generator in the session scratchpad: 4 layers x 4000 tokens, 8 slots. Clustered dump (2 of 8 picks
from a 16-expert hot set, 6 from one of two 136-expert clusters): held-out P(single) 23.3% at R=0,
100% at R=16 (hot set replicated), 96-98% at R=32/64; random-halves 0.7% / 3.0%. Uniform-random
dump: 0.7% at R=0 (matches 2 x 0.5^8), 1.5% at R=32; random-halves identical. A mixed n_sel dump
(6 and 8) parses through the sentinel pad. Runtime under 1 s per file at this size.
