# ep-placement-map-20260831: shared expert-placement map tool

Lane: `lane/ep-placement-map-20260831` (base `origin/main` ce0a42520).
Deliverable: `tools/build_expert_placement_map.py`, stdlib-only Python, no RNG.

## Objective (law, quoted verbatim; punctuation is the source's)

From `darklanes/agent-knowledge/gpu/kernel-craft.md:49`:

> LAW:coactivation-expert-placement | scope: every TP/EP arrangement of MoE experts fleet-wide (owner directive 2026-08-31, verbatim: "dividing the expert base on which are most active together, it require messuring the expert and deciding what bundle of expert where, and that the expert that always active are on a known card that the token go to imiidieatly then passed away") | expert placement is MEASURED, never even-split: (1) measure per-layer expert co-activation on real traffic pools, (2) partition experts into per-card bundles maximizing same-card top-k co-residency under VRAM balance, (3) pin the always-active set (shared expert + top-frequency experts) to a KNOWN card the token visits first and leaves immediately — deterministic first hop, peer dispatch only for the tail | keywords: EP, expert placement, co-activation, bundles, shared expert, first hop, MoE sharding | src: owner order 2026-08-31 + tp2-battery class-gate receipts (naive EP peer-touch ~99.3%) | since: 2026-08-31

## Input trace format (verified in engine source)

Written by `crates/memra-engine/src/hybrid_forward.rs`, `trace_moe_routes` (write
site at the `MEMRA_MOE_TRACE` branch, around line 5958 at ce0a42520):

- `MEMRA_MOE_TRACE`: one line per (layer, forward call): `<layer> <t> <id,id,...>`.
  Decode steps are `t == 1` (`--decode-only` keeps only those).
- `MEMRA_MOE_WEIGHT_TRACE`: one line per (layer, forward call):
  `<layer> <t> <expert:weight,...>`.

When weight traces are supplied, summed routing weight replaces pick count as
the hotness signal (seed order, frequency order, entry-rank choice).
Co-occurrence always comes from the id traces.

## Frozen output format: memra-ep-map-v1

```json
{
  "format": "memra-ep-map-v1",
  "strategy": "coactivation | frequency | even",
  "ranks": N,
  "entry_rank": 0,
  "expert_count": E,
  "traces": [{"path": "...", "lines": L, "sha256": "..."}],
  "params": {
    "balance_tolerance": 0.05,
    "decode_only": true,
    "layers": null,
    "weight_traces": [],
    "hotness_signal": "pick-count | routing-weight-mass"
  },
  "layers": [
    {
      "layer": L,
      "assignment": [rank per expert, index 0..E-1],
      "stats": {
        "intra_rank_coactivation_fraction": x,
        "expected_max_rank_touch": y,
        "even_baseline_expected_max_rank_touch": z,
        "peer_touch_fraction": p
      }
    }
  ]
}
```

Stats semantics, all computed from the input trace lines of that layer:

- `intra_rank_coactivation_fraction`: fraction of within-line expert pairs
  landing on the same rank (1.0 = every co-activation is co-resident).
- `expected_max_rank_touch`: mean over lines of the max experts-per-rank.
  Under the law's goal this RISES toward top-k (the whole selection sits on
  one card); it is not a load metric.
- `even_baseline_expected_max_rank_touch`: the same number under the engine's
  contiguous law, so every run self-receipts against the control arm.
- `peer_touch_fraction`: fraction of lines touching more than one rank, i.e.
  the peer-dispatch rate the tp2-battery receipts measured at ~99.3% for naive
  EP. This is the number the coactivation strategy exists to crush.

Strategies:

- `even`: contiguous ranges, `rank = expert // (expert_count / ranks)`. Exact
  reproduction of the engine's current law; the control arm.
- `frequency`: hotness-descending greedy to the lightest rank, global hottest
  expert pinned to the entry rank first. Balanced within one expert.
- `coactivation`: per layer, greedy bundle growth over within-line
  co-occurrence: seed = hottest unassigned expert, grow by max
  co-occurrence-to-bundle until the fair-share quota (expert_count / ranks,
  slack bounded by `--balance-tolerance`); first (hottest-seed) bundle goes to
  the entry rank. Tie-breaks: hotness desc, then id asc. Fully deterministic.

## Example run (files committed in this directory)

`example-trace.txt`: 16 experts, 2 layers, top-k 4, 59 lines. Layer 0 has four
interleaved co-activation bundles that the contiguous law cuts in half plus two
mixed lines; layer 1 has contiguous cliques the even law already handles; one
`t=64` prefill line is dropped by `--decode-only`.

```
python3 tools/build_expert_placement_map.py \
  --trace research/ep-placement-map-20260831/example-trace.txt \
  --ranks 2 --entry-rank 0 --strategy coactivation \
  --expert-count 16 --decode-only \
  --out research/ep-placement-map-20260831/example-map-coactivation.json
```

(same for `frequency` and `even`)

| strategy | layer | intra_rank_coactivation_fraction | expected_max_rank_touch | even_baseline_expected_max_rank_touch | peer_touch_fraction |
|---|---|---|---|---|---|
| coactivation | 0 | 0.9688 | 3.9375 | 2.7500 | 0.0625 |
| coactivation | 1 | 1.0000 | 4.0000 | 4.0000 | 0.0000 |
| frequency | 0 | 0.3438 | 2.0625 | 2.7500 | 1.0000 |
| frequency | 1 | 0.3333 | 2.0000 | 4.0000 | 1.0000 |
| even | 0 | 0.5833 | 2.7500 | 2.7500 | 0.6250 |
| even | 1 | 1.0000 | 4.0000 | 4.0000 | 0.0000 |

Reading: on the interleaved layer 0, coactivation cuts peer touch from 0.625
(even) to 0.0625 and lifts co-residency to 0.9688; the hot bundle {1,5,9,13}
sits on the entry rank. Frequency balances load but touches both ranks on
every line, which is exactly the naive-EP failure mode the law names.

## Selftest

`python3 tools/build_expert_placement_map.py --selftest` is 10/10 green;
verbatim output in `selftest-receipt.txt`. Teeth were proven once: assertion
`d.coactivation-peer-0` was temporarily inverted, the suite failed with exit 1
on exactly that check, and the assertion was restored (receipt notes this).

## Consumption seam

This tool mints the map only. Engine consumption (`MEMRA_GLM5_EP_MAP` and the
hy3 twin, loading `memra-ep-map-v1`, dispatching experts by the per-layer
`assignment` array, and routing the token through the entry rank first) is
each serving lane's seam work: the glm5 and hy3 lanes own wiring, gating, and
the real-traffic trace capture that replaces the synthetic example here.
