#!/usr/bin/env python3
"""Build a per-layer expert->rank placement map from memra MoE routing traces.

Objective (quoted verbatim from darklanes agent-knowledge/gpu/kernel-craft.md):

  LAW:coactivation-expert-placement | scope: every TP/EP arrangement of MoE
  experts fleet-wide (owner directive 2026-08-31, verbatim: "dividing the
  expert base on which are most active together, it require messuring the
  expert and deciding what bundle of expert where, and that the expert that
  always active are on a known card that the token go to imiidieatly then
  passed away") | expert placement is MEASURED, never even-split: (1) measure
  per-layer expert co-activation on real traffic pools, (2) partition experts
  into per-card bundles maximizing same-card top-k co-residency under VRAM
  balance, (3) pin the always-active set (shared expert + top-frequency
  experts) to a KNOWN card the token visits first and leaves immediately —
  deterministic first hop, peer dispatch only for the tail | keywords: EP,
  expert placement, co-activation, bundles, shared expert, first hop, MoE
  sharding | src: owner order 2026-08-31 + tp2-battery class-gate receipts
  (naive EP peer-touch ~99.3%) | since: 2026-08-31

Input traces (written by crates/memra-engine/src/hybrid_forward.rs,
trace_moe_routes):

  MEMRA_MOE_TRACE         one line per (layer, forward): "<layer> <t> <id,id,...>"
  MEMRA_MOE_WEIGHT_TRACE  one line per (layer, forward): "<layer> <t> <expert:weight,...>"

Decode steps are t == 1. Weight traces, when given, replace pick counts as the
expert hotness signal (seed ordering, frequency ordering, entry-rank choice);
co-occurrence always comes from the id traces.

Strategies:
  even          contiguous ranges, rank = expert // (expert_count / ranks);
                exact reproduction of the engine's current law (control arm).
  frequency     experts sorted by hotness desc, greedy to the lightest rank,
                global hottest expert pinned to the entry rank first.
  coactivation  per layer, greedy bundle growth over within-line co-occurrence:
                seed = hottest unassigned expert, grow by max co-occurrence to
                the bundle until the fair-share quota (expert_count / ranks,
                within --balance-tolerance); the first (hottest-seed) bundle
                lands on the entry rank. Deterministic tie-breaks everywhere:
                hotness desc, then id asc. No RNG.

Output: JSON, FORMAT "memra-ep-map-v1". Every layer row carries stats computed
from the trace under the chosen assignment AND the even baseline, so each run
self-receipts its improvement. Engine consumption (MEMRA_GLM5_EP_MAP and
friends) is each serving lane's seam work; this tool only mints the map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

FORMAT = "memra-ep-map-v1"
STRATEGIES = ("coactivation", "frequency", "even")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_layers(raw: str | None) -> set[int] | None:
    if raw is None:
        return None
    return {int(x) for x in raw.split(",") if x.strip()}


def check_id(expert: int, expert_count: int, where: str) -> None:
    if expert < 0 or expert >= expert_count:
        raise SystemExit(
            f"{where}: expert id {expert} outside --expert-count {expert_count}"
        )


def read_id_traces(
    paths: list[Path],
    expert_count: int,
    decode_only: bool,
    layers: set[int] | None,
) -> tuple[dict[int, list[list[int]]], list[dict[str, Any]]]:
    """Return {layer: [ids per line]} plus per-file receipts."""
    lines_by_layer: dict[int, list[list[int]]] = defaultdict(list)
    receipts = []
    for path in paths:
        n_lines = 0
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                fields = line.split(maxsplit=2)
                if len(fields) != 3:
                    raise SystemExit(
                        f"{path}:{line_no}: expected '<layer> <t> <id,id,...>'"
                    )
                layer, t = int(fields[0]), int(fields[1])
                if decode_only and t != 1:
                    continue
                if layers is not None and layer not in layers:
                    continue
                ids = [int(x) for x in fields[2].split(",") if x]
                for expert in ids:
                    check_id(expert, expert_count, f"{path}:{line_no}")
                lines_by_layer[layer].append(ids)
        receipts.append({"path": str(path), "lines": n_lines, "sha256": sha256(path)})
    return dict(lines_by_layer), receipts


def read_weight_traces(
    paths: list[Path],
    expert_count: int,
    decode_only: bool,
    layers: set[int] | None,
) -> tuple[dict[int, dict[int, float]], list[dict[str, Any]]]:
    """Return {layer: {expert: summed routing weight}} plus receipts."""
    mass: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    receipts = []
    for path in paths:
        n_lines = 0
        with path.open() as handle:
            for line_no, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                n_lines += 1
                fields = line.split(maxsplit=2)
                if len(fields) != 3:
                    raise SystemExit(
                        f"{path}:{line_no}: expected '<layer> <t> <expert:weight,...>'"
                    )
                layer, t = int(fields[0]), int(fields[1])
                if decode_only and t != 1:
                    continue
                if layers is not None and layer not in layers:
                    continue
                for pair in fields[2].split(","):
                    if not pair:
                        continue
                    expert_s, weight_s = pair.split(":", 1)
                    expert = int(expert_s)
                    check_id(expert, expert_count, f"{path}:{line_no}")
                    mass[layer][expert] += float(weight_s)
        receipts.append({"path": str(path), "lines": n_lines, "sha256": sha256(path)})
    return {k: dict(v) for k, v in mass.items()}, receipts


def pick_counts(lines: list[list[int]], expert_count: int) -> list[int]:
    counts = [0] * expert_count
    for ids in lines:
        for expert in ids:
            counts[expert] += 1
    return counts


def cooccurrence(lines: list[list[int]]) -> dict[tuple[int, int], int]:
    """Within-line unordered pair counts over the unique ids of each line."""
    co: dict[tuple[int, int], int] = defaultdict(int)
    for ids in lines:
        for a, b in combinations(sorted(set(ids)), 2):
            co[(a, b)] += 1
    return dict(co)


def even_assignment(expert_count: int, ranks: int) -> list[int]:
    """Engine law: contiguous ranges, rank = expert // (expert_count / ranks)."""
    per = max(1, expert_count // ranks)
    return [min(e // per, ranks - 1) for e in range(expert_count)]


def hot_order(heat: list[float]) -> list[int]:
    return sorted(range(len(heat)), key=lambda e: (-heat[e], e))


def frequency_assignment(
    heat: list[float], ranks: int, entry_rank: int
) -> list[int]:
    expert_count = len(heat)
    cap = math.ceil(expert_count / ranks)
    assignment = [-1] * expert_count
    load = [0] * ranks
    order = hot_order(heat)
    assignment[order[0]] = entry_rank
    load[entry_rank] += 1
    for expert in order[1:]:
        rank = min(
            (r for r in range(ranks) if load[r] < cap),
            key=lambda r: (load[r], r),
        )
        assignment[expert] = rank
        load[rank] += 1
    return assignment


def coactivation_assignment(
    heat: list[float],
    co: dict[tuple[int, int], int],
    ranks: int,
    entry_rank: int,
    tolerance: float,
) -> list[int]:
    expert_count = len(heat)
    quota = expert_count / ranks
    hard_cap = max(math.ceil(quota), math.floor(quota * (1.0 + tolerance)))
    min_size = max(1, math.floor(quota * (1.0 - tolerance)))
    remaining = set(range(expert_count))

    def co_score(expert: int, bundle: list[int]) -> int:
        total = 0
        for member in bundle:
            key = (expert, member) if expert < member else (member, expert)
            total += co.get(key, 0)
        return total

    bundles: list[list[int]] = []
    for b in range(ranks):
        bundles_left = ranks - b
        fair_cap = math.ceil(len(remaining) / bundles_left)
        seed = min(remaining, key=lambda e: (-heat[e], e))
        bundle = [seed]
        remaining.discard(seed)
        while remaining and len(bundle) < fair_cap:
            best = min(
                remaining, key=lambda e: (-co_score(e, bundle), -heat[e], e)
            )
            bundle.append(best)
            remaining.discard(best)
        # Tolerance slack: keep pulling positively co-activated stragglers as
        # long as later bundles are left at least min_size each.
        bundles_after = ranks - b - 1
        while remaining and len(bundle) < hard_cap:
            best = min(
                remaining, key=lambda e: (-co_score(e, bundle), -heat[e], e)
            )
            if co_score(best, bundle) <= 0:
                break
            if len(remaining) - 1 < bundles_after * min_size:
                break
            bundle.append(best)
            remaining.discard(best)
        bundles.append(bundle)

    # First (hottest-seed) bundle to the entry rank; the rest fill the
    # remaining rank indices in ascending order.
    rank_order = [entry_rank] + [r for r in range(ranks) if r != entry_rank]
    assignment = [-1] * expert_count
    for bundle, rank in zip(bundles, rank_order):
        for expert in bundle:
            assignment[expert] = rank
    return assignment


def trace_stats(
    lines: list[list[int]], assignment: list[int], ranks: int
) -> tuple[float | None, float, float]:
    """(intra_rank_coactivation_fraction, expected_max_rank_touch,
    peer_touch_fraction) for one assignment over one layer's trace lines."""
    same_pairs = 0
    total_pairs = 0
    max_touch_sum = 0
    peer_lines = 0
    for ids in lines:
        uniq = sorted(set(ids))
        for a, b in combinations(uniq, 2):
            total_pairs += 1
            if assignment[a] == assignment[b]:
                same_pairs += 1
        per_rank = [0] * ranks
        for expert in ids:
            per_rank[assignment[expert]] += 1
        max_touch_sum += max(per_rank)
        if sum(1 for c in per_rank if c > 0) > 1:
            peer_lines += 1
    n = len(lines)
    intra = (same_pairs / total_pairs) if total_pairs else None
    return intra, max_touch_sum / n, peer_lines / n


def build_layer_row(
    layer: int,
    lines: list[list[int]],
    heat: list[float],
    strategy: str,
    expert_count: int,
    ranks: int,
    entry_rank: int,
    tolerance: float,
) -> dict[str, Any]:
    even = even_assignment(expert_count, ranks)
    if strategy == "even":
        assignment = even
    elif strategy == "frequency":
        assignment = frequency_assignment(heat, ranks, entry_rank)
    else:
        co = cooccurrence(lines)
        assignment = coactivation_assignment(heat, co, ranks, entry_rank, tolerance)
    intra, exp_max, peer = trace_stats(lines, assignment, ranks)
    _, even_max, _ = trace_stats(lines, even, ranks)
    return {
        "layer": layer,
        "assignment": assignment,
        "stats": {
            "intra_rank_coactivation_fraction": intra,
            "expected_max_rank_touch": exp_max,
            "even_baseline_expected_max_rank_touch": even_max,
            "peer_touch_fraction": peer,
        },
    }


def build_map(
    lines_by_layer: dict[int, list[list[int]]],
    weight_by_layer: dict[int, dict[int, float]],
    strategy: str,
    expert_count: int,
    ranks: int,
    entry_rank: int,
    tolerance: float,
) -> list[dict[str, Any]]:
    rows = []
    for layer in sorted(lines_by_layer):
        lines = lines_by_layer[layer]
        if weight_by_layer.get(layer):
            heat = [0.0] * expert_count
            for expert, w in weight_by_layer[layer].items():
                heat[expert] = w
        else:
            heat = [float(c) for c in pick_counts(lines, expert_count)]
        rows.append(
            build_layer_row(
                layer, lines, heat, strategy, expert_count, ranks,
                entry_rank, tolerance,
            )
        )
    return rows


def selftest() -> int:
    checks: list[str] = []

    def ok(name: str, cond: bool, detail: str) -> None:
        if not cond:
            raise SystemExit(f"selftest FAIL [{name}]: {detail}")
        checks.append(f"selftest PASS [{name}]: {detail}")

    # Two disjoint cliques, interleaved across the even boundary so that the
    # contiguous law splits both. Clique A (hotter, 10 lines) = {0,2,4,6};
    # clique B (6 lines) = {1,3,5,7}.
    e_count, ranks, entry = 8, 2, 0
    clique_a, clique_b = [0, 2, 4, 6], [1, 3, 5, 7]
    lines = [list(clique_a) for _ in range(10)] + [list(clique_b) for _ in range(6)]
    heat = [float(c) for c in pick_counts(lines, e_count)]

    # (a) coactivation: cliques on separate ranks, hotter clique on entry rank,
    # intra_rank_coactivation_fraction 1.0.
    row = build_layer_row(0, lines, heat, "coactivation", e_count, ranks, entry, 0.05)
    asg = row["assignment"]
    ok(
        "a.cliques-separate",
        len({asg[e] for e in clique_a}) == 1
        and len({asg[e] for e in clique_b}) == 1
        and asg[clique_a[0]] != asg[clique_b[0]],
        f"assignment={asg}",
    )
    ok("a.hot-clique-on-entry", all(asg[e] == entry for e in clique_a), f"assignment={asg}")
    intra = row["stats"]["intra_rank_coactivation_fraction"]
    ok("a.intra-fraction-1.0", intra == 1.0, f"intra={intra}")

    # (b) even strategy equals the contiguous engine law exactly.
    ok(
        "b.even-8x2",
        even_assignment(8, 2) == [e // 4 for e in range(8)],
        f"map={even_assignment(8, 2)}",
    )
    ok(
        "b.even-64x4",
        even_assignment(64, 4) == [e // 16 for e in range(64)],
        "matches expert // (count/ranks) for 64 experts on 4 ranks",
    )

    # (c) frequency puts the global hottest expert on the entry rank.
    freq_heat = [1.0] * e_count
    freq_heat[5] = 100.0
    freq = frequency_assignment(freq_heat, ranks, 1)
    ok("c.hottest-on-entry", freq[5] == 1, f"assignment={freq}")
    loads = [freq.count(r) for r in range(ranks)]
    ok("c.balanced", max(loads) - min(loads) <= 1, f"loads={loads}")

    # (d) stats sanity on the clique trace: peer_touch 0.0 under coactivation,
    # > 0 under even (each clique straddles the contiguous boundary).
    peer_co = row["stats"]["peer_touch_fraction"]
    ok("d.coactivation-peer-0", peer_co == 0.0, f"peer_touch_fraction={peer_co}")
    even_row = build_layer_row(0, lines, heat, "even", e_count, ranks, entry, 0.05)
    peer_even = even_row["stats"]["peer_touch_fraction"]
    ok("d.even-peer-positive", peer_even > 0.0, f"peer_touch_fraction={peer_even}")
    ok(
        "d.max-touch-improves",
        row["stats"]["expected_max_rank_touch"] == 4.0
        and row["stats"]["even_baseline_expected_max_rank_touch"] == 2.0,
        f"coact={row['stats']['expected_max_rank_touch']} even={row['stats']['even_baseline_expected_max_rank_touch']}",
    )

    for line in checks:
        print(line)
    print(f"selftest: {len(checks)}/{len(checks)} checks green")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trace", type=Path, action="append", default=[])
    parser.add_argument("--weight-trace", type=Path, action="append", default=[])
    parser.add_argument("--ranks", type=int)
    parser.add_argument("--entry-rank", type=int, default=0)
    parser.add_argument("--strategy", choices=STRATEGIES)
    parser.add_argument("--expert-count", type=int)
    parser.add_argument("--balance-tolerance", type=float, default=0.05)
    parser.add_argument("--decode-only", action="store_true")
    parser.add_argument("--layers", type=str, default=None)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        return selftest()
    for name in ("trace", "ranks", "strategy", "expert_count", "out"):
        if not getattr(args, name):
            parser.error(f"--{name.replace('_', '-')} is required (or use --selftest)")
    if args.ranks < 1 or args.expert_count < args.ranks:
        parser.error("need expert_count >= ranks >= 1")
    if not 0 <= args.entry_rank < args.ranks:
        parser.error("--entry-rank must be in [0, ranks)")

    layers = parse_layers(args.layers)
    lines_by_layer, trace_receipts = read_id_traces(
        args.trace, args.expert_count, args.decode_only, layers
    )
    weight_by_layer, weight_receipts = read_weight_traces(
        args.weight_trace, args.expert_count, args.decode_only, layers
    )
    if not lines_by_layer:
        raise SystemExit("no trace lines survived the filters; nothing to place")

    rows = build_map(
        lines_by_layer, weight_by_layer, args.strategy, args.expert_count,
        args.ranks, args.entry_rank, args.balance_tolerance,
    )
    doc = {
        "format": FORMAT,
        "strategy": args.strategy,
        "ranks": args.ranks,
        "entry_rank": args.entry_rank,
        "expert_count": args.expert_count,
        "traces": trace_receipts,
        "params": {
            "balance_tolerance": args.balance_tolerance,
            "decode_only": args.decode_only,
            "layers": sorted(layers) if layers is not None else None,
            "weight_traces": weight_receipts,
            "hotness_signal": "routing-weight-mass" if weight_receipts else "pick-count",
        },
        "layers": rows,
    }
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out} ({len(rows)} layers, strategy={args.strategy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
