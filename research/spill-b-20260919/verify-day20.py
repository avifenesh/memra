#!/usr/bin/env python3
"""Offline replay of the day-20 prefix-policy A/B receipts on the second card class (memra#523 item 2).

NOT CUDA execution, NOT a cross-box timing comparison, NOT serving qualification. The replay re-derives the
harness's outcome from the recorded per-run rows (never trusting `VERDICT.txt`): the collector journal and lock
proof (the canonical `/tmp/memra-5090.lock`, rig `rtx5090`), the binary bound to its build receipt and to the
pre-registration source ref `9466b8912` plus the three cherry-picked capture commits, the card (the local RTX 5090
Laptop GPU), the shape gates and the pre-registered grid-aligned shares, the interleaved AB/BA schedule, every
run's primary and secondary recomputed from its rows, digest identity across runs and against the cache-off boot,
the grid law on every published and restored length, the pre-registered rule re-applied, the per-request
`cached_tokens` compared with `day20-predict.py`, and the admission-reclaim and driver-trim lines counted from
each run's server log (the day-16 confound; the scored cell must read zero reclaim events).

usage: verify-day20.py [--raw research/spill-b-20260919/rtx5090-day20] [--cell ab-full] [--expect-pool 0]
"""
import argparse
import importlib.util
import json
import re
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARD = "NVIDIA GeForce RTX 5090 Laptop GPU"
LOCK, RIG = "/tmp/memra-5090.lock", "rtx5090"
PREREG = "9466b89122a5815dacef0dd54db30fb1476ed533"
PICKS = ("fe28c5c07", "74ee484bc", "a843686ce")  # 269178070, 49d79b946, fa5d7a014 on the detached tree
ARMS = ("slru", "lru")
BOOT_WORD = {"slru": "byte-SLRU", "lru": "plain-LRU"}
GRID = 32
SHAPE_ARGS = (1024, "1248,1344,1440,1536", 12, 10912, 160, 3, 16384, 8, LOCK)
SHARES = {"cohort_share_of_budget": 0.738, "turn1_share_of_budget": 0.448, "last_turn_share_of_budget": 0.497}
RE_RECLAIM = re.compile(r"reclaim-on-defer: evicted (\d+) prefix entries")
RE_TRIM = re.compile(r"\[admit-trim\] ")
RE_PARK = re.compile(r"plain-affinity: (\w+)")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path):
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def journal(cell_dir):
    rows = [json.loads(ln) for ln in (cell_dir / "CELL.jsonl").read_text().splitlines() if ln.strip()]
    require(rows, f"{cell_dir}: empty CELL.jsonl")
    lock = load_json(cell_dir / "lock.json")
    require(lock.get("lock") == LOCK and lock.get("rig") == RIG, f"{cell_dir}: not the canonical {RIG} lock: {lock}")
    capture = load_json(cell_dir / "command.capture.json")
    require(capture.get("qualification") is False, f"{cell_dir}: a capture must never claim qualification")
    return rows, lock, capture


def predicted():
    spec = importlib.util.spec_from_file_location("day20_predict", HERE / "day20-predict.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict(mod.SHAPES["day20"])


def replay_run(run_dir, plan, cold_digest):
    run = load_json(run_dir / "run.json")
    rows = run["rows"]
    require(len(rows) == len(plan), f"{run_dir.name}: {len(rows)} rows for a {len(plan)}-request plan")
    computed = 0
    published = {}  # salt -> last published length (the grid law: restored == published == previous prompt)
    grid_ok = 0
    for row, p in zip(rows, plan):
        require(row["idx"] == p["idx"] and row["role"] == p["role"] and row["salt"] == p["salt"], f"{run_dir.name}: row {row['idx']} is not the plan's request")
        require(row["prompt_tokens"] == p["prompt_ids"], f"{run_dir.name}: request {row['idx']} prompt_tokens {row['prompt_tokens']} != plan {p['prompt_ids']}")
        require(row["prompt_tokens"] % GRID == 0, f"{run_dir.name}: request {row['idx']} prompt {row['prompt_tokens']} is off the {GRID}-token grid")
        require(row["computed_tokens"] == row["prompt_tokens"] - row["cached_tokens"], f"{run_dir.name}: request {row['idx']} computed is not prompt - cached")
        computed += row["computed_tokens"]
        w = row["window"]
        for ins in w["inserts"]:
            require(ins["tokens"] == row["prompt_tokens"], f"{run_dir.name}: request {row['idx']} published {ins['tokens']} tokens for a {row['prompt_tokens']}-token prompt (the seed must land at the on-grid prompt end)")
            published[row["salt"]] = ins["tokens"]
        for hit in w["hits"]:
            require(hit["hit"] == row["cached_tokens"] and hit["prompt"] == row["prompt_tokens"], f"{run_dir.name}: request {row['idx']} hit line {hit} disagrees with cached_tokens {row['cached_tokens']}")
            require(hit["hit"] % GRID == 0, f"{run_dir.name}: request {row['idx']} restored {hit['hit']} tokens, off the grid")
            grid_ok += 1
        if row["role"] == "seed2":
            require(row["cached_tokens"] == row["prompt_tokens"] and w["hits"], f"{run_dir.name}: cohort tenant {row['tenant']} did not promote")
        if row["role"] == "loop" and row["cached_tokens"]:
            require(row["cached_tokens"] < row["prompt_tokens"] and (row["prompt_tokens"] - row["cached_tokens"]) % GRID == 0, f"{run_dir.name}: loop turn {row['turn']} restored {row['cached_tokens']} of {row['prompt_tokens']}")
    require(computed == run["computed_tokens"], f"{run_dir.name}: recorded primary {run['computed_tokens']} != replayed {computed}")
    returns = [r for r in rows if r["role"] in ("return", "final")]
    require(sum(r["cached_tokens"] for r in returns) == run["return_cached_sum"], f"{run_dir.name}: recorded secondary differs from the rows")
    loop = [r for r in rows if r["role"] == "loop"]
    require(sum(1 for r in loop[1:] if r["cached_tokens"] == 0) == run["loop_cold_after_1"], f"{run_dir.name}: loop cold count differs from the rows")
    require(BOOT_WORD[run["arm"]] in run["policy_line"], f"{run_dir.name}: boot line {run['policy_line']!r} is not the {run['arm']} arm")
    identical = sum(1 for r in rows if r["text_sha256"] == cold_digest[r["idx"]])
    require(identical == run["digests_identical_to_cold"], f"{run_dir.name}: recorded cold identity {run['digests_identical_to_cold']} != {identical}")
    require(not run["refused_or_skipped"], f"{run_dir.name}: refusal or skip lines recorded: {run['refused_or_skipped']}")
    log = (run_dir / "server.log").read_text(errors="replace")
    reclaim = [int(m) for m in RE_RECLAIM.findall(log)]
    trims = len(RE_TRIM.findall(log))
    affinity = RE_PARK.findall(log)
    require(all(a == "declined" for a in affinity), f"{run_dir.name}: a parked session served a request: {set(affinity)}")
    policy_evicts = sum(len(r["window"]["evicts"]) for r in rows)
    require(policy_evicts == run["evictions"], f"{run_dir.name}: recorded policy evictions differ from the rows")
    metric_evicts = rows[-1]["metrics_after"]["prefix_cache_evictions"]
    require(metric_evicts == policy_evicts + sum(reclaim), f"{run_dir.name}: /metrics evictions {metric_evicts} != policy {policy_evicts} + reclaim {sum(reclaim)}")
    mism = [(r["idx"], r["role"], r["prompt_tokens"], r["cached_tokens"], r["text_sha256"][:16], cold_digest[r["idx"]][:16]) for r in rows if r["text_sha256"] != cold_digest[r["idx"]]]
    headroom = [(r["metrics_after"]["cuda_driver_free_bytes"] + r["metrics_after"]["cuda_pool_cached_bytes"] + r["metrics_after"]["prefix_cache_bytes"]) / 1e6 for r in rows]
    return run, [r["text_sha256"] for r in rows], {"events": len(reclaim), "entries": sum(reclaim), "trims": trims, "affinity_declined": len(affinity), "policy_evicts": policy_evicts, "mismatches": mism, "grid_hits": grid_ok, "headroom_min_mb": min(headroom), "headroom_max_mb": max(headroom)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "rtx5090-day20")
    ap.add_argument("--cell", default="ab-full")
    ap.add_argument("--build", default="build-tip")
    ap.add_argument("--expect-pool", default="0", help="the MEMRA_REUSE_POOL the cell was run with ('default' or '0'); a '0' cell must show zero reclaim events")
    args = ap.parse_args()
    raw = args.raw
    cell_dir = raw / args.cell
    _rows, lock, capture = journal(cell_dir)
    env = (cell_dir / "env.txt").read_text().strip()
    require(env == f"MEMRA_REUSE_POOL={args.expect_pool}", f"the cell ran with {env!r}, expected MEMRA_REUSE_POOL={args.expect_pool}")
    cell = cell_dir / "cell"
    summary = load_json(cell / "summary.json")
    verdict_txt = (cell / "VERDICT.txt").read_text().strip()
    require(summary["verdict"] == verdict_txt, "VERDICT.txt and summary.json disagree")
    rig = load_json(cell / "rig.json")
    require(CARD in rig["nvidia_smi"], f"rig line is not the local RTX 5090 Laptop GPU: {rig['nvidia_smi']}")
    require(rig["binary_sha256"] == summary["binary_sha256"], "rig.json and summary.json binary digests differ")
    build = raw / args.build
    recorded = (build / "binary.sha256").read_text().split()[0]
    require(recorded == summary["binary_sha256"], f"the cell ran binary {summary['binary_sha256'][:16]} but the build receipt records {recorded[:16]}")
    require((build / "exit").read_text().strip() == "0" and (build / "dirty.txt").read_text().strip() == "", "build receipt not clean")
    source = (build / "source.txt").read_text().strip()
    source_log = (build / "source-log.txt").read_text().splitlines()
    require(source.startswith(PICKS[-1]) and source_log[-1].startswith(PREREG) and all(ln.startswith(p) for ln, p in zip(source_log, reversed(PICKS))), f"the binary is not the pre-registration tree plus the three capture picks: {source_log}")
    lockproof = load_json(cell / "LOCK.json")
    require(lockproof.get("lock", lock.get("lock")) == LOCK, f"harness lock proof is not the canonical lock: {lockproof}")
    require(lockproof.get("inode") == lock.get("inode") and lockproof.get("device") == lock.get("device"), "harness and collector lock proofs name different inodes")
    require((cell_dir / "gate-source.txt").read_text().strip() == source, "the harness did not run from the built tree")
    plan = load_json(cell / "plan.json")
    plan_args, plan = plan["args"], plan["requests"]
    require((plan_args["budget_mib"], plan_args["cohort_tokens"], plan_args["turns"], plan_args["start_tokens"], plan_args["grow_tokens"], plan_args["return_every"], plan_args["ctx"], plan_args["max_tokens"], plan_args["gpu_lock"]) == SHAPE_ARGS, f"the plan is not the pre-registered grid-aligned shape: {plan_args}")
    require(all(p["prompt_ids"] % GRID == 0 for p in plan), "a planned prompt is off the grid")
    shape = summary["shape"]
    for gate in ("cohort_within_protected_share", "turn1_exceeds_free_share", "every_turn_fits_budget", "two_last_turns_fit_together"):
        require(shape[gate], f"shape gate {gate} did not hold: {shape}")
    require(shape["budget_bytes"] == 1024 << 20, f"budget {shape['budget_bytes']} is not 1024 MiB")
    for key, want in SHARES.items():
        require(abs(shape[key] - want) < 0.01, f"{key} {shape[key]:.3f} is not the pre-registered {want:.3f}")
    cold = load_json(cell / "cold" / "rows.json")
    require(len(cold) == len(plan) and all(not r["cached_tokens"] for r in cold), "the cache-off boot is not a cold replay of the plan")
    cold_digest = {r["idx"]: r["text_sha256"] for r in cold}
    cold_log = (cell / "cold" / "server.log").read_text(errors="replace")
    cold_reclaim = len(RE_RECLAIM.findall(cold_log))

    sched = summary["schedule"]
    n = len(sched) // 4
    require(n >= 1 and len(sched) == 4 * n, f"schedule of {len(sched)} runs is not whole pairs in both orders")
    smoke = n < 5
    expected = []
    for i in range(n):
        for order in ("AB", "BA"):
            for pos, letter in enumerate(order):
                expected.append((f"{order}-{i}", order, pos, "slru" if letter == "A" else "lru"))
    require([(s["pair_id"], s["order"], s["position"], s["arm"]) for s in sched] == expected, "the schedule is not the interleaved AB/BA pairing")
    run_dirs = sorted((cell / "runs").iterdir())
    require(len(run_dirs) == len(sched), f"{len(run_dirs)} run directories for {len(sched)} scheduled runs")
    runs, digests, extras = [], [], []
    for d, s in zip(run_dirs, sched):
        run, dg, extra = replay_run(d, plan, cold_digest)
        require((run["pair_id"], run["order"], run["position"], run["arm"]) == (s["pair_id"], s["order"], s["position"], s["arm"]), f"{d.name}: run identity differs from the schedule")
        runs.append(run)
        digests.append(dg)
        extras.append(extra)
    per_request = [len({cold_digest[p["idx"]]} | {dg[p["idx"]] for dg in digests}) == 1 for p in plan]
    identical = all(per_request)
    require(identical == summary["digests_identical"], "recorded digest identity differs from the replay")
    require(sum(per_request) == int(verdict_txt.split("digests_identical=")[1].split("/")[0]), "the verdict line's digest count differs from the replay")
    for arm in ARMS:
        arm_digests = [dg for run, dg in zip(runs, digests) if run["arm"] == arm]
        require(all(len({dg[i] for dg in arm_digests}) == 1 for i in range(len(plan))), f"{arm}: two runs of the same arm produced different bytes")
    pairs = []
    for i in range(n):
        for order in ("AB", "BA"):
            pid = f"{order}-{i}"
            a = next(r for r in runs if r["pair_id"] == pid and r["arm"] == "slru")
            b = next(r for r in runs if r["pair_id"] == pid and r["arm"] == "lru")
            pairs.append((pid, a["computed_tokens"], b["computed_tokens"], a["return_cached_sum"], b["return_cached_sum"]))
    rec_pairs = summary["pairs"]
    require([(p["pair_id"], p["slru_computed"], p["lru_computed"]) for p in rec_pairs] == [(p[0], p[1], p[2]) for p in pairs], "recorded pairs differ from the runs")
    slru_all = all(a < b for _, a, b, _, _ in pairs)
    lru_all = all(b < a for _, a, b, _, _ in pairs)
    rule = "WINNER=slru" if slru_all else "WINNER=lru" if lru_all else "INCONCLUSIVE"
    outcome = "DIGEST-FAIL" if not identical else "SMOKE" if smoke else rule
    require(verdict_txt.endswith(f"-> {outcome}"), f"the verdict line does not end with the replayed outcome {outcome}")
    require(summary["winner"] == (None if smoke or not identical else rule), f"recorded winner {summary['winner']} differs from the replay")
    if identical:
        require(capture["status"] == "executed-not-qualified" and capture["exit_code"] == 0, f"collector status {capture['status']} exit {capture['exit_code']}")
    else:
        require(capture["status"] == "failed" and capture["exit_code"] == 1, f"a digest failure must be a failed capture with exit 1, not {capture['status']} exit {capture['exit_code']}")
    reclaim_events = sum(e["events"] for e in extras)
    if args.expect_pool == "0":
        require(reclaim_events == 0 and cold_reclaim == 0, f"a MEMRA_REUSE_POOL=0 cell must read zero reclaim events; read {reclaim_events} across the runs and {cold_reclaim} in the cold boot")
    stats = {}
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        comp = [r["computed_tokens"] for r in rs]
        ret = [r["return_cached_sum"] for r in rs]
        require(len(rs) == 2 * n, f"{arm}: {len(rs)} runs for {n} pairs per order")
        require(statistics.median(comp) == summary["arms"][arm]["computed_tokens"]["median"], f"{arm}: recorded median differs")
        stats[arm] = (len(rs), statistics.median(comp), min(comp), max(comp), statistics.median(ret), max(r["loop_cold_after_1"] for r in rs), [r["evictions"] for r in rs])
    temps = [t for r in runs for t in (r["gpu_start"]["temperature_c"], r["gpu_end"]["temperature_c"]) if t is not None]
    pred = predicted()
    matched = {arm: 0 for arm in ARMS}
    deviations = []
    for run in runs:
        for row, (_role, _tenant, prompt, cached, _computed) in zip(run["rows"], pred[run["arm"]]["rows"]):
            require(row["prompt_tokens"] == prompt, f"{run['pair_id']} {run['arm']}: predicted table and plan disagree on request {row['idx']}")
            if row["cached_tokens"] == cached:
                matched[run["arm"]] += 1
            else:
                deviations.append((run["pair_id"], run["arm"], row["idx"], row["role"], row["salt"], cached, row["cached_tokens"]))
    total = sum(len(r["rows"]) for r in runs) // 2

    print("DAY20 REPLAY OK" + ("" if identical else ": receipts consistent, precondition FAILED on this card, no verdict"))
    print(f"source {source[:9]} (= {PREREG[:9]} + picks {', '.join(PICKS)}) binary {summary['binary_sha256'][:16]} {env}: {verdict_txt}")
    for pid, a, b, ra, rb in pairs:
        better = "slru" if a < b else "lru" if b < a else "tie"
        print(f"  pair {pid}: slru {a} vs lru {b} computed tokens (diff {a - b:+d}, {better}); return cached slru {ra} lru {rb}")
    for arm, (N, med, lo, hi, rmed, cold_max, ev) in stats.items():
        print(f"{arm}: N={N} computed median {med:.0f} (min {lo}, max {hi}); return cached median {rmed:.0f}; loop cold after turn 1 max {cold_max}; policy evictions {ev}")
    print(f"digests identical to the cache-off boot on {sum(per_request)}/{len(plan)} requests across {len(runs)} runs; identical across runs of the same arm on {len(plan)}/{len(plan)}")
    print(f"grid: every prompt, every published entry and every restored length a multiple of {GRID}; restored == previous prompt on every loop hit; {sum(e['grid_hits'] for e in extras)} hit lines checked")
    for run, extra in zip(runs, extras):
        print(f"  {run['pair_id']} {run['arm']}: policy evict lines {extra['policy_evicts']}, admission reclaim events {extra['events']} evicting {extra['entries']} prefix entries, admit-trim lines {extra['trims']}, plain-affinity declined {extra['affinity_declined']}, non-cache headroom {extra['headroom_min_mb']:.0f}..{extra['headroom_max_mb']:.0f} MB; digest mismatches vs cold {extra['mismatches']}")
    print(f"reclaim events: {reclaim_events} across the runs, {cold_reclaim} in the cache-off boot ({env})")
    print(f"shape: cohort {shape['cohort_share_of_budget']:.3f}, turn 1 {shape['turn1_share_of_budget']:.3f}, turn 12 {shape['last_turn_share_of_budget']:.3f} of the {shape['budget_bytes']} B budget; fit {shape['fit_bytes_per_token']:.0f} B/token + {shape['fit_fixed_bytes']:.0f} B")
    print(f"rig: {rig['nvidia_smi']}; temperature {min(temps)}..{max(temps)} C at run boundaries; power limit {runs[0]['gpu_start']['power_limit_w']} (nvidia-smi reports no limit on this laptop GPU)")
    print(f"prediction (day20-predict.py): slru {pred['slru']['computed']} lru {pred['lru']['computed']}; per-request cached_tokens matched slru {matched['slru']}/{total} lru {matched['lru']}/{total}")
    for dev in deviations:
        print("  deviation from the prediction:", dev)
    print(f"replayed rule on the primary: {rule}; harness outcome: {outcome}; collector {capture['status']} exit {capture['exit_code']}; no cross-box timing compared.")


if __name__ == "__main__":
    main()
