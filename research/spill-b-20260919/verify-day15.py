#!/usr/bin/env python3
"""Offline replay of the day-15 prefix-policy A/B receipts (memra#523 item 2).

NOT CUDA execution, NOT a cross-box timing comparison, NOT serving qualification: the collector
labels every cell `executed-not-qualified`. The replay re-derives the verdict from the recorded
per-run rows (never trusting `VERDICT.txt`), checks the collector journal and lock proof, binds the
binary to its build receipt and source ref, confirms the rig (one RTX PRO 6000 Blackwell at 600 W),
confirms the shape gates the harness asserted, confirms the schedule is the interleaved AB/BA
pairing with N pairs per order, recomputes every run's primary and secondary from its rows,
re-checks digest identity across runs and against the cache-off boot, and re-applies the
pre-registered rule: a policy wins only if better on the primary at every pair in both orders.

usage: verify-day15.py [--raw research/spill-b-20260919/pro-single-day15] [--cell ab-full]
"""
import argparse
import json
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
POWER = "600.00 W, 600.00 W"
CARD = "NVIDIA RTX PRO 6000 Blackwell"
ARMS = ("slru", "lru")
BOOT_WORD = {"slru": "byte-SLRU", "lru": "plain-LRU"}


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
    require(lock.get("lock") == "/tmp/memra-gpu.lock" and lock.get("rig") == "pro-single", f"{cell_dir}: not the canonical pro-single lock: {lock}")
    capture = load_json(cell_dir / "command.capture.json")
    require(capture.get("qualification") is False, f"{cell_dir}: a capture must never claim qualification")
    return rows, lock, capture


def replay_run(run_dir, plan, cold_digest):
    run = load_json(run_dir / "run.json")
    rows = run["rows"]
    require(len(rows) == len(plan), f"{run_dir.name}: {len(rows)} rows for a {len(plan)}-request plan")
    computed = 0
    for row, p in zip(rows, plan):
        require(row["idx"] == p["idx"] and row["role"] == p["role"] and row["salt"] == p["salt"], f"{run_dir.name}: row {row['idx']} is not the plan's request")
        require(row["prompt_tokens"] == p["prompt_ids"], f"{run_dir.name}: request {row['idx']} prompt_tokens {row['prompt_tokens']} != plan {p['prompt_ids']}")
        require(row["computed_tokens"] == row["prompt_tokens"] - row["cached_tokens"], f"{run_dir.name}: request {row['idx']} computed is not prompt - cached")
        computed += row["computed_tokens"]
        if row["role"] == "seed2":
            require(row["cached_tokens"] == row["prompt_tokens"] and row["window"]["hits"], f"{run_dir.name}: cohort tenant {row['tenant']} did not promote")
    require(computed == run["computed_tokens"], f"{run_dir.name}: recorded primary {run['computed_tokens']} != replayed {computed}")
    returns = [r for r in rows if r["role"] in ("return", "final")]
    require(sum(r["cached_tokens"] for r in returns) == run["return_cached_sum"], f"{run_dir.name}: recorded secondary differs from the rows")
    loop = [r for r in rows if r["role"] == "loop"]
    require(sum(1 for r in loop[1:] if r["cached_tokens"] == 0) == run["loop_cold_after_1"], f"{run_dir.name}: loop cold count differs from the rows")
    require(BOOT_WORD[run["arm"]] in run["policy_line"], f"{run_dir.name}: boot line {run['policy_line']!r} is not the {run['arm']} arm")
    identical = sum(1 for r in rows if r["text_sha256"] == cold_digest[r["idx"]])
    require(identical == run["digests_identical_to_cold"], f"{run_dir.name}: recorded cold identity {run['digests_identical_to_cold']} != {identical}")
    return run, [r["text_sha256"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "pro-single-day15")
    ap.add_argument("--cell", default="ab-full")
    ap.add_argument("--build", default="build-tip")
    ap.add_argument("--landed-cell", default="gate-landed-retry2")
    args = ap.parse_args()
    raw = args.raw
    cell_dir = raw / args.cell
    _rows, lock, capture = journal(cell_dir)
    cell = cell_dir / "cell"
    summary = load_json(cell / "summary.json")
    verdict_txt = (cell / "VERDICT.txt").read_text().strip()
    require(summary["verdict"] == verdict_txt, "VERDICT.txt and summary.json disagree")
    rig = load_json(cell / "rig.json")
    require(CARD in rig["nvidia_smi"] and POWER in rig["nvidia_smi"], f"rig line is not the 600 W PRO 6000: {rig['nvidia_smi']}")
    require(rig["binary_sha256"] == summary["binary_sha256"], "rig.json and summary.json binary digests differ")
    build = raw / args.build
    recorded = (build / "binary.sha256").read_text().split()[0]
    require(recorded == summary["binary_sha256"], f"the cell ran binary {summary['binary_sha256'][:16]} but the build receipt records {recorded[:16]}")
    require((build / "exit").read_text().strip() == "0" and (build / "dirty.txt").read_text().strip() == "", "build receipt not clean")
    source = (build / "source.txt").read_text().strip()
    lockproof = load_json(cell / "LOCK.json")
    require(lockproof.get("lock", lock.get("lock")) == "/tmp/memra-gpu.lock", f"harness lock proof is not the canonical lock: {lockproof}")
    plan = load_json(cell / "plan.json")["requests"]
    shape = summary["shape"]
    for gate in ("cohort_within_protected_share", "turn1_exceeds_free_share", "every_turn_fits_budget", "two_last_turns_fit_together"):
        require(shape[gate], f"shape gate {gate} did not hold: {shape}")
    cold = load_json(cell / "cold" / "rows.json")
    require(len(cold) == len(plan) and all(not r["cached_tokens"] for r in cold), "the cache-off boot is not a cold replay of the plan")
    cold_digest = {r["idx"]: r["text_sha256"] for r in cold}

    # Schedule: AB-0, BA-0, ..., AB-(n-1), BA-(n-1); A = slru, B = lru; every pair adjacent.
    sched = summary["schedule"]
    n = len(sched) // 4
    require(n >= 5, f"fewer than five pairs per order ({n})")
    expected = []
    for i in range(n):
        for order in ("AB", "BA"):
            for pos, letter in enumerate(order):
                expected.append((f"{order}-{i}", order, pos, "slru" if letter == "A" else "lru"))
    require([(s["pair_id"], s["order"], s["position"], s["arm"]) for s in sched] == expected, "the schedule is not the interleaved AB/BA pairing")
    run_dirs = sorted((cell / "runs").iterdir())
    require(len(run_dirs) == len(sched), f"{len(run_dirs)} run directories for {len(sched)} scheduled runs")
    runs = []
    digests = []
    for d, s in zip(run_dirs, sched):
        run, dg = replay_run(d, plan, cold_digest)
        require((run["pair_id"], run["order"], run["position"], run["arm"]) == (s["pair_id"], s["order"], s["position"], s["arm"]), f"{d.name}: run identity differs from the schedule")
        runs.append(run)
        digests.append(dg)
    # Digest identity across every run and the cold boot, per request.
    identical = all(len({cold_digest[p["idx"]]} | {dg[p["idx"]] for dg in digests}) == 1 for p in plan)
    require(identical == summary["digests_identical"], "recorded digest identity differs from the replay")
    require(identical, "a completion digest differed across runs or from the cold boot: the day is a FAIL")
    # The pairs and the rule.
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
    winner = "WINNER=slru" if slru_all else "WINNER=lru" if lru_all else "INCONCLUSIVE"
    require(summary["winner"] == winner, f"recorded winner {summary['winner']} differs from the replayed {winner}")
    require(verdict_txt.endswith(f"-> {winner}"), "the verdict line does not end with the replayed verdict")
    require(capture["status"] == "executed-not-qualified" and capture["exit_code"] == 0, f"collector status {capture['status']} exit {capture['exit_code']}")
    stats = {}
    for arm in ARMS:
        rs = [r for r in runs if r["arm"] == arm]
        comp = [r["computed_tokens"] for r in rs]
        ret = [r["return_cached_sum"] for r in rs]
        require(len(rs) == 2 * n, f"{arm}: {len(rs)} runs for {n} pairs per order")
        require(statistics.median(comp) == summary["arms"][arm]["computed_tokens"]["median"], f"{arm}: recorded median differs")
        stats[arm] = (len(rs), statistics.median(comp), min(comp), max(comp), statistics.median(ret), max(r["loop_cold_after_1"] for r in rs))
    temps = [t for r in runs for t in (r["gpu_start"]["temperature_c"], r["gpu_end"]["temperature_c"]) if t is not None]
    # The landed binary's twin gate (the day-14 gate on the new default): PASS, plain-LRU boot line,
    # restored == cold on every turn, and the same completion digests day 14 recorded for the
    # segmented fix, so the landed default produced the bytes the segmented program produced.
    landed = raw / args.landed_cell / "cell"
    if landed.is_dir():
        _r, _l, lcap = journal(raw / args.landed_cell)
        ls = load_json(landed / "summary.json")
        require(ls["pass"] and ls["verdict"].endswith("-> PASS"), f"landed twin gate is not green: {ls['verdict']}")
        require("plain-LRU" in ls["boot"]["policy"], f"landed boot line is not the plain-LRU default: {ls['boot']['line']}")
        require(ls["turns_identical_to_cold"] == len(ls["turns"]) == 8, "landed twin gate: a restored turn differs from the cold boot")
        require(lcap["status"] == "executed-not-qualified", f"landed twin gate collector status {lcap['status']}")
        day14 = HERE / "pro-single-day14" / "gate-fix-r3" / "cell" / "summary.json"
        if day14.is_file():
            d14 = [t["text_sha256"] for t in load_json(day14)["turns"]]
            require([t["text_sha256"] for t in ls["turns"]] == d14, "landed twin gate digests differ from day 14's recorded fix digests")
        for battery, needle in (("serve-smoke", "serve-smoke: 0 failed"), ("cache-meter", "cache-meter-gate: 0 failed")):
            d = raw / battery
            if d.is_dir():
                _rows, _lock, bcap = journal(d)
                require(needle in (d / "command.log").read_text(errors="replace"), f"{battery}: `{needle}` not in command.log")
                require(bcap["status"] == "executed-not-qualified", f"{battery}: collector status {bcap['status']}")
    print("DAY15 REPLAY OK")
    print(f"source {source[:9]} binary {summary['binary_sha256'][:16]}: {verdict_txt}")
    for pid, a, b, ra, rb in pairs:
        better = "slru" if a < b else "lru" if b < a else "tie"
        print(f"  pair {pid}: slru {a} vs lru {b} computed tokens (diff {a - b:+d}, {better}); return cached slru {ra} lru {rb}")
    for arm, (N, med, lo, hi, rmed, cold_max) in stats.items():
        print(f"{arm}: N={N} computed median {med:.0f} (min {lo}, max {hi}); return cached median {rmed:.0f}; loop cold after turn 1 max {cold_max}")
    print(f"digests identical across {len(runs)} runs and the cache-off boot on all {len(plan)} requests")
    print(f"rig: {rig['nvidia_smi']}; temperature {min(temps)}..{max(temps)} C at run boundaries; power limit {runs[0]['gpu_start']['power_limit_w']} W")
    if landed.is_dir():
        print(f"landed twin gate ({args.landed_cell}): {ls['verdict']}; boot {ls['boot']['policy']}; digests identical to day 14's fix on all 8 turns")
    print(f"replayed rule: {winner}; executed-not-qualified; no cross-box timing compared.")


if __name__ == "__main__":
    main()
