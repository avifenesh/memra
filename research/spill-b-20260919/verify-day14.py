#!/usr/bin/env python3
"""Offline replay of the day-14 newest-turn-fits receipts (memra#523 items 1 and 3).

NOT CUDA execution, NOT a timing comparison, NOT serving qualification: every cell is N=1 and
`executed-not-qualified` in the collector's vocabulary. The replay re-derives each arm's verdict
from the recorded per-turn rows (never trusting `VERDICT.txt`), checks the collector journal
and lock proof, binds each arm's binary to its build receipt and source ref, confirms the rig
(one RTX PRO 6000 Blackwell at 600 W), confirms the shape gates the gate itself asserted, and
compares the two arms: same artifact, same prompts, same cohort, the base red with its cold
turns quoted from the route line, the fix green, and identical completion digests on the
turns where both arms produced a completion (the numeric program did not move).

usage: verify-day14.py [--raw research/spill-b-20260919/pro-single-day14]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
MIB = 1 << 20
V3_SLACK = 64 * MIB
ARMS = {"main": "build-main", "fix": "build-fix"}
CELLS = {"main": "gate-main-r3", "fix": "gate-fix-r3"}
# Earlier rounds stay on disk as they ran: round 1 asserted consumed == grew with no footprint
# control; round 2 subtracted the calibration turn's consumed bytes (misaligned before turn 1).
FIRST_ROUND = {"main": "gate-main", "fix": "gate-fix"}
SECOND_ROUND = {"main": "gate-main-final", "fix": "gate-fix-final"}
POWER = "600.00 W, 600.00 W"
CARD = "NVIDIA RTX PRO 6000 Blackwell"
RE_ON_LINE = "[prefix-cache] on:"
RE_REFUSED = re.compile(r"\[prefix-cache\] insert refused: entry (\d+) (exceeds budget|cannot fit beside) (\d+)")


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


def replay_verdict(summary):
    turns = summary["turns"]
    v1_rows = [t["cached_tokens"] is not None and t["prev_prompt_tokens"] is not None and t["cached_tokens"] >= t["prev_prompt_tokens"] for t in turns[1:]]

    def publishes(t):
        return any(i["tokens"] == t["prompt_tokens"] for i in t["window"]["inserts"])

    v2_rows = []
    for t in turns:
        w = t["window"]
        clean = not w["refused"] and not w["skipped"]
        if t["turn"] == 1:
            v2_rows.append(clean and publishes(t))
        else:
            hit_ok = len(w["hits"]) == 1 and w["hits"][0]["hit"] >= (t["prev_prompt_tokens"] or 0)
            v2_rows.append(clean and hit_ok and publishes(t))
    v3_rows = []
    for t in turns:
        eff_after = t["metrics_after"]["cuda_driver_free_bytes"] + t["metrics_after"]["cuda_pool_cached_bytes"]
        require(eff_after == t["effective_free_after"], f"turn {t['turn']}: effective free after is not driver free + pool cached")
        err = t["calibration_effective_free_after"] - (eff_after + t["metrics_after"]["prefix_cache_bytes"])
        require(err == t["v3_state_error_bytes"], f"turn {t['turn']}: recorded V3 state error {t['v3_state_error_bytes']} is not cal - (meas + resident) = {err}")
        v3_rows.append(abs(err) <= V3_SLACK)
        if t["window"]["evicts"]:
            derr = t["effective_free_consumed"] - t["footprint_calibration_bytes"] - t["prefix_bytes_grew"]
            require(derr == t["delta_error_bytes"], f"turn {t['turn']}: recorded delta error is not consumed - footprint - grew")
    protected = sum(1 for t in turns for e in t["window"]["evicts"] if e["segment"] == "Protected")
    v = {"V1": all(v1_rows), "V2": all(v2_rows), "V3": bool(v3_rows) and all(v3_rows), "V4": protected >= 1}
    return v, all(v.values())


def check_arm(raw, arm):
    cell_dir = raw / CELLS[arm]
    rows, lock, capture = journal(cell_dir)
    cell = cell_dir / "cell"
    summary = load_json(cell / "summary.json")
    verdict_txt = (cell / "VERDICT.txt").read_text().strip()
    require(summary["verdict"] == verdict_txt, f"{arm}: VERDICT.txt and summary.json disagree")
    rig = load_json(cell / "rig.json")
    require(CARD in rig["nvidia_smi"] and POWER in rig["nvidia_smi"], f"{arm}: rig line is not the 600 W PRO 6000: {rig['nvidia_smi']}")
    require(rig["binary_sha256"] == summary["binary_sha256"], f"{arm}: rig.json and summary.json binary digests differ")
    build = raw / ARMS[arm]
    recorded = (build / "binary.sha256").read_text().split()[0]
    require(recorded == summary["binary_sha256"], f"{arm}: the gate ran binary {summary['binary_sha256'][:16]} but the build receipt records {recorded[:16]}")
    require((build / "exit").read_text().strip() == "0" and (build / "dirty.txt").read_text().strip() == "", f"{arm}: build receipt not clean")
    source = (build / "source.txt").read_text().strip()
    lockproof = load_json(cell / "LOCK.json")
    require(lockproof.get("lock", lock.get("lock")) == "/tmp/memra-gpu.lock", f"{arm}: gate lock proof is not the canonical lock: {lockproof}")
    boot = summary["boot"]
    require("SLRU" in boot["policy"].upper(), f"{arm}: the boot did not report the SLRU default")
    shape = summary["shape"]
    require(shape["cohort_within_protected_share"] and shape["turn1_exceeds_free_share"] and shape["every_turn_fits_budget"], f"{arm}: the shape gates did not hold: {shape}")
    require(len(summary["turns"]) == 8, f"{arm}: expected 8 turns, got {len(summary['turns'])}")
    cal = summary["calibration"]
    require(len(cal["turns"]) == 8 and not any(RE_ON_LINE in ln for ln in cal["boot_lines"]), f"{arm}: the calibration boot is not a cache-off replay of 8 turns")
    for c, m in zip(cal["turns"], summary["turns"]):
        require(c["prompt_ids"] == m["prompt_ids"] and c["prompt_tokens"] == m["prompt_tokens"], f"{arm}: calibration and measured prompts differ on turn {m['turn']}")
        require(c["effective_free_consumed"] == m["footprint_calibration_bytes"], f"{arm}: turn {m['turn']} footprint not taken from the calibration boot")
        cal_eff = c["metrics_after"]["cuda_driver_free_bytes"] + c["metrics_after"]["cuda_pool_cached_bytes"]
        require(cal_eff == m["calibration_effective_free_after"], f"{arm}: turn {m['turn']} calibration effective free not taken from the calibration boot")
        require(not c["metrics_after"]["prefix_cache_bytes"], f"{arm}: the calibration boot held prefix bytes")
        require(m["text_identical_to_cold"] == (m["text_sha256"] == c["text_sha256"]), f"{arm}: cold identity flag disagrees on turn {m['turn']}")
    for t in summary["turns"]:
        require(t["status"] == 200, f"{arm}: turn {t['turn']} was not served")
        require(t["prompt_tokens"] == t["prompt_ids"], f"{arm}: turn {t['turn']} prompt_tokens {t['prompt_tokens']} != prompt_ids {t['prompt_ids']}")
    v, ok = replay_verdict(summary)
    require(ok == summary["pass"], f"{arm}: replayed pass {ok} differs from the recorded {summary['pass']}")
    require(verdict_txt.endswith("-> PASS" if ok else "-> FAIL"), f"{arm}: verdict line does not end with the replayed verdict")
    require(v == {"V1": summary["assertions"]["V1_cached"], "V2": summary["assertions"]["V2_lines"], "V3": summary["assertions"]["V3_effective_free"], "V4": summary["assertions"]["V4_protected_evicted"]}, f"{arm}: replayed assertions {v} differ from the recorded ones")
    # The collector's vocabulary: exit 0 is `executed-not-qualified` (never `completed`, never
    # qualification), exit 1 is `failed`.
    require(capture["status"] == ("executed-not-qualified" if ok else "failed"), f"{arm}: collector status {capture['status']} does not match verdict")
    return {"summary": summary, "verdict": verdict_txt, "source": source, "binary": summary["binary_sha256"], "pass": ok, "assertions": v, "rig": rig["nvidia_smi"],
            "footprint": [t["footprint_calibration_bytes"] for t in summary["turns"]],
            "state_errors": [t["v3_state_error_bytes"] for t in summary["turns"]],
            "cold_identical": summary["turns_identical_to_cold"]}


def check_earlier_round(raw, name, arm):
    """An earlier round's cell stays on disk as it ran: a failed collector cell, never relabelled."""
    cell_dir = raw / name[arm]
    if not cell_dir.is_dir():
        return None
    rows, lock, capture = journal(cell_dir)
    summary = load_json(cell_dir / "cell" / "summary.json")
    require(not summary["pass"] and capture["status"] == "failed", f"{name[arm]}: expected a failed cell")
    return summary


def cold_receipts(summary):
    """Per-request server receipts that say the cache did not engage: the glm5 route line's
    `cold=1`, or the spec-k admission line's `cached=0`."""
    out = []
    for t in summary["turns"][1:]:
        for r in t["window"]["route"]:
            if (r["kind"] == "route" and r["cold"] == 1) or (r["kind"] == "spec-k" and r["cached"] == 0):
                out.append((t["turn"], r["line"]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "pro-single-day14")
    args = ap.parse_args()
    raw = args.raw
    arms = {arm: check_arm(raw, arm) for arm in ARMS}
    base, fix = arms["main"], arms["fix"]
    require(not base["pass"], "the base must be red")
    require(fix["pass"], "the fix must be green")
    require(base["source"] != fix["source"], "the two arms were built from the same source")
    require(base["binary"] != fix["binary"], "the two arms ran the same binary")
    bs, fs = base["summary"], fix["summary"]
    require(bs["boot"]["budget_bytes"] == fs["boot"]["budget_bytes"], "budgets differ between arms")
    require([c["tokens"] for c in bs["cohort"]] == [c["tokens"] for c in fs["cohort"]], "cohorts differ between arms")
    require([t["prompt_ids"] for t in bs["turns"]] == [t["prompt_ids"] for t in fs["turns"]], "prompts differ between arms")
    require(bs["rig"] == fs["rig"], "the two arms did not run on the same card line")
    # Base: every turn after the first is cold, quotable from the route line and from cached_tokens.
    base_cold = [t for t in bs["turns"][1:] if (t["cached_tokens"] or 0) == 0]
    require(len(base_cold) == 7, f"base: expected 7 cold turns after turn 1, got {len(base_cold)}")
    base_routes = cold_receipts(bs)
    require(len(base_routes) == 7, f"base: expected a cold server receipt on each of the 7 later turns, got {len(base_routes)}")
    require(all("restored=0" in ln or "cached=0" in ln for _, ln in base_routes), "base: a cold receipt claims a restore")
    require(base["footprint"] == fix["footprint"], f"the cache-off footprint differs between binaries: {base['footprint']} vs {fix['footprint']}")
    first = {arm: check_earlier_round(raw, FIRST_ROUND, arm) for arm in ARMS}
    if first["main"] and first["fix"]:
        for bt, ft in zip(first["main"]["turns"], first["fix"]["turns"]):
            if ft["v3_credit_error_bytes"] is not None:
                require(ft["v3_credit_error_bytes"] == bt["effective_free_consumed"], f"round 1 turn {ft['turn']}: the fix's V3 error {ft['v3_credit_error_bytes']} is not the base's cache-free consumption {bt['effective_free_consumed']}")
    second = {arm: check_earlier_round(raw, SECOND_ROUND, arm) for arm in ARMS}
    if second["fix"]:
        errs = [t["v3_credit_error_bytes"] for t in second["fix"]["turns"]]
        require(errs[0] == 149094400 and all(e == 0 for e in errs[1:]), f"round 2 fix: expected the turn-1 cohort-phase misalignment only, got {errs}")
        # The state identity already held exactly in round 2's own rows.
        for t, c in zip(second["fix"]["turns"], second["fix"]["calibration"]["turns"]):
            cal_eff = c["metrics_after"]["cuda_driver_free_bytes"] + c["metrics_after"]["cuda_pool_cached_bytes"]
            require(cal_eff == t["effective_free_after"] + t["metrics_after"]["prefix_cache_bytes"], f"round 2 fix turn {t['turn']}: the state identity did not hold")
    # Fix: every turn >= 2 restored the previous turn, a protected eviction happened, no refusal.
    require(all(t["cached_tokens"] >= t["prev_prompt_tokens"] for t in fs["turns"][1:]), "fix: a turn did not restore its predecessor")
    require(not fs["refused_or_skipped_lines"], "fix: a refusal or skip line was printed for the growing tenant")
    # Identity across binaries on every turn where both produced a completion.
    identical = [t["turn"] for bt, t in zip(bs["turns"], fs["turns"]) if bt["text_sha256"] == t["text_sha256"]]
    differing = [t["turn"] for bt, t in zip(bs["turns"], fs["turns"]) if bt["text_sha256"] != t["text_sha256"]]
    cohort_identical = all(bc["first"]["text_sha256"] == fc["first"]["text_sha256"] and bc["second"]["text_sha256"] == fc["second"]["text_sha256"] for bc, fc in zip(bs["cohort"], fs["cohort"]))
    # Batteries on the fix, same card.
    for battery, needle in (("serve-smoke", "serve-smoke: 0 failed"), ("cache-meter", "cache-meter-gate: 0 failed")):
        d = raw / battery
        if d.is_dir():
            rows, _, capture = journal(d)
            log = (d / "command.log").read_text(errors="replace")
            require(needle in log, f"{battery}: `{needle}` not in command.log")
            require(capture["status"] == "executed-not-qualified", f"{battery}: collector status {capture['status']}")
    # Refused first sitting kept as-is.
    refused = raw / "refused-sitting1"
    if refused.is_dir():
        for name in ("gate-main", "gate-fix", "serve-smoke", "cache-meter"):
            log = refused / f"{name}-driver.log"
            require(log.is_file() and log.read_text().startswith("REFUSED:"), f"refused-sitting1/{name}: refusal text missing")
    print("DAY14 REPLAY OK")
    print(f"base {base['source'][:9]} binary {base['binary'][:16]}: {base['verdict']}")
    print(f"fix  {fix['source'][:9]} binary {fix['binary'][:16]}: {fix['verdict']}")
    print(f"base cold turns after turn 1: {len(base_cold)} (server receipts saying so: {len(base_routes)})")
    print(f"identical completion digests across binaries on turns {identical}; differing on turns {differing}; cohort identical: {cohort_identical}")
    print(f"measured completions identical to the cache-off boot's cold completion: base {base['cold_identical']}/8, fix {fix['cold_identical']}/8")
    print(f"cache-off footprint per turn (both binaries): {fix['footprint']}")
    print(f"V3 state error per turn: base {base['state_errors']}, fix {fix['state_errors']} (slack {V3_SLACK})")
    if first["main"] and first["fix"]:
        print("round 1 (V3 as consumed == grew, no footprint control) on disk: base FAIL, fix FAIL with V3 error == base cache-free consumption on every evicting turn")
    if second["fix"]:
        print("round 2 (V3 as per-turn delta minus calibration consumed) on disk: fix FAIL on turn 1 only, 149,094,400 B of cohort-phase misalignment; the state identity held exactly on every turn")
    print(f"rig: {base['rig']}")
    print("N=1 per arm; executed-not-qualified; no timing compared.")


if __name__ == "__main__":
    main()
