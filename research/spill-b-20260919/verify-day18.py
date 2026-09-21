#!/usr/bin/env python3
"""Offline replay of the day-18 receipts on both cards (memra#602: the prompt-end seed on the GDN prime grid).

NOT CUDA execution, NOT a timing comparison, NOT serving qualification. The replay re-derives every statement
DAY18.md makes from the recorded rows (never trusting a VERDICT.txt): the collector journal and lock proof of
every cell (the canonical lock of its rig), the card, the binary bound to its build receipt, the twin gate's
verdict clauses re-derived from the per-turn rows (V5 identity and V6 grid included), the restore gate's
identity and grid flags re-derived from the kept texts and lines, the cross-day identity of the base cell's
twelve digests with day 17's cell B and of the fix's cold digests with day 17's cold digests, the #379 gate's
two failing cells on the fix and its ALL GREEN on base, and the evict-reclaim attribution (the 71-token busy
peer's seed refused on the fix, published on base).

usage: verify-day18.py [--local research/spill-b-20260919/rtx5090-day18] [--target research/spill-b-20260919/pro-single-day18]
                       [--day17 research/spill-b-20260919/rtx5090-day17]
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID, FLOOR, MIN = 32, 16, 64
CARDS = {"local": ("NVIDIA GeForce RTX 5090 Laptop GPU", "/tmp/memra-5090.lock", "rtx5090"),
         "target": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "/tmp/memra-gpu.lock", "pro-single")}
DAY17_TURN10 = ("22f023976ebc22fd5bfc9a699ff63056fb08186f1faec8429537e154a5e07f85",
                "4415b7e361fc6f6ba2492adef1f27cfb00d0105e4588177470cecc4214b35284")


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def capture_len(p):
    if p % GRID == 0:
        return p
    b = p // GRID * GRID
    while b >= GRID and p - b < FLOOR:
        b -= GRID
    return b if b >= MIN else None


def collector(cell_dir, expect_status, card):
    name, lock, rig = CARDS[card]
    rows = [json.loads(ln) for ln in (cell_dir / "CELL.jsonl").read_text().splitlines() if ln.strip()]
    require(rows, f"{cell_dir}: empty CELL.jsonl")
    lk = load(cell_dir / "lock.json")
    require(lk.get("lock") == lock and lk.get("rig") == rig and lk.get("owner") == "collector",
            f"{cell_dir}: not the inherited canonical {rig} lock: {lk}")
    cap = load(cell_dir / "command.capture.json")
    require(cap["status"] == expect_status, f"{cell_dir}: collector status {cap['status']!r}, expected {expect_status!r}")
    proof = load(cell_dir / "cell" / "LOCK.json")
    require(lock in json.dumps(proof), f"{cell_dir}: cell lock proof is not {lock}")
    summary = load(cell_dir / "cell" / "summary.json")
    rig_str = summary.get("rig") or summary["plan"]["rig"]
    require(name in rig_str, f"{cell_dir}: rig {rig_str!r} is not the {name}")
    return summary


def binary_sha(build_dir):
    return (build_dir / "binary.sha256").read_text().split()[0]


def twin(summary, sha, expect_pass, expect_identity, expect_grid_ok):
    require((summary.get("binary_sha256")) == sha, "twin gate: binary is not the build receipt's")
    turns = summary["turns"]
    v5 = [t["text_identical_to_cold"] for t in turns]
    rows = summary["capture_law"]["rows"]
    v6 = []
    for r in rows:
        v6.append(r["tokens"] is not None and r["expected"] is not None and r["tokens"] == r["expected"] and r["tokens"] % GRID == 0)
    require(sum(v5) == expect_identity, f"twin gate: identity {sum(v5)}/{len(v5)}, expected {expect_identity}")
    require(sum(v6) == expect_grid_ok, f"twin gate: grid_ok {sum(v6)}/{len(v6)}, expected {expect_grid_ok}")
    for t in turns:
        exp = capture_len(t["prompt_tokens"])
        require(t["expected_capture_tokens"] == exp, f"turn {t['turn']}: expected capture {t['expected_capture_tokens']} != {exp}")
        if t["turn"] > 1:
            require(t["cached_tokens"] == t["prev_published_tokens"], f"turn {t['turn']}: V1 row")
    v = summary["verdict"]
    require((" -> PASS" in v) == expect_pass and f"identity_ok={sum(v5)}/{len(v5)}" in v and f"grid_ok={sum(v6)}/{len(v6)}" in v,
            f"twin gate: verdict line disagrees with the rows: {v}")
    return turns


def restore(summary, sha, expect_identical, expect_grid_ok, target_len):
    require(summary["plan"]["binary_sha256"] == sha, "restore gate: binary is not the build receipt's")
    require(summary["plan"]["target_tokens"] == target_len, f"restore gate: target {summary['plan']['target_tokens']} != {target_len}")
    cold = summary["cold"]
    ident = grid = 0
    for r in summary["points"]:
        same = r["hit"]["text"] == cold["text"]
        require(same == r["identical_to_cold"], f"point {r['point']}: identity flag disagrees with the texts")
        ident += same
        exp = capture_len(r["point"])
        ok = (r["published"] == exp and r["restored"] == r["published"] and r["hit_line"] == r["published"]
              and r["published"] % GRID == 0 and r["hit_calls"] >= 1 and r["off_grid_calls"] == 0 and not r["refused"])
        require(ok == r["grid_ok"], f"point {r['point']}: grid flag disagrees with the rows")
        grid += ok
    require(ident == expect_identical and grid == expect_grid_ok, f"restore gate: identical={ident} grid_ok={grid}, expected {expect_identical}/{expect_grid_ok}")
    require(f"identical={ident}/" in summary["verdict"] and f"grid_ok={grid}/" in summary["verdict"], "restore gate: verdict line disagrees")
    return cold


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, default=HERE / "rtx5090-day18")
    ap.add_argument("--target", type=Path, default=HERE / "pro-single-day18")
    ap.add_argument("--day17", type=Path, default=HERE / "rtx5090-day17")
    a = ap.parse_args()
    L, T = a.local, a.target

    # builds
    lfix, lbase = binary_sha(L / "build-fix"), binary_sha(L / "build-base")
    tfix, tbase = binary_sha(T / "build-fix"), binary_sha(T / "build-base")
    require((L / "build-fix" / "source.txt").read_text().strip() == (T / "build-fix" / "source.txt").read_text().strip() == "03beaa0b95085b6f51429c95d36c02d81260d18c", "fix source")
    require((L / "build-base" / "source.txt").read_text().strip() == (T / "build-base" / "source.txt").read_text().strip() == "1b354be594acc92d9dc28d6421a93538cecde202", "base source")
    for d in (L / "build-fix", L / "build-base", T / "build-fix", T / "build-base"):
        require((d / "exit").read_text().strip() == "0", f"{d}: build exit")

    # local twin gate: fix and base
    fix = twin(collector(L / "gate-fix-day16-shape", "executed-not-qualified", "local"), lfix, True, 12, 31)
    base = twin(collector(L / "gate-base-day16-shape", "failed", "local"), lbase, False, 11, 0)
    d17 = load(a.day17 / "gate-b-day16-lru-shape" / "cell" / "summary.json")["turns"]
    require(all(x["text_sha256"] == y["text_sha256"] for x, y in zip(d17, base)), "base digests != day 17 cell B digests")
    require(all(x["cold_text_sha256"] == y["cold_text_sha256"] for x, y in zip(d17, fix)), "fix cold digests != day 17 cold digests")
    require(all(x["cold_text_sha256"] == y["cold_text_sha256"] for x, y in zip(base, fix)), "base and fix cold digests differ")
    b10 = next(t for t in base if t["turn"] == 10)
    require((b10["text_sha256"], b10["cold_text_sha256"]) == DAY17_TURN10 and not b10["text_identical_to_cold"], "base turn 10 is not the day-17 flip")
    require(all(t["off_grid_calls"] == 0 for t in fix) and sum(t["off_grid_calls"] for t in base) == 11, "off-grid call counts")

    # local restore gate: turn 10 both arms, turn 12 both arms
    cb = restore(collector(L / "restore-base-t10", "failed", "local"), lbase, 3, 2, 12350)
    cf = restore(collector(L / "restore-fix-t10", "executed-not-qualified", "local"), lfix, 5, 5, 12350)
    require(cb["text"] == cf["text"] == "_\n", "the 12,350 cold completion is not the day-17 one")
    restore(collector(L / "restore-base", "failed", "local"), lbase, 5, 2, 12650)
    restore(collector(L / "restore-fix", "executed-not-qualified", "local"), lfix, 5, 5, 12650)

    # target card: fix on both gates
    twin(collector(T / "gate-fix-day16-shape", "executed-not-qualified", "target"), tfix, True, 12, 31)
    restore(collector(T / "restore-fix-t10", "executed-not-qualified", "target"), tfix, 5, 5, 12350)
    restore(collector(T / "restore-fix", "executed-not-qualified", "target"), tfix, 5, 5, 12650)
    for cell, status in (("serve-smoke", "executed-not-qualified"), ("cache-meter", "executed-not-qualified")):
        cap = load(T / cell / "command.capture.json")
        require(cap["status"] == status, f"{cell}: {cap['status']}")
    require("serve-smoke: 0 failed" in (T / "serve-smoke" / "command.log").read_text(), "serve-smoke")
    require("cache-meter-gate: 0 failed" in (T / "cache-meter" / "command.log").read_text(), "cache-meter")

    # evict-reclaim attribution: base publishes the 71-token busy-peer seed, the fix refuses it
    er_fix = load(T / "evict-reclaim" / "cell" / "summary.json")
    er_base = load(T / "evict-reclaim-base" / "cell" / "summary.json")
    require(not er_fix["pass"] and er_base["pass"], "evict-reclaim: expected fix FAIL (V3) and base PASS")
    require(er_fix["assertions"]["V3_driver_free_moved"] is False and all(v for k, v in er_fix["assertions"].items() if k != "V3_driver_free_moved"), "evict-reclaim fix: only V3 red")
    require(er_fix["calibration"]["e0_busy_peer_entry_bytes"] == 0 and er_base["calibration"]["e0_busy_peer_entry_bytes"] > 0, "busy peer seed: expected 0 on the fix, published on base")
    logs_fix = "".join((T / "evict-reclaim" / "cell" / b / "server.log").read_text() for b in ("calibration", "measured"))
    require("seed REFUSED (grid): prompt 71 tokens" in logs_fix, "fix: the 71-token refusal line")
    require("insert (seed): 71 tokens" in "".join((T / "evict-reclaim-base" / "cell" / b / "server.log").read_text() for b in ("calibration", "measured")), "base: the 71-token seed")
    require(er_fix["binary_sha256"] == tfix and er_base["binary_sha256"] == tbase, "evict-reclaim binaries")
    require("aa6cc3291b981646" in er_fix["verdict"] and "aa6cc3291b981646" in er_base["verdict"], "evict-reclaim identity digest")

    # lane A's gate: the moved equalities red, the restored ones green
    kv1 = (T / "kv-host-fix" / "command.log").read_text()
    kv2 = (T / "kv-host-fix-2" / "command.log").read_text()
    require("GATE: kv-host-tenant-reclaim (fix arm) FAIL (2 assertions)" in kv1 and kv1.count("  FAIL:") == 2, "kv-host first run")
    require("GATE: kv-host-tenant-reclaim (fix arm) PASS" in kv2 and "  FAIL:" not in kv2, "kv-host rerun")
    require("insert (spec-boundary): 83 tokens" in (T / "kv-host-fix-2" / "cell" / "server.log").read_text(), "kv-host entries are spec publications")

    # the #379 gate, local
    hf = (L / "hitgate-fix" / "gate.log").read_text()
    hb = (L / "hitgate-base" / "gate.log").read_text()
    require("SPEC-ON-CACHE-HIT GATE: 2 FAILURE(S) (qwen)" in hf and hf.count("  FAIL:") == 2
            and "FAIL: r3 spec-on text != spec-off text (identity law)" in hf and "FAIL: g2 spec-on text != spec-off text (identity law)" in hf, "#379 fix")
    require("SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)" in hb and "  FAIL:" not in hb, "#379 base")
    for n in ("r3", "g2"):
        on = load(L / "hitgate-fix" / "qwen" / f"qwen-on-{n}.json")["usage"]
        off = load(L / "hitgate-fix" / "qwen" / f"qwen-off-{n}.json")["usage"]
        require(on["prompt_tokens_details"]["cached_tokens"] == 106 and off["prompt_tokens_details"]["cached_tokens"] == capture_len(106) == 64, f"#379 {n} cached counts")
    require((L / "hitgate-fix" / "binary.sha256").read_text().split()[0] == lfix and (L / "hitgate-base" / "binary.sha256").read_text().split()[0] == lbase, "#379 binaries")

    print("DAY18 REPLAY OK")


if __name__ == "__main__":
    main()
