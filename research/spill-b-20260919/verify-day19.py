#!/usr/bin/env python3
"""Offline replay of the day-19 receipts on both cards (memra#602: both capture sites on the GDN prime grid).

NOT CUDA execution, NOT a timing comparison, NOT serving qualification. Re-derives every statement DAY19.md
makes from the recorded rows: collector journals and lock proofs, the binaries bound to their build receipts,
the twin gate's V5/V6 rows and the restore gate's flags on the completed fix (both cards), the fix's cold
digests against day 18's, the #379 gate's summaries on base and fix (the corrected gate: the fix ALL GREEN
with the fc full-cover cell on an on-grid prompt and every identity check green; base red on exactly the
accounting clauses), the evict-reclaim gate's two arms under the V3 that counts pool_retained_bytes, lane A's
gate on the aligned equalities, serve-smoke and cache-meter, and the local-ci correctness stage.

usage: verify-day19.py [--local research/spill-b-20260919/rtx5090-day19] [--target research/spill-b-20260919/pro-single-day19]
                       [--day18 research/spill-b-20260919/rtx5090-day18]
"""
import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
GRID, FLOOR, MIN = 32, 16, 64
CARDS = {"local": ("NVIDIA GeForce RTX 5090 Laptop GPU", "/tmp/memra-5090.lock", "rtx5090"),
         "target": ("NVIDIA RTX PRO 6000 Blackwell Server Edition", "/tmp/memra-gpu.lock", "pro-single")}
FIX_SOURCE = "49d79b946cbc4e6ce2ba2e181b078d16098f9159"


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
    require(lk.get("lock") == lock and lk.get("rig") == rig, f"{cell_dir}: not the canonical {rig} lock: {lk}")
    cap = load(cell_dir / "command.capture.json")
    require(cap["status"] == expect_status, f"{cell_dir}: collector status {cap['status']!r}, expected {expect_status!r}")
    return cap


def twin(cell_dir, card, sha):
    collector(cell_dir, "executed-not-qualified", card)
    s = load(cell_dir / "cell" / "summary.json")
    require(CARDS[card][0] in s["rig"] and s["binary_sha256"] == sha, f"{cell_dir}: rig or binary")
    turns = s["turns"]
    require(all(t["text_identical_to_cold"] for t in turns) and len(turns) == 12, f"{cell_dir}: identity")
    rows = s["capture_law"]["rows"]
    require(len(rows) == 31 and all(r["tokens"] == r["expected"] and r["tokens"] % GRID == 0 for r in rows), f"{cell_dir}: grid rows")
    require(all(t["off_grid_calls"] == 0 for t in turns), f"{cell_dir}: off-grid calls")
    for t in turns:
        require(t["expected_capture_tokens"] == capture_len(t["prompt_tokens"]), f"{cell_dir}: capture_len turn {t['turn']}")
    require("identity_ok=12/12 grid_ok=31/31 grid=32 off_grid_calls=0" in s["verdict"] and s["verdict"].endswith("-> PASS"), f"{cell_dir}: verdict")
    return turns


def restore(cell_dir, card, sha):
    collector(cell_dir, "executed-not-qualified", card)
    s = load(cell_dir / "cell" / "summary.json")
    require(s["plan"]["binary_sha256"] == sha and s["plan"]["target_tokens"] == 12350, f"{cell_dir}: binary or target")
    cold = s["cold"]
    for r in s["points"]:
        require(r["hit"]["text"] == cold["text"] == "_\n", f"{cell_dir}: point {r['point']} text")
        require(r["published"] == capture_len(r["point"]) == r["restored"] == r["hit_line"] and r["off_grid_calls"] == 0, f"{cell_dir}: point {r['point']} grid")
    require("identical=5/5 grid_ok=5/5" in s["verdict"] and s["verdict"].endswith("-> PASS"), f"{cell_dir}: verdict")


def hitgate(log_path, expect_green, expect_fail_names=()):
    text = log_path.read_text()
    fails = re.findall(r"^  FAIL: (.*)$", text, re.M)
    if expect_green:
        require("SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)" in text and not fails, f"{log_path}: expected ALL GREEN, got {fails}")
    else:
        require("FAILURE(S) (qwen)" in text, f"{log_path}: expected failures")
        for n in expect_fail_names:
            require(any(n in f for f in fails), f"{log_path}: expected a failure matching {n!r}")
    require("on-grid prompt: 128 tokens" in text, f"{log_path}: the fc cell's on-grid prompt")
    require("  ok: fc on-grid prompt (tokens a multiple of 32)" in text, f"{log_path}: fc geometry")
    for n in ("r1", "r2", "r3", "g1", "g2"):
        require(f"  ok: {n} spec==plain byte identity" in text, f"{log_path}: identity {n}")
    return fails


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--local", type=Path, default=HERE / "rtx5090-day19")
    ap.add_argument("--target", type=Path, default=HERE / "pro-single-day19")
    ap.add_argument("--day18", type=Path, default=HERE / "rtx5090-day18")
    a = ap.parse_args()
    L, T = a.local, a.target

    # builds: the fix at 49d79b946 on both cards; base is day 18's origin/main 1b354be59 binary on both
    lfix = (L / "build-fix" / "binary.sha256").read_text().split()[0]
    tfix = (T / "build-fix" / "binary.sha256").read_text().split()[0]
    for d in (L / "build-fix", T / "build-fix"):
        require((d / "source.txt").read_text().strip() == FIX_SOURCE and (d / "exit").read_text().strip() == "0", f"{d}: fix build")
    lbase = (a.day18 / "build-base" / "binary.sha256").read_text().split()[0]
    tbase = (T / "base-binary.sha256").read_text().split()[0]
    require((T / "base-source.txt").read_text().strip() == "1b354be594acc92d9dc28d6421a93538cecde202", "target base source")

    # the twin and restore gates on the completed fix, both cards
    fix = twin(L / "gate-fix-day16-shape", "local", lfix)
    d18 = load(a.day18 / "gate-fix-day16-shape" / "cell" / "summary.json")["turns"]
    require(all(x["cold_text_sha256"] == y["cold_text_sha256"] for x, y in zip(d18, fix)), "fix cold digests != day 18 cold digests")
    require(all(x["text_sha256"] == y["text_sha256"] for x, y in zip(d18, fix)), "fix digests != day 18 fix digests")
    restore(L / "restore-fix-t10", "local", lfix)
    twin(T / "gate-fix-day16-shape", "target", tfix)
    restore(T / "restore-fix-t10", "target", tfix)

    # the #379 gate on the corrected gate source: fix ALL GREEN, base red on the accounting only
    hitgate(L / "hitgate-fix-2" / "gate.log", True)
    bfails = hitgate(L / "hitgate-base-2" / "gate.log", False,
                     ("s7 sampled hit restores the whole published entry", "g2 restores g1's published entry",
                      "g3's restored prefix == turn 2's republished boundary"))
    require(not any("identity" in f or "bytes ==" in f or "reproduces" in f for f in bfails), f"base: an identity clause failed: {bfails}")
    require((L / "hitgate-fix-2" / "binary.sha256").read_text().split()[0] == lfix and (L / "hitgate-base-2" / "binary.sha256").read_text().split()[0] == lbase, "#379 binaries")
    # the first day-19 run (the site list this lane added and removed the same day): one failure on the fix, that site only
    first = hitgate(L / "hitgate-fix" / "gate.log", False, ("boundary site restore-suffix-feed never fired",))
    require(len(first) == 1, f"first fix run: expected exactly the site failure, got {first}")

    # target card: serve-smoke, cache-meter, lane A's gate, the evict-reclaim arms
    for cell in ("serve-smoke", "cache-meter"):
        collector(T / cell, "executed-not-qualified", "target")
    require("serve-smoke: 0 failed" in (T / "serve-smoke" / "command.log").read_text(), "serve-smoke")
    require("cache-meter-gate: 0 failed" in (T / "cache-meter" / "command.log").read_text(), "cache-meter")
    kv1 = (T / "kv-host-fix" / "command.log").read_text()
    require("GATE: kv-host-tenant-reclaim (fix arm) FAIL (2 assertions)" in kv1, "kv-host first run: the two whole-prompt equalities")
    kv2 = (T / "kv-host-fix-2" / "command.log").read_text()
    require("GATE: kv-host-tenant-reclaim (fix arm) PASS" in kv2 and "  FAIL:" not in kv2, "kv-host rerun on the aligned equalities")
    require("insert (spec-boundary): 64 tokens" in (T / "kv-host-fix-2" / "cell" / "server.log").read_text(), "kv-host entries are aligned spec publications")
    for arm, sha in (("evict-reclaim-fix", tfix), ("evict-reclaim-base", tbase)):
        collector(T / arm, "executed-not-qualified", "target")
        s = load(T / arm / "cell" / "summary.json")
        require(s["pass"] and s["binary_sha256"] == sha and s["assertions"]["V3_driver_free_moved"], f"{arm}: PASS under the corrected V3")
        m = re.search(r"driver_free_delta_bytes=(\d+) trim_released_bytes=(\d+) pool_retained_bytes=(\d+)", s["verdict"])
        e1 = int(re.search(r"entry_bytes=(\d+)", s["verdict"]).group(1))
        require(m and int(m.group(1)) + int(m.group(3)) >= e1 - 2 * 1024 * 1024, f"{arm}: V3 arithmetic")
    er_fix = load(T / "evict-reclaim-fix" / "cell" / "summary.json")
    require(er_fix["calibration"]["e0_busy_peer_entry_bytes"] == 0, "fix: the 71-token busy peer publishes nothing")

    # the local-ci correctness stage
    lci = (L / "local-ci" / "local-ci.log").read_text()
    require((L / "local-ci.exit").read_text().strip() == "0", "local-ci exit")
    require("SPEC-ON-CACHE-HIT GATE: ALL GREEN (qwen)" in lci, "local-ci: the hit gate inside the battery")
    print("DAY19 REPLAY OK")


if __name__ == "__main__":
    main()
