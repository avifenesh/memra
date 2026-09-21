#!/usr/bin/env python3
"""Offline replay of the day-17 receipts on the local RTX 5090 (memra#523 thread, the day-16 digest failure).

NOT CUDA execution, NOT a timing comparison, NOT serving qualification. The replay re-derives every
statement DAY17.md makes from the recorded rows (never trusting a VERDICT.txt): the collector journal and
lock proof of every cell (the canonical `/tmp/memra-5090.lock`, rig `rtx5090`), the card, the binary bound
to its build receipt at the lane tip, the twin gate's verdict lines and identity columns re-derived from the
per-turn rows, the probe's identity flags and first-differing characters re-derived from the kept texts,
the server's own `[primeseg]` call receipts (cold calls on the 32-token GDN grid, the restored suffix call
at the restored entry's end), the cross-binary identity of cell B's twelve loop digests with day 16's lru
run, and the restore-point and sequential-scan cells' rows.

usage: verify-day17.py [--raw research/spill-b-20260919/rtx5090-day17] [--day16 research/spill-b-20260919/rtx5090-day16]
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARD = "NVIDIA GeForce RTX 5090 Laptop GPU"
LOCK, RIG = "/tmp/memra-5090.lock", "rtx5090"
GRID = 32
RE_SEG = re.compile(r"\[primeseg\] call start=(\d+) take=(\d+) grid_off=(\d+)")
DAY16_TURN10 = ("22f023976ebc22fd5bfc9a699ff63056fb08186f1faec8429537e154a5e07f85",
                "4415b7e361fc6f6ba2492adef1f27cfb00d0105e4588177470cecc4214b35284")


def require(value, message):
    if not value:
        raise ValueError(message)


def load_json(path):
    require(path.is_file(), f"missing {path}")
    return json.loads(path.read_text())


def collector(cell_dir, expect_status):
    rows = [json.loads(ln) for ln in (cell_dir / "CELL.jsonl").read_text().splitlines() if ln.strip()]
    require(rows, f"{cell_dir}: empty CELL.jsonl")
    lock = load_json(cell_dir / "lock.json")
    require(lock.get("lock") == LOCK and lock.get("rig") == RIG and lock.get("owner") == "collector",
            f"{cell_dir}: not the inherited canonical {RIG} lock: {lock}")
    cap = load_json(cell_dir / "command.capture.json")
    require(cap["status"] == expect_status, f"{cell_dir}: collector status {cap['status']!r}, expected {expect_status!r}")
    proof = load_json(cell_dir / "cell" / "LOCK.json")
    require(proof.get("lock") == LOCK or proof.get("path") == LOCK or LOCK in json.dumps(proof), f"{cell_dir}: cell lock proof is not {LOCK}: {proof}")
    return cap


def rig_and_binary(summary, binsha, cell_dir):
    rig = summary.get("rig") or summary["plan"]["rig"]
    require(CARD in rig, f"{cell_dir}: rig {rig!r} is not the {CARD}")
    sha = summary.get("binary_sha256") or summary["plan"]["binary_sha256"]
    require(sha == binsha, f"{cell_dir}: binary {sha[:16]} is not the build receipt's {binsha[:16]}")


def segs(lines):
    return [(int(m.group(1)), int(m.group(2)), int(m.group(3))) for ln in lines for m in [RE_SEG.search(ln)] if m]


def replay_gate(raw, name, binsha, expect_identity):
    cell_dir = raw / name
    collector(cell_dir, "executed-not-qualified")
    s = load_json(cell_dir / "cell" / "summary.json")
    rig_and_binary(s, binsha, cell_dir)
    verdict = (cell_dir / "cell" / "VERDICT.txt").read_text().strip()
    require(verdict == s["verdict"], f"{name}: VERDICT.txt differs from summary")
    require("policy plain-LRU" in s["boot"]["line"], f"{name}: boot line is not plain-LRU: {s['boot']['line']}")
    turns = s["turns"]
    cached_ok = sum(1 for t in turns[1:] if t["cached_tokens"] is not None and t["cached_tokens"] >= t["prev_prompt_tokens"])
    cold_after_1 = sum(1 for t in turns[1:] if (t["cached_tokens"] or 0) == 0)
    require(f"cached_ok={cached_ok}/{len(turns) - 1}" in verdict and f"cold_turns_after_1={cold_after_1}" in verdict,
            f"{name}: V1 row counts do not re-derive: cached_ok={cached_ok} cold_after_1={cold_after_1} vs {verdict}")
    require(verdict.endswith("-> PASS") and s["pass"], f"{name}: expected the mechanics verdict PASS: {verdict}")
    identical = [t["turn"] for t in turns if t["text_sha256"] == t["cold_text_sha256"]]
    require(all(t["text_identical_to_cold"] == (t["text_sha256"] == t["cold_text_sha256"]) for t in turns), f"{name}: identity flags do not re-derive")
    require(len(identical) == s["turns_identical_to_cold"], f"{name}: turns_identical_to_cold {s['turns_identical_to_cold']} vs {len(identical)}")
    divergent = [t for t in turns if t["turn"] not in identical]
    require([t["turn"] for t in divergent] == expect_identity["divergent_turns"],
            f"{name}: divergent turns {[t['turn'] for t in divergent]} vs expected {expect_identity['divergent_turns']}")
    for t in divergent:
        require((t["text_sha256"], t["cold_text_sha256"]) == DAY16_TURN10, f"{name}: turn {t['turn']} digests are not day 16's ({t['text_sha256'][:16]}, {t['cold_text_sha256'][:16]})")
        require(t["cached_tokens"] == 12200 and t["prompt_tokens"] == 12350, f"{name}: the divergent turn is not 12,200 restored + 150")
        lines = t["window"]["lines"]
        require(any("reclaim spared the prompt's prefix entry (12200 tokens" in ln for ln in lines) or not any("reclaim-on-defer" in ln for ln in lines),
                f"{name}: a reclaim ran at turn 10 without sparing the 12200 entry")
        require(any("hit: 12200 of 12350" in ln for ln in lines) and any("source lease released after restore fence (entry 12200" in ln for ln in lines),
                f"{name}: turn 10 did not restore 12200 under its lease")
    reclaims = sum(1 for t in turns for ln in t["window"]["lines"] if "reclaim-on-defer" in ln)
    return {"verdict": verdict, "identical": f"{len(identical)}/{len(turns)}", "reclaims": reclaims, "turns": turns}


def replay_probe(raw, name, binsha, expect_status):
    cell_dir = raw / name
    collector(cell_dir, expect_status)
    s = load_json(cell_dir / "cell" / "summary.json")
    rig_and_binary(s, binsha, cell_dir)
    vt = (cell_dir / "cell" / "VERDICT.txt").read_text().splitlines()
    require(vt[0] == s["verdict"], f"{name}: VERDICT.txt line 1 differs from summary")
    cold = {c["turn"]: c for c in s["cold"]}
    for c in s["cold"]:
        require(c["cached_tokens"] in (0, None) and hashlib.sha256(c["text"].encode()).hexdigest() == c["text_sha256"], f"{name}: cold turn {c['turn']} row inconsistent")
        if any("TOKENWISE prompt token" in ln for ln in c["lines"]):
            require(not segs(c["lines"]) and any("TOKENWISE prompt token at fed=0 " in ln for ln in c["lines"]), f"{name}: cold turn {c['turn']} tokenwise walk shows a prime-branch call")
        for start, take, off in segs(c["lines"]):
            require(off == start % GRID == 0, f"{name}: cold turn {c['turn']} prime call at {start} is off the {GRID}-token grid (grid_off {off})")
    compared = [t for t in s["turns"] if t["turn"] in cold]
    for t in compared:
        c = cold[t["turn"]]
        require(t["identical_to_cold"] == (t["text"] == c["text"]), f"{name}: turn {t['turn']} identity flag does not re-derive from the texts")
        if t["turn"] > 1:
            calls = segs(t["lines"])
            if any("TOKENWISE prompt token" in ln for ln in t["lines"]):
                require(not calls and any(f"TOKENWISE prompt token at fed={t['cached_tokens']} " in ln for ln in t["lines"]),
                        f"{name}: turn {t['turn']} walked tokenwise but shows a prime-branch call or does not start at the restored end")
            else:
                require(len(calls) == 1 and calls[0][0] == t["cached_tokens"] == t["prev_prompt_tokens"] and calls[0][1] == t["prompt_tokens"] - t["cached_tokens"]
                        and calls[0][2] == calls[0][0] % GRID,
                        f"{name}: turn {t['turn']} suffix prime receipt {calls} is not one call at the restored end")
    identical = [t for t in compared if t["identical_to_cold"]]
    divergent = [t for t in compared if not t["identical_to_cold"]]
    require(f"identical={len(identical)}/{len(compared)}" in s["verdict"], f"{name}: identical count does not re-derive: {s['verdict']}")
    require(s["verdict"].endswith("-> DIVERGENT" if divergent else "-> IDENTICAL"), f"{name}: verdict word does not match the rows")
    points = s.get("restore_points") or []
    for r in points:
        require(r["grid_off"] == r["point"] % GRID, f"{name}: point {r['point']} grid_off")
        c = cold[s["plan"]["turns"]]
        require(r["identical_to_cold"] == (r["hit"]["text"] == c["text"]), f"{name}: point {r['point']} identity does not re-derive")
        require(r["hit"]["cached_tokens"] == r["point"], f"{name}: point {r['point']} restored {r['hit']['cached_tokens']} tokens, not the entry")
        calls = segs(r["hit"]["lines"])
        require(len(calls) == 1 and calls[0] == (r["point"], len(c["text"]) * 0 + (r["hit"]["prompt_tokens"] - r["point"]), r["point"] % GRID),
                f"{name}: point {r['point']} suffix prime receipt {calls}")
    if points:
        require(s["points_verdict"] and vt[1] == s["points_verdict"], f"{name}: points verdict missing or differs")
    return {"verdict": s["verdict"], "points_verdict": s.get("points_verdict"), "divergent": divergent, "identical": identical, "compared": compared, "cold": cold, "points": points, "turns": s["turns"]}


def s_turns(r):
    return max(r["cold"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=HERE / "rtx5090-day17")
    ap.add_argument("--day16", type=Path, default=HERE / "rtx5090-day16")
    args = ap.parse_args()
    raw = args.raw
    binsha = (raw / "build-tip" / "binary.sha256").read_text().split()[0]
    require(len(binsha) == 64 and (raw / "build-tip" / "exit").read_text().strip() == "0" and (raw / "build-tip" / "dirty.txt").read_text().strip() == "",
            "build receipt: not a clean exit-0 build")
    source = (raw / "build-tip" / "source.txt").read_text().strip()
    lines = []

    a = replay_gate(raw, "gate-a-default", binsha, {"divergent_turns": []})
    lines.append(f"cell A (default shape): {a['verdict']}; identical to cold {a['identical']}; reclaim lines {a['reclaims']}")
    b = replay_gate(raw, "gate-b-day16-lru-shape", binsha, {"divergent_turns": [10]})
    lines.append(f"cell B (day-16 lru shape): {b['verdict']}; identical to cold {b['identical']}; divergent turn 10 carries day 16's digests; reclaim lines {b['reclaims']}")

    # cross-binary identity with day 16's lru run (a different binary, 9466b8912): the twelve loop rows
    d16 = load_json(args.day16 / "ab-smoke" / "cell" / "runs" / "02-AB-0-lru" / "run.json")
    d16_loop = {r["prompt_tokens"]: r["text_sha256"] for r in d16["rows"] if r["role"] == "loop"}
    for t in b["turns"]:
        require(d16_loop[t["prompt_tokens"]] == t["text_sha256"], f"cell B turn {t['turn']}: digest differs from day 16's lru run at the same prompt")
    lines.append("cell B: all 12 loop digests identical to day 16's lru run (binary 4a55ad8e at 9466b8912), including the divergent turn 10")

    c = replay_probe(raw, "probe-c-batched", binsha, "failed")
    require([t["turn"] for t in c["divergent"]] == [10] and c["divergent"][0]["first_diff_char"] == 1
            and c["divergent"][0]["text"] == "_\t\t\"\t\t\"\t" and c["cold"][10]["text"] == "_\n",
            "probe C: the divergent turn is not turn 10 at generated token 2 with the recorded texts")
    t10 = c["divergent"][0]
    lines.append(f"probe C (batched): {c['verdict']}; restored {json.dumps(t10['text'])} ({t10['completion_tokens']} tokens, {t10['finish_reason']}) vs cold {json.dumps(c['cold'][10]['text'])} ({c['cold'][10]['completion_tokens']} tokens, {c['cold'][10]['finish_reason']})")
    cold_calls = segs(c["cold"][10]["lines"])
    chain_calls = segs(t10["lines"])
    lines.append(f"probe C turn 10 prime receipts: cold calls {[(s, t) for s, t, _ in cold_calls]} all grid_off=0; restored call {chain_calls} (12200 % {GRID} = {12200 % GRID})")

    for name, status_ok in (("probe-e-restore-points", None), ("probe-f-gdn-sequential", None), ("probe-d-tokenwise", None)):
        if not (raw / name / "cell" / "summary.json").is_file():
            lines.append(f"{name}: no receipts (cell not run)")
            continue
        cap = load_json(raw / name / "command.capture.json")
        r = replay_probe(raw, name, binsha, cap["status"])
        if r["points"]:
            by = {p["point"]: p for p in r["points"]}
            require(set(by) == {12288, 12320, 12200, 12250, 12300}, f"{name}: unexpected restore points {sorted(by)}")
            require(all(by[p]["identical_to_cold"] for p in (12288, 12320)), f"{name}: an on-grid restore point did not reproduce the cold bytes")
            require(by[12200]["identical_to_cold"], f"{name}: the single-restore 12200 point was expected identical (the chain's 12200 is the one that flips)")
            require(all(not by[p]["identical_to_cold"] and by[p]["hit"]["text"] == "_\t\t\"\t\t\"\t" and by[p]["first_diff_char"] == 1 for p in (12250, 12300)),
                    f"{name}: the off-grid 12250/12300 points were expected to flip to the chain's alternative stream at generated token 2")
            cold_last = segs(r["cold"][s_turns(r)]["lines"])[-1]
            require(segs(by[12288]["hit"]["lines"])[0] == cold_last == (12288, 62, 0), f"{name}: the 12288 hit call is not the cold prime's own last call")
            lines.append(f"{name}: {r['points_verdict']}")
            for p in r["points"]:
                lines.append(f"  point {p['point']} grid_off {p['grid_off']} suffix {p['suffix']}: {'identical' if p['identical_to_cold'] else 'DIVERGENT at char ' + str(p['first_diff_char'])} {json.dumps(p['hit']['text'])}")
        else:
            if name == "probe-d-tokenwise":
                require(len(r["compared"]) == 1 and not r["divergent"] and r["compared"][0]["turn"] == 10 == s_turns(r), f"{name}: expected one compared turn, turn 10, identical")
                t10 = r["compared"][0]
                require(t10["cached_tokens"] == 12200 and t10["text"] == r["cold"][10]["text"] == "_\n", f"{name}: turn 10 is not 12200 restored reproducing the cold bytes")
                for side, row, n in (("cold", r["cold"][10], 12350), ("chain turn 1", [t for t in r["turns"] if t["turn"] == 1][0], 11000), ("chain turn 10", t10, 150)):
                    require(not segs(row["lines"]) and any(f"TOKENWISE x{n} prompt tokens" in ln for ln in row["lines"]),
                            f"{name}: {side} did not walk exactly {n} prompt tokens through decode_step with no prime-branch call")
                lines.append(f"{name}: no prime-branch call in either boot; 12,350 tokenwise receipts cold, 11,000 + 9 x 150 chain; turn 10 restored 12200 == cold")
            if name == "probe-f-gdn-sequential":
                require(not r["divergent"] and len(r["compared"]) == 12, f"{name}: expected 12/12 identical under the sequential scan")
                t10 = [t for t in r["turns"] if t["turn"] == 10][0]
                require(segs(t10["lines"]) == [(12200, 150, 8)] and t10["text"] == r["cold"][10]["text"] == "_\t\t\"\t\t\"\t",
                        f"{name}: turn 10 should keep the chunked arm's call geometry (12200,150,8) and reproduce its own cold bytes, which are the chunked arm's restored stream")
                chunked = {x["turn"]: x["text_sha256"] for x in c["cold"].values()}
                differ = sorted(k for k, v in r["cold"].items() if v["text_sha256"] != chunked[k])
                require(differ == [6, 10], f"{name}: sequential-scan cold digests differ from the chunked arm's at turns {differ}, expected exactly the two near-tie prompts 6 and 10")
                lines.append(f"{name}: sequential-scan COLD digests equal the chunked arm's on 10/12 turns and differ at turns {differ} (11,750 and 12,350: the day-16 slru and lru flip prompts)")
            lines.append(f"{name}: {r['verdict']}")
            for t in r["divergent"]:
                lines.append(f"  turn {t['turn']} DIVERGENT: restored {json.dumps(t['text'])} vs cold {json.dumps(r['cold'][t['turn']]['text'])}")

    print(f"binary {binsha[:16]} built at {source[:9]} (exit 0, clean)")
    for ln in lines:
        print(ln)
    print("DAY17 REPLAY OK: receipts consistent, every statement above re-derived from the rows")


if __name__ == "__main__":
    main()
