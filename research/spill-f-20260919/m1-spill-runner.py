#!/usr/bin/env python3
"""M1 B3 cell runner: one cache regime, round-robin arms, under the collector (OWED 12).

  run     --arms-lock m1-prereg/b3-arms.lock.json --regime cold|warm|bounded --binary run-gen
          --artifact FILE --proof PRIVATE_PROOF.json --out DIR --rig pro-single|rtx5090
          --lock-fd N [--rounds 10] [--arms a,b,...] [--visit-timeout-s 1200]
          [--bounded-leave-bytes 7000000000 --floor-bytes 6442450944 | --balloon-bytes N]
  reparse DIR      re-derive every visit's parse from its saved raw log; exit 3 on any difference

Registration: M1-PREREG.md section B and m1-prereg/b3-arms.lock.json. Every child writes stdout
and stderr straight into its visit directory (never a pipe); parsing reads only saved files.
Each visit re-checks the proof identity triple before and after, applies the regime with
m1-cache-regime.py, samples with m1-host-sampler.py, and records the child's I/O and CPU rusage from wait4.
The summary applies the registered verdict rule. `--stub-no-lock` exists only for CPU tests
with a stub binary; a real run-gen never runs without the verified canonical lock.
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("battery", ROOT / "tools/tier-battery.py")
CACHE = load("regime", HERE / "m1-cache-regime.py")
SAMPLER = HERE / "m1-host-sampler.py"
SAMPLE = load("sampler", SAMPLER)

PATTERNS = {
    "gate": r"argmax=(\d+)\s+decode argmax=(\d+)\s+logit maxdiff=\S+\s+(MATCH|MISMATCH)",
    "ttft": r"\[ttft\] prompt_tokens=(\d+) prefill_wall_s=([\d.]+)",
    "generated": r"^generated (\d+) tokens in ([\d.]+)s = ([\d.]+) tok/s",
    "tokens": r"^tokens: \[([\d, ]*)\]",
    "placed": r"\[spill\] experts placed: (\d+) pinned .*?, (\d+) mmap'd",
    "pread_enabled": r"\[spill-pread\] enabled: depth=(\d+) buffer_bytes=(\d+) payload_capacity=(\d+)",
    "window": r"spill worker DECODE-WINDOW: reads=(\d+) bytes=(\d+) waits=(\d+) ring_full=(\d+) fallbacks=(\d+)",
    "stages": (r"spill stages DECODE-WINDOW: worker_read_ms=([\d.]+) demand_read_ms=([\d.]+) "
               r"wait_ms=([\d.]+) h2d_submits=(\d+) overread_bytes=(\d+)"),
    "drop": (r"\[spill-pread\] reads=(\d+) bytes=(\d+) errors=(\d+) short_reads=(\d+) fallbacks=(\d+) "
             r"buffer_waits=(\d+) ring_full=(\d+) overread_bytes=(\d+) worker_read_ns=(\d+) "
             r"demand_read_ns=(\d+) wait_ns=(\d+) h2d_submits=(\d+)"),
    "moe_window": r"MoE cache DECODE-WINDOW: (\d+) slots \| hits=(\d+) misses=(\d+) .*?staged ([\d.]+) GB H2D",
}


def parse_log(text):
    """Pure function of the saved raw log (reparse must reproduce it exactly)."""
    out = {}
    for key, pattern in PATTERNS.items():
        m = re.search(pattern, text, re.M)
        out[key] = list(m.groups()) if m else None
    if out["tokens"] is not None:
        body = out["tokens"][0].strip()
        out["tokens"] = [int(x) for x in body.split(",")] if body else []
    out["mmap_fallback_lines"] = text.count("[spill-pread] falling back to mmap")
    out["panics"] = text.count("panicked at")
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proof_view(path):
    proof = json.loads(Path(path).read_text())
    B.require(proof.get("verdict") == "PASS" and proof.get("class") == "nvme-local-direct",
              "M1 proof is not a PASS")
    identity = proof.get("A8_identity") or {}
    B.require("filesystem_id" in identity, "runner needs the PRIVATE proof receipt (raw filesystem id)")
    graph = proof.get("A4_block_graph") or {}
    leaves = [leaf["name"] for leaf in graph.get("leaves", [])]
    top = graph.get("nodes", [{}])[0].get("name")
    B.require(leaves and top, "M1 proof has no block graph")
    return proof, identity, leaves, top


def order_for(arms, round_index):
    return list(arms) if round_index % 2 == 0 else list(reversed(arms))


def arm_env(lock, arm):
    env = dict(os.environ)
    for key in lock["unset_env"]:
        env.pop(key, None)
    env.update(lock["common_env"])
    env.update(lock["prompt"]["env"])
    env["MEMRA_PROMPT_FILE"] = str(ROOT / lock["prompt"]["env"]["MEMRA_PROMPT_FILE"])
    env.update(arm["env"])
    return env


def expected_depth(arm):
    mode = arm["env"].get("MEMRA_SPILL_IO")
    if mode is None:
        return None
    return int(arm["env"].get("MEMRA_SPILL_PREAD_DEPTH", "2"))


def correctness(arm, parsed, oracle_tokens, ngen, overread_per_read):
    problems = []
    if parsed["panics"]:
        problems.append("panic in log")
    if not parsed["gate"] or parsed["gate"][2] != "MATCH":
        problems.append("argmax gate not MATCH")
    if not parsed["generated"] or int(parsed["generated"][0]) != ngen:
        problems.append(f"did not generate {ngen} tokens")
    if parsed["tokens"] is None or (oracle_tokens is not None and parsed["tokens"] != oracle_tokens):
        problems.append("token ids differ from the byte oracle")
    depth = expected_depth(arm)
    if depth is None:
        if parsed["pread_enabled"] is not None:
            problems.append("mmap arm enabled a positioned-read pool")
    else:
        if parsed["pread_enabled"] is None or int(parsed["pread_enabled"][0]) != depth:
            problems.append(f"positioned-read pool missing or depth != {depth}")
        drop = parsed["drop"]
        if drop is None:
            problems.append("no [spill-pread] totals line")
        elif int(drop[2]) or int(drop[3]):
            problems.append(f"read errors={drop[2]} short_reads={drop[3]}")
        if arm["name"].startswith("direct"):
            window = parsed["window"]
            stages = parsed["stages"]
            if window is None or stages is None:
                problems.append("direct arm without window or stage lines")
            else:
                if int(window[4]):
                    problems.append(f"direct arm fell back to mmap {window[4]} times")
                if int(stages[4]) != overread_per_read * int(window[0]):
                    problems.append(f"overread_bytes {stages[4]} != {overread_per_read} x reads {window[0]}")
    return problems


def contamination(ticks, leaves, proc_io):
    """Foreign device bytes over the visit: leaf sector deltas minus the child's own bytes."""
    if len(ticks) < 2 or proc_io is None:
        return None
    first, last = ticks[0]["diskstats"], ticks[-1]["diskstats"]
    rd = sum(last[d]["read_sectors"] - first[d]["read_sectors"] for d in leaves) * 512
    wr = sum(last[d]["write_sectors"] - first[d]["write_sectors"] for d in leaves) * 512
    own = proc_io.get("read_bytes", 0) + proc_io.get("write_bytes", 0)
    device = rd + wr
    foreign = max(0, device - own)
    return {"device_read_bytes": rd, "device_write_bytes": wr, "own_bytes": own,
            "foreign_bytes": foreign, "foreign_share": (foreign / device) if device else 0.0}


def thermal_ok(header, ticks):
    limits = header.get("hwmon_limits_mC", {})
    for t in ticks:
        for key, value in t.get("nvme_temp_mC", {}).items():
            lim = limits.get(f"{key}_max_mC")
            if value is not None and lim and value >= lim:
                return False
    return True


def run_visit(args, lock, arm, round_index, position, visit_dir, identity, leaves, top, env_extra=None):
    visit_dir.mkdir(parents=True)
    rec = {"arm": arm["name"], "round": round_index + 1, "position": position,
           "order": "forward" if round_index % 2 == 0 else "reverse",
           "utc_start": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    before = B.filesystem_identity(Path(args.artifact).parent)
    rec["identity_before_ok"] = before == identity
    B.require(rec["identity_before_ok"], f"identity changed before visit {visit_dir.name}")
    if args.regime in ("cold", "bounded"):
        rec["regime_ok"] = CACHE.cold([args.artifact])
    elif args.regime == "warm":
        rec["regime_ok"] = CACHE.warm([args.artifact])
    resident, pages = CACHE.residency(args.artifact)
    rec["residency_start"] = [resident, pages]
    env = arm_env(lock, arm)
    env.update(env_extra or {})
    raw = visit_dir / "run.log"
    started = time.monotonic_ns()
    with raw.open("xb") as log:
        child = subprocess.Popen([str(args.binary), str(args.artifact)], stdout=log,
                                 stderr=subprocess.STDOUT, env=env, cwd=ROOT)
    with (visit_dir / "sampler.err").open("xb") as sampler_err:
        sampler = subprocess.Popen([sys.executable, str(SAMPLER), "sample", "--devices",
                                    ",".join(leaves + [top]), "--pid", str(child.pid),
                                    "--interval-ms", "250", "--out", str(visit_dir / "host.jsonl")],
                                   stdout=subprocess.DEVNULL, stderr=sampler_err)
    timed_out = False
    deadline = time.monotonic() + args.visit_timeout_s
    while True:
        try:
            info = os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG)
        except ChildProcessError:
            info = None
        if info is not None and info.si_pid == child.pid:
            break
        if time.monotonic() > deadline:
            timed_out = True
            child.kill()
            os.waitid(os.P_PID, child.pid, os.WEXITED | os.WNOWAIT)
            break
        time.sleep(0.05)
    # /proc/<pid>/io is not readable for a zombie; the reaping call returns the same accounting:
    # Linux fills ru_inblock/ru_oublock from the task's read_bytes/write_bytes in 512-byte units.
    _, status, usage = os.wait4(child.pid, 0)
    code = os.waitstatus_to_exitcode(status)
    child.returncode = code
    proc_io = {"read_bytes": usage.ru_inblock * 512, "write_bytes": usage.ru_oublock * 512,
               "utime_s": usage.ru_utime, "stime_s": usage.ru_stime, "maxrss_kB": usage.ru_maxrss,
               "source": "wait4 rusage (ru_inblock, ru_oublock x 512)"}
    rec["wall_ns"] = time.monotonic_ns() - started
    sampler.send_signal(signal.SIGTERM)
    sampler.wait(timeout=10)
    rec.update(exit_code=code, timed_out=timed_out, proc_io=proc_io, raw_log_sha256=sha(raw))
    after = B.filesystem_identity(Path(args.artifact).parent)
    rec["identity_after_ok"] = after == identity
    resident, pages = CACHE.residency(args.artifact)
    rec["residency_end"] = [resident, pages]
    problems = SAMPLE.validate(visit_dir / "host.jsonl", leaves + [top], 500)
    rec["telemetry_ok"] = not problems
    rec["telemetry_problems"] = problems[:10]
    rows = [json.loads(l) for l in (visit_dir / "host.jsonl").read_text().splitlines() if l.strip()]
    header, ticks = rows[0], [r for r in rows[1:] if r.get("kind") == "tick"]
    rec["contamination"] = contamination(ticks, leaves, proc_io)
    rec["thermal_ok"] = thermal_ok(header, ticks)
    rec["parsed"] = parse_log(raw.read_text(errors="replace"))
    (visit_dir / "visit.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def verdicts(visits, arms, baseline, contam_limit=0.02, max_contaminated=2):
    by = {}
    for v in visits:
        by.setdefault(v["arm"], []).append(v)
    contaminated = {a: sum(1 for v in vs if not v.get("clean_timing")) for a, vs in by.items()}
    regime_scored = all(n <= max_contaminated for n in contaminated.values())
    out = {"regime_scored": regime_scored, "contaminated_visits": contaminated, "arms": {}}
    base = {v["round"]: v for v in by.get(baseline, []) if v.get("scored")}
    for arm in arms:
        if arm == baseline:
            continue
        pairs = {"forward": [], "reverse": []}
        for v in by.get(arm, []):
            b = base.get(v["round"])
            if v.get("scored") and b is not None:
                pairs[v["order"]].append(v["tok_s"] / b["tok_s"])
        ratios = pairs["forward"] + pairs["reverse"]
        if not regime_scored:
            verdict = "unscored-regime"
        elif min(len(pairs["forward"]), len(pairs["reverse"])) < 4:
            verdict = "insufficient"
        else:
            med = statistics.median(ratios)
            up = [sum(r > 1 for r in pairs[o]) for o in ("forward", "reverse")]
            down = [sum(r < 1 for r in pairs[o]) for o in ("forward", "reverse")]
            if med >= 1.05 and min(up) >= 4:
                verdict = "winner"
            elif med <= 0.95 and min(down) >= 4:
                verdict = "loser"
            else:
                verdict = "flat"
        out["arms"][arm] = {"verdict": verdict, "n_pairs": len(ratios),
                            "median_ratio": statistics.median(ratios) if ratios else None,
                            "ratios_forward": pairs["forward"], "ratios_reverse": pairs["reverse"]}
    return out


def run(args):
    lock = json.loads(Path(args.arms_lock).read_text())
    arms = [a for a in lock["arms"] if not args.arms or a["name"] in args.arms.split(",")]
    names = [a["name"] for a in arms]
    B.require(lock["baseline"] in names, "baseline arm must be in the run")
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.stub_no_lock:
        B.require(Path(args.binary).name.startswith("m1-stub"), "--stub-no-lock is only for the stub binary")
        lock_proof = {"stub": True}
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-lock-proof.py"), "--fd",
                               str(args.lock_fd), "--lock", B.LOCKS[args.rig]],
                              pass_fds=(args.lock_fd,), capture_output=True, text=True)
        B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
        lock_proof = json.loads(proc.stdout)
    proof, identity, leaves, top = proof_view(args.proof)
    B.require(B.filesystem_identity(Path(args.artifact).parent) == identity,
              "artifact is not on the proven filesystem")
    identity_rec = {"runner_sha256": sha(__file__), "cache_regime_sha256": sha(HERE / "m1-cache-regime.py"),
                    "sampler_sha256": sha(SAMPLER), "arms_lock_sha256": sha(args.arms_lock),
                    "binary_sha256": sha(args.binary), "artifact_bytes": os.path.getsize(args.artifact),
                    "proof_sha256": sha(args.proof), "proof_tool_sha256": proof.get("tool_sha256"),
                    "regime": args.regime, "rounds": args.rounds, "arms": names, "lock": lock_proof,
                    "leaves": leaves, "top": top, "qualified": False}
    if not args.stub_no_lock:
        identity_rec["artifact_sha256"] = sha(args.artifact)
        B.require(identity_rec["artifact_sha256"] == lock["artifact"]["sha256"], "staged artifact hash mismatch")
    (args.out / "identity.json").write_text(json.dumps(identity_rec, indent=1) + "\n")
    balloon = None
    if args.regime == "bounded":
        nbytes = args.balloon_bytes
        if nbytes is None:
            half_bank = lock["artifact"]["expert_bank_bytes"] // 2
            B.require(args.floor_bytes <= args.bounded_leave_bytes < half_bank,
                      "bounded regime needs floor <= leave < half the expert bank")
            nbytes = CACHE.meminfo_kb("MemAvailable") * 1024 - args.bounded_leave_bytes
        balloon_log = (args.out / "balloon.log").open("xb")
        balloon = subprocess.Popen([sys.executable, str(HERE / "m1-cache-regime.py"), "balloon",
                                    "--bytes", str(nbytes), "--floor-bytes", str(args.floor_bytes)],
                                   stdout=balloon_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 120
        while "LOCKED" not in (args.out / "balloon.log").read_text():
            B.require(balloon.poll() is None and time.monotonic() < deadline,
                      "balloon refused or did not lock; see balloon.log")
            time.sleep(0.1)
    visits = []
    oracle = json.loads(Path(args.oracle_tokens).read_text()) if args.oracle_tokens else None
    ngen = int(lock["common_env"]["MEMRA_NGEN"])
    overread = int(lock["artifact"].get("overread_bytes_per_read", 4096))
    try:
        for r in range(args.rounds):
            for position, name in enumerate(order_for(names, r)):
                arm = next(a for a in arms if a["name"] == name)
                vdir = args.out / f"r{r + 1:02d}-{position + 1}-{name}"
                v = run_visit(args, lock, arm, r, position, vdir, identity, leaves, top)
                if name == lock["byte_oracle"] and oracle is None and v["parsed"]["tokens"]:
                    oracle = v["parsed"]["tokens"]
                v["dir"] = vdir.name
                visits.append(v)
                print(f"M1-VISIT {vdir.name} exit={v['exit_code']} tok_s="
                      f"{v['parsed']['generated'][2] if v['parsed']['generated'] else None}", flush=True)
    finally:
        if balloon is not None:
            balloon.send_signal(signal.SIGTERM)
            balloon.wait(timeout=60)
    # Correctness needs the oracle, which may appear after the first visits; judge every visit
    # only once the whole regime has run.
    if oracle is not None:
        (args.out / "oracle-tokens.json").write_text(json.dumps(oracle) + "\n")
    for v in visits:
        arm = next(a for a in arms if a["name"] == v["arm"])
        v["correctness_problems"] = correctness(arm, v["parsed"], oracle, ngen, overread)
        if oracle is None:
            v["correctness_problems"].append("no byte oracle tokens in this run")
        c = v["contamination"]
        v["clean_timing"] = bool(v["telemetry_ok"] and v["thermal_ok"] and v["identity_after_ok"]
                                 and v.get("regime_ok", True) and c is not None
                                 and c["foreign_share"] <= args.contamination_limit)
        v["scored"] = bool(v["clean_timing"] and not v["correctness_problems"]
                           and v["exit_code"] == 0 and not v["timed_out"])
        v["tok_s"] = float(v["parsed"]["generated"][2]) if v["parsed"]["generated"] else None
        (args.out / v["dir"] / "visit.json").write_text(json.dumps(v, indent=1) + "\n")
    refused = {n for n in names if any(v["correctness_problems"] for v in visits if v["arm"] == n)}
    summary = verdicts([v for v in visits if v["arm"] not in refused], [n for n in names if n not in refused],
                       lock["baseline"])
    summary.update(refused_arms=sorted(refused), regime=args.regime, visits=len(visits), qualified=False,
                   identity_sha256=sha(args.out / "identity.json"))
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("M1-SUMMARY " + json.dumps({k: summary[k] for k in ("regime", "visits", "regime_scored", "refused_arms")}))
    for arm, s in summary["arms"].items():
        print(f"M1-VERDICT regime={args.regime} arm={arm} vs {lock['baseline']}: {s['verdict']} "
              f"median_ratio={s['median_ratio']} pairs={s['n_pairs']}")
    return 0


def reparse(directory):
    bad = 0
    for vj in sorted(Path(directory).glob("*/visit.json")):
        v = json.loads(vj.read_text())
        raw = vj.parent / "run.log"
        if sha(raw) != v["raw_log_sha256"] or parse_log(raw.read_text(errors="replace")) != v["parsed"]:
            print(f"REPARSE MISMATCH {vj.parent.name}")
            bad += 1
    print(f"M1-REPARSE {'PASS' if not bad else 'FAIL'} mismatches={bad}")
    return 0 if not bad else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arms-lock", required=True)
    r.add_argument("--regime", choices=["cold", "warm", "bounded"], required=True)
    r.add_argument("--binary", required=True)
    r.add_argument("--artifact", required=True)
    r.add_argument("--proof", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--rig", choices=["pro-single", "rtx5090"], default="pro-single")
    r.add_argument("--lock-fd", type=int)
    r.add_argument("--rounds", type=int, default=10)
    r.add_argument("--arms")
    r.add_argument("--visit-timeout-s", type=float, default=1200)
    r.add_argument("--bounded-leave-bytes", type=int, default=7_000_000_000)
    r.add_argument("--floor-bytes", type=int, default=6 << 30)
    r.add_argument("--balloon-bytes", type=int)
    r.add_argument("--oracle-tokens", help="byte-oracle token ids when the oracle arm is not in the run")
    r.add_argument("--contamination-limit", type=float, default=0.02)
    r.add_argument("--stub-no-lock", action="store_true")
    p = sub.add_parser("reparse")
    p.add_argument("dir")
    args = ap.parse_args(argv)
    if args.cmd == "reparse":
        return reparse(args.dir)
    B.require(args.stub_no_lock or args.lock_fd is not None, "--lock-fd (inherited canonical lock) required")
    B.require(args.rounds >= 10 or args.stub_no_lock, "registered protocol is 10 rounds")
    B.require(args.contamination_limit == 0.02 or args.stub_no_lock, "the registered co-tenancy limit is 2%")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
