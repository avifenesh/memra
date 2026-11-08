"""Quiet decode A/B on the RTX 5090 for #1102's sm_120a perf flag (coordinator, 2026-11-08 08:40).

The same binaries the cell tool ran: A = run-gen of the push base (c347e782d2, cells-5090/base-120a),
B = run-gen of the head (173cd71c57, cells-5090/tree-120a). The model is qwen/qwen3.8-27b and the
command is the tool's perf command (MEMRA_NGEN=128 on its fixed prompt, the tool's build environment,
cwd = the head tree). The order is A B A B ..., in pairs, until both arms have N_PER_ARM kept runs or MAX_PER_ARM runs. The caller holds the rig build
lock and the 5090 lease.

Fixed before the first run:
- Drop rule: a run is dropped when the 1-minute load average sampled just before it exceeds LOAD_MAX
  (24 = the rig's thread count). Load is also sampled just after each run and logged.
- Decision: ACCEPT (the flag is noise) when the B median >= the A median, or when |delta| < 1.5% and
  the two-sided Mann-Whitney p >= 0.05. Otherwise INVESTIGATE. With fewer than 8 kept runs in either
  arm the result is INCONCLUSIVE.

usage: decode-ab.py REPO WORK OUT
"""
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

repo, work, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(repo / "tools"))
import cell_evidence as ce  # noqa: E402

N_PER_ARM = 10        # kept runs wanted per arm
MAX_PER_ARM = 24      # runs per arm at most, when the drop rule discards some
LOAD_MAX = 24.0
MODEL = "/data/ai-ml/hf-models/qwen38-27b-nvfp4-mtp/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf"
tree, base = work / "tree-120a", work / "base-120a"
bins = {"A": base / "target/release/run-gen", "B": tree / "target/release/run-gen"}
out.mkdir(parents=True, exist_ok=False)
prompt = out / "prompt.txt"
prompt.write_text("The quick brown fox jumps over the lazy dog. " * 40 + "The quick brown fox jumps over the lazy")
env = {**ce.build_env("120a", tree), "MEMRA_NGEN": ce.PERF_NGEN, "MEMRA_PROMPT_FILE": str(prompt)}
meta = {"bins": {k: {"path": str(v).replace(str(Path.home()), "~"), "sha256": ce.digest(v.read_bytes())}
                 for k, v in bins.items()},
        "commits": {"A": ce.git(base, "rev-parse", "HEAD").decode().strip(),
                    "B": ce.git(tree, "rev-parse", "HEAD").decode().strip()},
        "n_per_arm": N_PER_ARM, "max_per_arm": MAX_PER_ARM, "load_max": LOAD_MAX, "runs": []}


def load1():
    return float(Path("/proc/loadavg").read_text().split()[0])


def kept_count(arm):
    return sum(1 for r in meta["runs"] if r["arm"] == arm and not r["dropped"])


i = 0
while (min(kept_count("A"), kept_count("B")) < N_PER_ARM) and i < 2 * MAX_PER_ARM:
    arm = "AB"[i % 2]
    name = f"{i + 1:02d}-{arm}.log"
    before = load1()
    t0 = time.time()
    with open(out / name, "wb") as log:
        rc = subprocess.run([str(bins[arm]), MODEL], cwd=tree, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    after = load1()
    try:
        tps = ce.decode_tps((out / name).read_text(errors="replace"))
    except ce.EvidenceError:
        tps = None
    run = {"seq": i + 1, "arm": arm, "log": name, "rc": rc, "tps": tps, "load_before": before, "load_after": after,
           "seconds": round(time.time() - t0, 1), "dropped": before > LOAD_MAX or tps is None}
    meta["runs"].append(run)
    print(json.dumps(run), flush=True)
    i += 1


def mann_whitney_p(a, b):
    """Two-sided normal-approximation Mann-Whitney U p-value with tie correction."""
    allv = sorted((v, k) for k, vs in (("a", a), ("b", b)) for v in vs)
    ranks, i = {}, 0
    vals = [v for v, _ in allv]
    rank_of = []
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        r = (i + j) / 2 + 1
        rank_of.extend([r] * (j - i + 1))
        i = j + 1
    ra = sum(r for r, (_, k) in zip(rank_of, allv) if k == "a")
    n1, n2 = len(a), len(b)
    u = ra - n1 * (n1 + 1) / 2
    mu = n1 * n2 / 2
    n = n1 + n2
    ties = {}
    for v in vals:
        ties[v] = ties.get(v, 0) + 1
    tie = sum(t ** 3 - t for t in ties.values())
    sigma = math.sqrt(n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1))))
    if sigma == 0:
        return 1.0
    z = (abs(u - mu) - 0.5) / sigma
    return math.erfc(max(z, 0) / math.sqrt(2))


kept = {k: [r["tps"] for r in meta["runs"] if r["arm"] == k and not r["dropped"]] for k in "AB"}
summary = {"kept": {k: len(v) for k, v in kept.items()},
           "dropped": [r["seq"] for r in meta["runs"] if r["dropped"]]}
if min(len(v) for v in kept.values()) < 8:
    summary["decision"] = "INCONCLUSIVE"
else:
    ma, mb = statistics.median(kept["A"]), statistics.median(kept["B"])
    delta = 100 * (mb / ma - 1)
    p = mann_whitney_p(kept["A"], kept["B"])
    summary.update({"median_A": ma, "median_B": mb, "delta_pct": round(delta, 2), "mann_whitney_p": round(p, 4),
                    "mean_A": round(statistics.mean(kept["A"]), 3), "mean_B": round(statistics.mean(kept["B"]), 3)})
    summary["decision"] = "ACCEPT" if (mb >= ma or (abs(delta) < 1.5 and p >= 0.05)) else "INVESTIGATE"
meta["summary"] = summary
(out / "decode-ab.json").write_text(json.dumps(meta, indent=1) + "\n")
print("SUMMARY", json.dumps(summary), flush=True)
