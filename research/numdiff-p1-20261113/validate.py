#!/usr/bin/env python3
"""Run numdiff over the verified ground-truth cases and score it.

Input: cases.json (verified cases from the numdiff-groundtruth workflow).
base-vs-head: `numdiff rev` with its own unit selection at the case's arch; the change-level
verdict decides positives, the named kernels decide negatives and give the cause. kernel-vs-kernel: `numdiff twins` at the commit.

Scores
- negatives (a GPU gate saw different bits), judged on the case's named kernels:
    UNSOUND      the change is IDENTICAL, a named kernel is IDENTICAL with no host change, or a
                 twin pair is IDENTICAL-STREAM
    UNCOVERED    numdiff says it does not cover the change (KERNELS-IDENTICAL, UNCOVERED, or a
                 named kernel IDENTICAL next to a host change)
    CAUGHT-CAUSE flagged, with the recorded cause among the classes
    CAUGHT       flagged, cause not named
- positives (a GPU gate proved equal bits):
    PROVEN          IDENTICAL change / IDENTICAL-STREAM twin
    PROVEN-KERNELS  every kernel identical, host code changed (not covered)
    MISSED-PROOF    CHANGED (conservative)
- NO-DATA: nothing compared (name not found, file not compiled for the arch)
Usage: validate.py cases.json out.json
"""
import json
import os
import sys
import time
from pathlib import Path

WT = Path.home() / ".cache/engineagent/wt"
sys.path.insert(0, str(WT / "tools"))
os.environ.setdefault("NUMDIFF_WORK", str(Path.home() / ".cache/engineagent/numdiff/work"))
import numdiff as nd  # noqa: E402

nd.WORK = Path(os.environ["NUMDIFF_WORK"])
REPO = Path.home() / "projects/memra"
CAUSE = {"fma-contraction": {"fma-contraction"}, "reduction-order": {"reduction", "same-fp-ops"},
         "tensor-core": {"tensor-core"}, "build-flag": {"build-flag"}, "approximation": {"approximation"},
         "precision": {"precision"}}


def run_case(c):
    arch = c["arch"].replace("sm_", "")
    files = [f.split("crates/memra-engine/", 1)[-1] for f in c["cu_files"]]
    if c["mode"] == "base-vs-head":
        class A:
            pass
        a = A()
        # rev's own selection: the touched units (every unit when a header or build input moved)
        a.repo, a.base, a.head, a.arch, a.files = str(REPO), c["commit_before"], c["commit_after"], arch, None
        rep = nd.rev_command(a)
        rows = [r for f in rep["files"].values() for r in f.get("kernels", [])]
        by_name = {}
        for r in rows:
            by_name.setdefault(r["kernel"], []).append(r)
            if r.get("to"):
                by_name.setdefault(r["to"], []).append(r)
        picked = []
        for k in c["kernels"]:
            hits = nd.find_kernels(k, by_name)
            picked += [r for h in hits for r in by_name[h]] or [{"kernel": k, "verdict": "NOT-FOUND"}]
        notes = [f"{s}: {f['note']}" for s, f in rep["files"].items() if f.get("note")]
        return {"change": rep["verdict"], "rows": picked, "notes": notes, "all": rep["summary"],
                "host_changed": rep["host_changed"], "launch_mentions": rep.get("launch_mentions", {}),
                "uncovered": rep.get("uncovered", []), "one_sided": rep.get("one_sided", []),
                "nvcc": rep["nvcc"]["version"]}
    if len(files) == 1:
        files = files * 2
    rep = nd.twins_command(REPO, c["commit_before"], arch, f"{files[0]}:{c['kernels'][0]}",
                           f"{files[1]}:{c['kernels'][1]}")
    return {"rows": rep["rows"], "notes": [rep["note"]] if rep.get("note") else []}


def score(c, res):
    rows = [r for r in res.get("rows", []) if r["verdict"] != "NOT-FOUND"]
    neg = c["kind"].startswith("negative")
    want = CAUSE.get(c["cause_class"], set())
    if c["mode"] == "base-vs-head":
        change = res.get("change")
        if change is None:
            return "NO-DATA"
        # negatives are judged on the named rows: a named kernel called IDENTICAL while the GPU
        # saw it move is unsound unless host code changed (then the launch, not the kernel, can
        # carry the drift, and numdiff says so)
        if neg:
            if change == "IDENTICAL":
                return "UNSOUND"
            if any(r["verdict"] == "IDENTICAL" for r in rows):
                return "UNCOVERED" if res.get("host_changed") else "UNSOUND"
            if change in ("KERNELS-IDENTICAL", "UNCOVERED") or not rows:
                return "UNCOVERED"
            classes = {x for r in rows for x in r.get("classes", [])}
            return "CAUGHT-CAUSE" if want & classes else "CAUGHT"
        return {"IDENTICAL": "PROVEN", "KERNELS-IDENTICAL": "PROVEN-KERNELS"}.get(change, "MISSED-PROOF")
    if not rows:
        return "NO-DATA"
    same = all(r["verdict"] == "IDENTICAL-STREAM" for r in rows)
    classes = {x for r in rows for x in r.get("classes", [])}
    if neg:
        if any(r["verdict"] == "IDENTICAL-STREAM" for r in rows):
            return "UNSOUND"
        return "CAUGHT-CAUSE" if want & classes else "CAUGHT"
    return "PROVEN" if same else "MISSED-PROOF"


def size_ratio(row):
    n = row.get("instructions")
    if isinstance(n, list) and min(n):
        return round(max(n) / min(n), 2)
    return 1.0


def main():
    cases = json.loads(Path(sys.argv[1]).read_text())
    only = set(sys.argv[3:])
    out = []
    for c in cases:
        if only and c["id"] not in only:
            continue
        # the data disk is shared with the GPU lanes: stop (results so far kept) past 82%
        st = os.statvfs(str(Path.home() / ".cache"))
        used = 1 - st.f_bavail / st.f_blocks
        if used > 0.82:
            print(f"STOPPED: ~/.cache at {used:.0%} (> 82%) before {c['id']}", flush=True)
            break
        t = time.time()
        try:
            res = run_case(c)
        except Exception as e:  # keep going; record the failure
            res = {"rows": [], "notes": [f"error: {e}"[:1500]]}
        s = score(c, res)
        rec = {"id": c["id"], "kind": c["kind"], "mode": c["mode"], "cause": c["cause_class"], "score": s,
               "seconds": round(time.time() - t, 1), **res}
        if c["mode"] == "kernel-vs-kernel" and res.get("rows"):
            rec["size_ratio"] = size_ratio(res["rows"][0])
        out.append(rec)
        head = res.get("change", "")
        print(f"{s:14s} {c['kind']:18s} {c['id']} {head} ({rec['seconds']}s"
              + (f", size x{rec['size_ratio']}" if "size_ratio" in rec else "") + "): "
              + "; ".join(f"{r['kernel'][:50]} {r['verdict']} {','.join(r.get('classes', []))}" for r in res.get("rows", [])[:4])
              + (f"  [host: {len(res['host_changed'])} files]" if res.get("host_changed") else "")
              + (f"  [{' | '.join(res['notes'])[:300]}]" if res.get("notes") else ""), flush=True)
        Path(sys.argv[2]).write_text(json.dumps(out, indent=1))  # keep partial progress
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
    tally = {}
    for r in out:
        tally.setdefault(r["kind"], {}).setdefault(r["score"], 0)
        tally[r["kind"]][r["score"]] += 1
    print(json.dumps(tally))


if __name__ == "__main__":
    main()
