#!/usr/bin/env python3
"""Per-arch line liveness of every repo CUDA source, from the exact nvcc calls build.rs makes.

For each captured nvcc invocation (stage0/buildrs/out-<variant>/nvcc.jsonl), ask nvcc for its
--dryrun plan, run only the preprocessor steps (device pass: -D__CUDA_ARCH__=...; host pass: the
cudafe/host gcc -E), and read the linemarkers. A source line is LIVE for (variant, pass) when the
preprocessor emitted it. Output: liveness.json
  {variant: {"units": [{src, defines, pass_lines: {device: {file: [ranges]}, host: {...}}}]}}
"""
import concurrent.futures as cf
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

ROOT = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
LM = re.compile(r'^# (\d+) "([^"]+)"')


def ranges(lines):
    lines = sorted(set(lines))
    out = []
    for n in lines:
        if out and n == out[-1][1] + 1:
            out[-1][1] = n
        else:
            out.append([n, n])
    return out


def live_lines(pp_text, crate_dir):
    live = {}
    cur = None
    line = 0
    for raw in pp_text.splitlines():
        m = LM.match(raw)
        if m:
            line = int(m.group(1))
            f = m.group(2)
            if f.startswith(crate_dir):
                cur = os.path.relpath(f, crate_dir)
            elif not f.startswith("/") and not f.startswith("<"):
                cur = f
            else:
                cur = None
            continue
        if cur is not None and raw.strip():
            live.setdefault(cur, []).append(line)
        line += 1
    return {f: ranges(v) for f, v in live.items()}


def run_unit(args, crate_dir, workdir):
    # replace output with /dev/null and ask for the plan
    a = list(args)
    if "-o" in a:
        a[a.index("-o") + 1] = "/dev/null"
    src = [x for x in a if x.endswith(".cu")][-1]
    plan = subprocess.run(["nvcc", *a, "--dryrun"], cwd=crate_dir, capture_output=True, text=True)
    steps = [l[3:] for l in plan.stderr.splitlines() if l.startswith("#$ ")]
    env = dict(os.environ)
    result = {"src": src, "defines": [x for x in a if x.startswith("-D")], "passes": {}}
    if plan.returncode != 0:
        result["passes"]["plan"] = {"error": "nvcc --dryrun failed: " + plan.stderr[-2000:]}
        return result
    for st in steps:
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", st)
        if m and " " not in st.split("=")[0]:
            val = m.group(2)
            val = re.sub(r"\$([A-Za-z_][A-Za-z0-9_]*)", lambda mm: env.get(mm.group(1), ""), val)
            env[m.group(1)] = val.strip('"')
            continue
        if not (st.startswith("gcc ") or st.startswith('"$') and " -E " in st):
            continue
        if " -E " not in st:
            continue
        passname = "device" if "-D__CUDA_ARCH__=" in st else "host"
        toks = shlex.split(st)
        if "-o" in toks:
            i = toks.index("-o")
            toks = toks[:i] + toks[i + 2:]
        toks = [os.path.expandvars(t) if "$" in t else t for t in toks]
        pp = subprocess.run(toks, cwd=crate_dir, capture_output=True, text=True, env=env)
        if pp.returncode != 0:
            result["passes"][passname] = {"error": pp.stderr[-2000:]}
            continue
        result["passes"][passname] = live_lines(pp.stdout, crate_dir + "/")
    if "device" not in result["passes"]:
        # every CUDA unit has a device compile; a plan without one gives no liveness, not "nothing live"
        result["passes"]["device"] = {"error": "the nvcc plan has no device preprocessing step"}
    return result


def main():
    crate_dir = os.path.join(os.environ.get("MEMRA_IMPACT_REPO", str(Path(__file__).resolve().parents[2])), "crates/memra-engine")
    out = {}
    jobs = []
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        for variant in sorted(os.listdir(os.path.join(ROOT, "stage0/buildrs"))):
            if not variant.startswith("out-"):
                continue
            log = os.path.join(ROOT, "stage0/buildrs", variant, "nvcc.jsonl")
            if not os.path.exists(log):
                continue
            v = variant[4:]
            out[v] = {"units": []}
            for l in open(log):
                args = json.loads(l)["args"]
                jobs.append((v, ex.submit(run_unit, args, crate_dir, None)))
        for v, fut in jobs:
            out[v]["units"].append(fut.result())
    json.dump(out, open(os.path.join(ROOT, "stage0/liveness.json"), "w"))
    for v, d in out.items():
        errs = sum(1 for u in d["units"] for p in u["passes"].values() if "error" in p)
        print(v, "units", len(d["units"]), "errors", errs)


if __name__ == "__main__":
    main()
