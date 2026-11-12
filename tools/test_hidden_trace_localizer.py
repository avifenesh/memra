#!/usr/bin/env python3
"""Fault-injection proof for the first-divergence localizer (CPU only).

For each pack with a tiny fixture: two clean reference runs must agree bit for bit,
a x1.0 injection must too, and a x1.5 injection into the largest dense weight of the middle
trunk layer must be named by `hidden_trace_diff.py --mode exact` at exactly that layer.

Usage: test_hidden_trace_localizer.py <hidden_trace_engine binary> <out dir> [family ...]
(CPU only: the reference side of hidden_trace_engine; no CUDA device is opened.)
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIFF = os.path.join(HERE, "hidden_trace_diff.py")
DEFAULT_FAMILIES = ["qwen3", "qwen3_moe", "qwen35", "qwen35_moe", "llama_dense", "gemma4_dense",
                    "gemma4_moe", "gemma4_e4b", "glm5_next", "glm_dsa", "hy3"]


def trace(binary, family, out, inject=None):
    env = dict(os.environ, MEMRA_HYPER_TRACE=out)
    cmd = [binary, "--tiny", family, "--side", "reference"] + (["--inject", inject] if inject else [])
    run = subprocess.run(cmd, env=env, capture_output=True, text=True)
    if run.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}: {run.stderr.strip()}")
    return run.stdout.strip()


def diff(a, b):
    run = subprocess.run([sys.executable, "-I", DIFF, "--mode", "exact", a, b], capture_output=True, text=True)
    return run.returncode, run.stdout.strip().splitlines()[-1] if run.stdout.strip() else run.stderr.strip()


def main():
    binary, out_dir = sys.argv[1], sys.argv[2]
    families = sys.argv[3:] or DEFAULT_FAMILIES
    os.makedirs(out_dir, exist_ok=True)
    failures = 0
    for family in families:
        p = lambda name: os.path.join(out_dir, f"{family}-{name}.trace")
        try:
            info = trace(binary, family, p("clean-a"))
            trace(binary, family, p("clean-b"))
            layers = int(re.search(r"layers=(\d+)", info).group(1))
            if layers == 0:
                raise RuntimeError("the reference run reported no trunk layers: nothing to localize")
            trace(binary, family, p("noop"), f"{layers // 2}:largest:1.0")
            tensors = {}
            for k in range(layers):
                hit = trace(binary, family, p(f"inject-L{k}"), f"{k}:largest:1.5")
                tensors[k] = re.search(r"injected=(.*)$", hit).group(1)
        except (RuntimeError, AttributeError) as error:
            print(f"FAIL  {family}: {error}")
            failures += 1
            continue
        checks = []
        code, text = diff(p("clean-a"), p("clean-b"))
        checks.append(("determinism", code == 0, text))
        code, text = diff(p("clean-a"), p("noop"))
        checks.append(("noop", code == 0, text))
        for k in range(layers):
            code, text = diff(p("clean-a"), p(f"inject-L{k}"))
            m = re.search(r"FIRST DIVERGENCE at (\S+)@(-?\d+)", text)
            named = m is not None and int(m.group(2)) == k
            checks.append((f"inject L{k} {tensors[k].split('tensor: ')[-1].rstrip(' }')}", code == 1 and named, text))
        ok = all(c[1] for c in checks)
        failures += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {family}: layers={layers}")
        for name, passed, text in checks:
            print(f"      {'ok ' if passed else 'BAD'} {name}: {text}")
    print(f"{len(families) - failures}/{len(families)} families pass")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
