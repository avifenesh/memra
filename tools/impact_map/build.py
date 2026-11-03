#!/usr/bin/env python3
"""Regenerate docs/impact-map/map.json for the current tree (CPU only; no GPU, no kernel build).

    python3 tools/impact_map/build.py [--scip INDEX.scip | --run-scip] [--work DIR]

Steps (each output lands under --work, default ~/.cache/memra-impact-map-build):
  1. SCIP index of the workspace (rust-analyzer, DOCS_RS=1 so no nvcc runs; ~3 min, ~4 GB RSS;
     under the shared build lock at nice 15). --scip reuses an index.
  2. Rust item scan (rust_items.py) and CUDA item scan (cu_items.py) of every tracked source.
  3. build.rs replay per build variant with a logging nvcc stand-in (buildrs_capture.py), then the
     per-arch preprocessor liveness of every CUDA line (cu_liveness.py; nvcc -E only).
  4. The item graph (build_graph.py): SCIP references, trait fan-out, kernel names, FFI, launches.
  5. assemble.py: re-anchor docs/impact-map/annotations.json onto this tree and propagate reach.

The annotations are the agent-made labels (predicate regions, file defaults, entries, dynamic
edges), each with file:line evidence; refresh them with the workflow described in
docs/impact-map/README.md when a subsystem is rewritten. A region whose item moved is re-anchored
by the item's token hash or its evidence quote; one that cannot be re-anchored is dropped, which
makes the map wider, never narrower.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def run(cmd, env, **kw):
    print("+", " ".join(str(c) for c in cmd), file=sys.stderr)
    r = subprocess.run([str(c) for c in cmd], env=env, **kw)
    if r.returncode != 0:
        sys.exit(f"step failed: {cmd[0]} {cmd[1] if len(cmd) > 1 else ''} (exit {r.returncode})")
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--work", type=Path, default=Path.home() / ".cache/memra-impact-map-build")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--scip", type=Path, help="reuse an existing SCIP index of this tree")
    g.add_argument("--run-scip", action="store_true", help="build the SCIP index with rust-analyzer")
    ap.add_argument("--out", type=Path, default=REPO / "docs/impact-map/map.json")
    a = ap.parse_args()
    s0 = a.work / "stage0"
    s0.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MEMRA_IMPACT_WORK=str(a.work), MEMRA_IMPACT_REPO=str(REPO))
    index = s0 / "index.scip"
    if a.run_scip:
        ra = shutil.which("rust-analyzer")
        if ra is None:
            sys.exit("rust-analyzer not found; install the rust-analyzer component or pass --scip")
        lock = Path.home() / ".cache/memra-rig-build.lock"
        scip_env = dict(env, DOCS_RS="1", CARGO_TARGET_DIR=str(a.work / "ra-target"), CARGO_BUILD_JOBS="2")
        run(["flock", "-w", "3600", lock, "nice", "-n", "15", ra, "scip", REPO, "--output", index], scip_env)
    else:
        shutil.copyfile(a.scip, index)
    run([sys.executable, HERE / "scip_decode.py", index, s0 / "scip"], env)
    files = subprocess.run(["git", "-C", REPO, "ls-files", "*.rs"], capture_output=True, text=True).stdout
    (s0 / "rs-files.txt").write_text(files)
    with open(s0 / "items.jsonl", "w") as out:
        run([sys.executable, HERE / "rust_items.py", "--files-from", s0 / "rs-files.txt"], env, stdout=out, cwd=REPO)
    cu = subprocess.run(["git", "-C", REPO / "crates/memra-engine", "ls-files", "cu"], capture_output=True, text=True).stdout.split()
    cu = [f for f in cu if f.endswith((".cu", ".cuh", ".h"))]
    with open(s0 / "cu_items.jsonl", "w") as out:
        run([sys.executable, HERE / "cu_items.py", *cu], env, stdout=out, cwd=REPO / "crates/memra-engine")
    run([sys.executable, HERE / "buildrs_capture.py"], env)
    run([sys.executable, HERE / "cu_liveness.py"], env)
    run([sys.executable, HERE / "build_graph.py"], env)
    run([sys.executable, HERE / "assemble.py", REPO / "docs/impact-map/annotations.json", a.out], env)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
