#!/usr/bin/env python3
"""Fail-before setup for the env_audit retired-name test.

Builds three throwaway trees under ~/.cache/memra-gb10-r5/seam/failbefore/:
  old     FLAGS.md and engine sources at the base commit 9390bd3755 (the seam row is live)
  new     FLAGS.md and sources at the change
and writes regen.rs, build.rs's own generate_env_registry with a main around it.
Every crate other than memra-engine is a symlink into the worktree; memra-engine/src is a real
copy so the old tree carries the old spec.rs read.
"""
import os
import shutil
import subprocess
import sys

WT = os.path.expanduser("~/.cache/memra-gb10-r5/wt-seam")
FB = os.path.expanduser("~/.cache/memra-gb10-r5/seam/failbefore")
BASE = "9390bd3755"


def git_show(path):
    return subprocess.run(["git", "-C", WT, "show", f"{BASE}:{path}"], check=True,
                          capture_output=True).stdout


def tree(name, old):
    root = os.path.join(FB, name)
    if os.path.exists(root):
        shutil.rmtree(root)
    os.makedirs(os.path.join(root, "docs"))
    os.makedirs(os.path.join(root, "out"))
    crates = os.path.join(root, "crates")
    os.makedirs(crates)
    for c in sorted(os.listdir(os.path.join(WT, "crates"))):
        src = os.path.join(WT, "crates", c)
        if not os.path.isdir(src):
            continue
        if c == "memra-engine":
            eng = os.path.join(crates, c)
            os.makedirs(eng)
            shutil.copytree(os.path.join(src, "src"), os.path.join(eng, "src"))
            if old:
                for rel in ("src/spec.rs", "src/env_audit.rs"):
                    with open(os.path.join(eng, rel), "wb") as f:
                        f.write(git_show(f"crates/memra-engine/{rel}"))
        else:
            os.symlink(src, os.path.join(crates, c))
    flags = git_show("docs/FLAGS.md") if old else open(os.path.join(WT, "docs/FLAGS.md"), "rb").read()
    with open(os.path.join(root, "docs/FLAGS.md"), "wb") as f:
        f.write(flags)


def regen():
    lines = open(os.path.join(WT, "crates/memra-engine/build.rs")).read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("fn generate_env_registry("))
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == "}")
    body = "\n".join(lines[start:end + 1])
    src = ("// build.rs's generate_env_registry, verbatim, with a main (fail-before check)\n"
           "#![allow(dead_code)]\nuse std::path::{Path, PathBuf};\n" + body +
           "\nfn main() {\n    let out = PathBuf::from(std::env::args().nth(1).expect(\"out dir\"));\n"
           "    generate_env_registry(&out);\n}\n")
    with open(os.path.join(FB, "regen.rs"), "w") as f:
        f.write(src)


tree("old", True)
tree("new", False)
regen()
print("prepared", FB)
sys.exit(0)
