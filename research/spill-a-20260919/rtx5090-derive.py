#!/usr/bin/env python3
"""DAY68 section 1: derive the owed 5090 halves' scripts from the target sittings' own, by exact replacements only.

usage: rtx5090-derive.py        (run from anywhere in the repo; writes rtx5090-r1/ and rtx5090-l2/ beside this file)
Each source script is read from its sitting's tip tree (`git show <tip>:<path>`). Every replacement must match at
least once in every script it is listed for, or nothing is written. No cell, mode, environment, boot count, order or
bound changes: only the receipt root, the tree path, the rig's lock, the rig's CPU cap and target dir, and the box's
clone, fetch and named-branch checkout (the scratch worktree is created at the tip by half.sh).
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REL = "research/spill-a-20260919"
MODEL_DEFAULT = "${MEMRA_DAY38_MODEL:-/root/artifacts/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf}"
CAP = "systemd-run --user --scope -q -p CPUQuota=1200% -p MemoryMax=20G cargo"
HALVES = {
    "r1": ("15d7ed351", "pro-single-r1", "a-r1", "lane-a-r1", ["build", "gates", "ab"]),
    "l2": ("21984b527", "pro-single-l2", "a-l2", "lane-a-l2", ["build", "gates", "gates-red", "unit-cells", "ab"]),
    # DAY68 section 4: S4's half on its target tip; V's on ccfd26af0 (V's tip a324503df plus the revised pause gate
    # of DAY47 section 3a; the crates identical).
    "s4": ("a0f9968e3", "pro-single-s2", "a-s2", "lane-a-s2-tip",
           ["build", "ab-demote", "ab-promote", "hump", "gates", "hitgate", "unit-cells", "trace"]),
    "v": ("ccfd26af0", "pro-single-v", "a-v", "lane-a-v-tip", ["build", "ab-pause", "gates", "hitgate", "unit-cells"]),
}
LATER = ("s4", "v")


def reps(name, receipts, branch, later=False):
    """(old, new, required) for one script."""
    common = [
        (f"R=/root/spill-receipts/{receipts}\n", "R=${A_OUT:?}\n", True),
        ("export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH\n", "export PATH=/usr/local/cuda/bin:$PATH\n", True),
        ("/tmp/memra-gpu.lock", "/tmp/memra-5090.lock", name != "build"),
        (MODEL_DEFAULT, "${MEMRA_DAY38_MODEL:?}", name not in ("build", "unit-cells")),
    ]
    if name == "build":
        return common + [
            ("[ -d /root/wt-a/.git ] || git clone -q --filter=blob:none https://github.com/avifenesh/memra.git "
             "/root/wt-a >> \"$L\" 2>&1\n", "", not later),
            ("cd /root/wt-a || exit 1\n", "cd \"${A_TREE:?}\" || exit 1\n", True),
            ("git fetch -q origin lane/spill-a-20260919 >> \"$L\" 2>&1\n", "", True),
            (f"git checkout -q -B {branch} \"$1\"", "git checkout -q --detach \"$1\"", True),
            ("nice -n 5 cargo", CAP, True),
            ("cp target/release/memra-server", "cp \"${CARGO_TARGET_DIR:?}/release/memra-server\"", True),
            ("cmp -s target/release/memra-server", "cmp -s \"${CARGO_TARGET_DIR:?}/release/memra-server\"", False),
        ]
    tail = [
        ("cd /root/wt-a || exit 1\n", "cd \"${A_TREE:?}\" || exit 1\n", False),
        ("cd /root/wt-a\n", "cd \"${A_TREE:?}\" || exit 1\n", False),
    ]
    if later and name == "unit-cells":
        # The box ran the prebuilt test binaries through cargo under the hold; here cargo runs under the rig's cap.
        tail.append(("cargo test -p", CAP + " test -p", True))
    return common + tail


def main():
    out_all = {}
    for half, (tip, src, receipts, branch, names) in HALVES.items():
        for name in names:
            text = subprocess.run(["git", "show", f"{tip}:{REL}/{src}/{name}.sh"], capture_output=True, text=True,
                                  check=True, cwd=HERE).stdout
            for old, new, required in reps(name, receipts, branch, half in LATER):
                n = text.count(old)
                if required and n == 0:
                    sys.exit(f"REFUSED: {half}/{name}.sh: replacement did not match: {old!r}")
                text = text.replace(old, new)
            if "cd \"${A_TREE:?}\"" not in text:
                sys.exit(f"REFUSED: {half}/{name}.sh: no tree cd replaced")
            code = "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))
            for banned in ("/root/", "memra-gpu.lock", "nice -n 5", "target/release"):
                if banned in code:
                    sys.exit(f"REFUSED: {half}/{name}.sh still carries {banned!r}")
            lines = text.split("\n", 1)
            head = (f"{lines[0]}\n# Derived for the local RTX 5090 by rtx5090-derive.py (DAY68 section 1) from "
                    f"{src}/{name}.sh at {tip}: exact replacements only.\n")
            out_all[(half, name)] = head + lines[1]
    for (half, name), text in out_all.items():
        d = os.path.join(HERE, f"rtx5090-{half}")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}.sh"), "w") as f:
            f.write(text)
        print(f"wrote rtx5090-{half}/{name}.sh")


if __name__ == "__main__":
    main()
