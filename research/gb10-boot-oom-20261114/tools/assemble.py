#!/usr/bin/env python3
"""assemble.py SRC_UNIT_A SRC_UNIT_B DEST: copy memra#1316's cell records from the workstation's
fetch directories (one per unit, each a copy of the unit's ~/.cache/memra-1316/runs) into
DEST/unit-a and DEST/unit-b, as text with the units' host names replaced by `unit-a` and `unit-b`
and the home directory by `~`. Marker files (STOP, DONE, reader pids) and cells that were aborted
before a boot are left out.
Then writes DEST/summary.txt: tab_probe.py over the driver-level cells and tab_boots.py over the
server cells."""
import os
import re
import subprocess
import sys

src_a, src_b, dest = sys.argv[1:4]
HERE = os.path.dirname(os.path.abspath(__file__))
SKIP_NAMES = {"STOP", "DONE", "memwatch.stop", "stop", "server.pid"}
SKIP_PREFIX = ("reader.pids", "stop.")


def clean(text, unit_names):
    for host, unit in unit_names:
        text = text.replace(host, unit)
    text = re.sub(r"/home/[a-z0-9_-]+", "~", text)
    # probe_cell.sh's first exit-mode rows carry bc's ".5" for a fraction under a second, which
    # is not JSON; the digit is added here so every row of the record parses.
    text = re.sub(r'("since_exit_s": )(-?)\.', r"\g<1>\g<2>0.", text)
    # Trailing blanks (the 1 Hz memory lines end in one) and blank lines at the end of a file go,
    # so `git diff --check` stays quiet over the record.
    text = re.sub(r"[ \t]+$", "", text, flags=re.M)
    return text.rstrip("\n") + "\n" if text.strip() else text


def copy_unit(src, unit, unit_names):
    n = 0
    for root, dirs, files in os.walk(src):
        dirs[:] = sorted(d for d in dirs if not d.endswith(".aborted"))
        rel = os.path.relpath(root, src)
        for name in sorted(files):
            if name in SKIP_NAMES or name.startswith(SKIP_PREFIX):
                continue
            out_dir = os.path.join(dest, unit, rel)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(root, name), errors="replace") as f:
                text = clean(f.read(), unit_names)
            with open(os.path.join(out_dir, name), "w") as f:
                f.write(text)
            n += 1
    return n


def main():
    names = []
    for src, unit in ((src_a, "unit-a"), (src_b, "unit-b")):
        host_file = os.path.join(src, "HOST")
        if os.path.exists(host_file):
            names.append((open(host_file).read().strip(), unit))
    for src, unit in ((src_a, "unit-a"), (src_b, "unit-b")):
        print(f"{unit}: {copy_unit(src, unit, names)} files")
        host_copy = os.path.join(dest, unit, "HOST")
        if os.path.exists(host_copy):
            os.remove(host_copy)
    lines = []
    for unit in ("unit-a", "unit-b"):
        base = os.path.join(dest, unit)
        cells = sorted(d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d)))
        probe = [os.path.join(base, c) for c in cells
                 if os.path.exists(os.path.join(base, c, "boots.jsonl")) and not os.path.exists(os.path.join(base, c, "env.txt"))]
        boots = [os.path.join(base, c) for c in cells
                 if os.path.exists(os.path.join(base, c, "env.txt")) or os.path.exists(os.path.join(base, c, "cycles.jsonl"))]
        cwd = os.path.dirname(os.path.abspath(dest))
        for tool, runs in (("tab_probe.py", probe), ("tab_boots.py", boots)):
            if runs:
                out = subprocess.run([sys.executable, os.path.join(HERE, tool)] + (["-v"] if tool == "tab_boots.py" else []) +
                                     [os.path.relpath(r, cwd) for r in runs],
                                     capture_output=True, text=True, cwd=cwd).stdout
                lines.append(f"# {unit}: {tool}\n{out}")
    with open(os.path.join(dest, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"summary: {os.path.join(dest, 'summary.txt')}")


if __name__ == "__main__":
    main()
