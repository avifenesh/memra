#!/usr/bin/env python3
"""Lane C's check for a step status read after a command substitution in the same statement (`echo "$(date) rc=$?"`
reads date's status, always 0), over every shell script under this lane's records. Prints each hit with its file and
line and exits 1 when there is one; `PIPESTATUS` is not affected (it survives a substitution) and is not reported.
usage: rc-scan.py [--live] [root]   (default: this file's directory; --live skips the mirrored receipt directories,
pro-single-* and rtx5090-*, whose scripts are records of what ran and are not edited)
"""
import pathlib
import sys

args = [a for a in sys.argv[1:] if a != "--live"]
live = "--live" in sys.argv
root = pathlib.Path(args[0]) if args else pathlib.Path(__file__).resolve().parent
hits = []
for f in sorted(root.rglob("*.sh")):
    if live and f.relative_to(root).parts[0].startswith(("pro-single-", "rtx5090-")):
        continue
    for n, line in enumerate(f.read_text(errors="replace").splitlines(), 1):
        for statement in line.split(";"):
            at = statement.find("$?")
            if at > 0 and "$(" in statement[:at]:
                hits.append(f"{f.relative_to(root)}:{n}: {statement.strip()[:160]}")
for h in hits:
    print(h)
print(f"rc-scan: {len(hits)} status read(s) after a command substitution in {root.name}{' (live scripts)' if live else ''}")
sys.exit(1 if hits else 0)
