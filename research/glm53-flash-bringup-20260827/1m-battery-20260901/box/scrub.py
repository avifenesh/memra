#!/usr/bin/env python3
"""Scrub box identity from banked receipts before they enter git.

The box is a rented spot instance; its IP, ssh port, instance id and hostname are FLEET STATE
and belong in darklanes, never in the public engine repo (public-boundary law).

This script deliberately contains NO literal host or port. Hardcoding them would make the
scrubber itself the leak - the first version did exactly that and the residual-secret sweep
caught its own regex table. Values come from the environment instead:
    SCRUB_HOST, SCRUB_PORT, SCRUB_EXTRA (comma-separated)
usage: SCRUB_HOST=... SCRUB_PORT=... scrub.py <root>
"""
import os
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
PATS = [
    (re.compile(r"ssh2\.vast\.ai"), "<vast-ssh-host>"),
    (re.compile(r"--gui-apikey=\S+"), "--gui-apikey=<redacted>"),
    (re.compile(r"token=[0-9a-f]{32,}"), "token=<redacted>"),
    (re.compile(r"NotebookApp\.token=\S+"), "NotebookApp.token=<redacted>"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "<ip>"),          # any bare IPv4
]
if os.environ.get("SCRUB_HOST"):
    PATS.append((re.compile(re.escape(os.environ["SCRUB_HOST"])), "<box-b-ip>"))
if os.environ.get("SCRUB_PORT"):
    PATS.append((re.compile(r"\b" + re.escape(os.environ["SCRUB_PORT"]) + r"\b"), "<box-b-ssh-port>"))
for e in (os.environ.get("SCRUB_EXTRA") or "").split(","):
    if e.strip():
        PATS.append((re.compile(re.escape(e.strip())), "<redacted>"))

SKIP = {"scrub.py"}
n_files = n_hits = 0
for p in sorted(root.rglob("*")):
    if not p.is_file() or p.name in SKIP:
        continue
    try:
        t = p.read_text(encoding="utf-8", errors="strict")
    except Exception:
        continue          # binary receipt, nothing to scrub
    o, hits = t, 0
    for rx, rep in PATS:
        t, k = rx.subn(rep, t)
        hits += k
    if t != o:
        p.write_text(t, encoding="utf-8")
        n_files += 1
        n_hits += hits
print(f"scrub: rewrote {n_files} files, {n_hits} substitutions under {root}")
