#!/usr/bin/env python3
"""Box-scrub for banked 3way-decision payloads.

Redacts the serve-time `system_fingerprint` (it echoes the live build commit id) and the
per-request `id` from every banked response JSON, in place, per box-scrub policy (the
spec-battery-20260830 precedent). Boot nonces are KEPT: they are the arm-identity receipt
(A/B arm-identity law) and are per-boot random values, not credentials.

Usage: scrub_bank.py <dir> [<dir> ...]
"""
import json
import sys
from pathlib import Path

REDACT = {"system_fingerprint": "<redacted: build fingerprint>", "id": "<redacted>"}


def scrub_obj(o):
    hits = 0
    if isinstance(o, dict):
        for k, v in list(o.items()):
            if k in REDACT and isinstance(v, str):
                o[k] = REDACT[k]
                hits += 1
            else:
                hits += scrub_obj(v)
    elif isinstance(o, list):
        for v in o:
            hits += scrub_obj(v)
    return hits


def main():
    files = 0
    total = 0
    for d in sys.argv[1:]:
        p = Path(d)
        for f in sorted(p.rglob("*.json")) if p.is_dir() else [p]:
            try:
                o = json.load(open(f))
            except Exception:  # noqa: BLE001
                continue
            n = scrub_obj(o)
            if n:
                json.dump(o, open(f, "w"), indent=1)
                files += 1
                total += n
    print(f"[scrub] files_rewritten={files} fields_redacted={total}")


if __name__ == "__main__":
    main()
