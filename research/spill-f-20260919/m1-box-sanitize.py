#!/usr/bin/env python3
"""Replace rented-box identifiers in mirrored receipts; every change goes to EXPORT-MANIFEST.json.

No arguments: the BOX27 form, run from the box27 directory (originals copied to the private
directory first; the recorded 2026-09-25 run).
With arguments (BOX36 on): copy a raw private mirror into a new lane directory, then replace each
`--token` by its `--repl` in the copy. The raw mirror stays the private original.
  m1-box-sanitize.py --src PRIVATE_MIRROR --dst LANE_DIR --token V.<id> --repl V.<volume-id>
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def box27():
    P = Path("/home/avifenesh/.local/share/memra-lane-f-private/box27/originals")
    TOKEN, REPL = b"V.52617309", b"V.<volume-id>"
    m = json.loads(Path("EXPORT-MANIFEST.json").read_text())
    n = 0
    for f in sorted(Path(".").rglob("*")):
        if not f.is_file():
            continue
        data = f.read_bytes()
        if TOKEN not in data:
            continue
        dest = P / f
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        f.write_bytes(data.replace(TOKEN, REPL))
        m["exported"].append({"path": str(f), "original_sha256": hashlib.sha256(data).hexdigest(),
                              "export_sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                              "change": "rented volume id replaced by V.<volume-id>"})
        n += 1
    Path("EXPORT-MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    print(f"{n} sanitized, {len(m['exported'])} total")


def export(src, dst, pairs):
    src, dst = Path(src), Path(dst)
    if dst.exists():
        raise SystemExit(f"REFUSED: {dst} exists")
    shutil.copytree(src, dst)
    changes, files = [], 0
    for f in sorted(dst.rglob("*")):
        if not f.is_file():
            continue
        files += 1
        data = f.read_bytes()
        new = data
        for token, repl in pairs:
            new = new.replace(token, repl)
        if new != data:
            f.write_bytes(new)
            changes.append({"path": str(f.relative_to(dst)), "original_sha256": hashlib.sha256(data).hexdigest(),
                            "export_sha256": hashlib.sha256(new).hexdigest(),
                            "change": "rented-box identifier replaced by its placeholder"})
    manifest = {"source": "raw private mirror (kept outside git)", "files": files,
                "placeholders": [r.decode() for _, r in pairs], "exported": changes}
    (dst / "EXPORT-MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(f"{len(changes)} of {files} files sanitized into {dst}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--src")
    ap.add_argument("--dst")
    ap.add_argument("--token", action="append", default=[])
    ap.add_argument("--repl", action="append", default=[])
    a = ap.parse_args()
    if not a.src:
        return box27()
    if not a.dst or len(a.token) != len(a.repl) or not a.token:
        raise SystemExit("REFUSED: --src needs --dst and one --repl per --token")
    export(a.src, a.dst, [(t.encode(), r.encode()) for t, r in zip(a.token, a.repl)])


if __name__ == "__main__":
    main()
