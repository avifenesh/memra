#!/usr/bin/env python3
"""Measurement ledger: what was measured, on which exact code, and whether it still holds.

A ledger record is one measured cell: (model, family, route, device, test) at a commit, with
its receipt path. It still holds at HEAD when the cell's FINGERPRINT is unchanged: the sha of
(item id, token hash) over every item whose map reach includes (family, route, device). Any
token change on the cell's path moves the fingerprint; a change elsewhere does not.

    python3 tools/impact_map/ledger.py seed --raw RAW.jsonl --out docs/impact-map/ledger.json
    python3 tools/impact_map/ledger.py status [--ref HEAD]      # which records still hold

Seed sources, strongest first: qualification records (research/release-qualification/*/
source.json names the exact commit), receipts that name a commit that resolves in this repo,
then the commit that added the receipt (an upper bound on the measured code; marked
`added-upper-bound`).
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delta  # noqa: E402

LEDGER = "docs/impact-map/ledger.json"
SCHEMA = "memra-impact-ledger-v1"


class TreeScanner:
    """Item hashes of a whole commit, cached per blob so many commits cost little."""

    def __init__(self, repo):
        self.repo = repo
        self.blob_items = {}
        self.proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"], stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE)

    def read(self, oid):
        self.proc.stdin.write((oid + "\n").encode())
        self.proc.stdin.flush()
        header = self.proc.stdout.readline().decode().split()
        size = int(header[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)
        return data.decode("utf-8", "surrogateescape")

    def hashes(self, commit):
        out = subprocess.run(["git", "-C", str(self.repo), "ls-tree", "-r", commit, "--", "crates"],
                             capture_output=True, text=True).stdout
        h = {}
        for line in out.splitlines():
            meta, path = line.split("\t", 1)
            oid = meta.split()[2]
            if not (path.endswith(".rs") or path.startswith(delta.CU_PREFIX + "cu/")):
                continue
            key = (path, oid)
            if key not in self.blob_items:
                self.blob_items[key] = {i: v[0] for i, v in delta.scan(path, self.read(oid)).items()}
            h.update(self.blob_items[key])
        return h


def cell_items(m):
    """(family, route, device) -> item ids whose reach includes it (inverted map)."""
    inv = collections.defaultdict(list)
    set_cells = {}
    for i, it in m.items.items():
        idx = it["r"]
        if idx not in set_cells:
            set_cells[idx] = m.cells_of_set(idx)
        for c in set_cells[idx]:
            inv[c].append(i)
    return inv


def fingerprints(m, hashes, cells, inv=None):
    inv = inv or cell_items(m)
    return {c: delta.fingerprint(hashes, inv.get(c, ())) for c in cells}


def seed(repo, raw_path, out_path, map_path, models_path):
    m = delta.Map(json.loads(Path(map_path).read_text()))
    models = json.loads(Path(models_path).read_text()) if models_path and Path(models_path).exists() else {"families": {}}
    fam_of = {}
    for fam, ms in models.get("families", {}).items():
        for mm in ms:
            fam_of[mm["id"]] = fam
    recs = []
    for line in Path(raw_path).read_text().splitlines():
        r = json.loads(line)
        if not (r["models"] and r["devices"] and r["tests"]):
            continue
        commits = [(c, d) for c, d in r["commits"]]
        conf = "explicit" if commits else "added-upper-bound"
        if not commits and r.get("added"):
            commits = [(r["added"][0], r["added"][1])]
        if "release-qualification" in r["path"]:
            conf = "record"
        routes = [x for x in r["routes"] if x in ("mtp", "dflash", "tp", "pp", "batch", "graph", "vision", "spill")] or ["plain"]
        for model in r["models"]:
            fam = fam_of.get(model) or next((f for f in r["families"]), None)
            for dev in r["devices"]:
                for test in r["tests"]:
                    for route in routes:
                        for c, date in commits:
                            recs.append({"model": model, "family": fam, "route": route, "device": dev, "test": test,
                                         "commit": c, "date": date, "confidence": conf, "source": r["path"]})
    # fingerprint every record at its commit; mark whether it still holds at the map commit
    scanner = TreeScanner(repo)
    head = m.data["source_commit"]
    by_commit = collections.defaultdict(list)
    cells_needed = set()
    for r in recs:
        cell = (r["family"], route_of(r["route"], m), r["device"])
        r["_cell"] = cell if r["family"] in m.vocab["families"] and cell[1] else None
        if r["_cell"]:
            cells_needed.add(r["_cell"])
        by_commit[r["commit"]].append(r)
    head_fp = delta.cell_fingerprints(m, scanner.hashes(head), cells_needed)
    print(f"ledger: {len(recs)} candidate records over {len(by_commit)} commits", file=sys.stderr)
    for n, (c, rs) in enumerate(sorted(by_commit.items())):
        if n % 200 == 0:
            print(f"  {n}/{len(by_commit)} commits", file=sys.stderr)
        cells = {r["_cell"] for r in rs if r["_cell"]}
        fps = delta.cell_fingerprints(m, scanner.hashes(c), cells) if cells else {}
        for r in rs:
            cell = r.pop("_cell")
            r["fingerprint"] = fps.get(cell) if cell else None
            r["holds_at_map_commit"] = (fps.get(cell) == head_fp.get(cell)) if cell else None
    # keep, per (model, route, device, test): the latest measured commit (the "last measured" the
    # delta starts from) and every record that still holds; the full harvest stays reproducible
    # from research/ with tools/impact_map/ledger.py seed
    best = {}
    for r in recs:
        if r["family"] is None:
            continue
        k = (r["model"], r["route"], r["device"], r["test"])
        key = (r.get("holds_at_map_commit") is True, r.get("date") or "", rank(r))
        if k not in best or key > best[k][0]:
            best[k] = (key, r)
    out = {"schema": SCHEMA, "map_commit": head,
           "note": "one row per (model, route, device, test): its latest receipt, preferring one that still holds",
           "records": sorted((r for _, r in best.values()), key=lambda r: (r["family"] or "", r["model"], r["device"], r["route"], r["test"]))}
    Path(out_path).write_text(json.dumps(out, indent=0, sort_keys=True) + "\n")
    held = sum(1 for r in out["records"] if r.get("holds_at_map_commit"))
    print(f"ledger: {len(out['records'])} records, {held} still hold at {head[:12]}")


def rank(r):
    return {"record": 3, "explicit": 2, "added-upper-bound": 1}[r["confidence"]]


def route_of(route, m):
    return route if route in m.vocab["routes"] else None


def status(repo, ref):
    data = delta.load_json_at(repo, ref, LEDGER)
    m = delta.Map(delta.load_json_at(repo, ref, delta.MAP_PATH))
    scanner = TreeScanner(repo)
    recs = [r for r in data["records"] if r.get("fingerprint")]
    now = delta.cell_fingerprints(m, scanner.hashes(ref), {(r["family"], r["route"], r["device"]) for r in recs})
    latest = {}
    for r in recs:
        cell = (r["family"], r["route"], r["device"])
        ok = r["fingerprint"] == now.get(cell)
        k = (r["model"], r["route"], r["device"], r["test"])
        if ok or k not in latest:
            latest[k] = (ok, r)
    for k, (ok, r) in sorted(latest.items()):
        print(f"{'HOLDS ' if ok else 'stale '} {r['model']:<26} {r['route']:<8} {r['device']:<11} {r['test']:<16} {r['commit'][:12]} {r['source']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("seed")
    p.add_argument("--repo", type=Path, default=HERE.parent.parent)
    p.add_argument("--raw", required=True)
    p.add_argument("--map", required=True)
    p.add_argument("--models")
    p.add_argument("--out", required=True)
    p = sub.add_parser("status")
    p.add_argument("--repo", type=Path, default=HERE.parent.parent)
    p.add_argument("--ref", default="HEAD")
    a = ap.parse_args(argv)
    if a.cmd == "seed":
        seed(a.repo, a.raw, a.out, a.map, a.models)
    else:
        status(a.repo, a.ref)
    return 0


if __name__ == "__main__":
    sys.exit(main())
