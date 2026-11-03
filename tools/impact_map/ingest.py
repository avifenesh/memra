#!/usr/bin/env python3
"""Turn annotation-agent output into docs/impact-map/annotations.json: every claim's citations are
checked against the tree (quote must be a substring of the cited line, +-3 lines allowed and
re-pinned), each region is anchored to its innermost stage-0 item (id, start line, token hash)
so later trees can re-anchor it, and adversarial verdicts are applied (a refuted narrowing claim
is widened back to every cell; an unverified narrowing claim is kept and marked).

Usage: ingest.py RESULTS.json [VERDICTS.json] > annotations.json
Env: MEMRA_IMPACT_WORK (stage-0 dir parent), MEMRA_IMPACT_REPO (the tree the agents read).
"""
import collections
import json
import os
from pathlib import Path
import re
import subprocess
import sys

WORK = os.environ.get("MEMRA_IMPACT_WORK", os.path.expanduser("~/.cache/memra-impact-map-build"))
REPO = os.environ.get("MEMRA_IMPACT_REPO", str(Path(__file__).resolve().parents[2]))
S0 = os.path.join(WORK, "stage0")

_lines = {}


def file_lines(f):
    if f not in _lines:
        try:
            _lines[f] = open(os.path.join(REPO, f), errors="replace").read().split("\n")
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            _lines[f] = None
    return _lines[f]


def norm(s):
    return re.sub(r"\s+", " ", s or "").strip().strip("`")


def clean_path(f):
    f = (f or "").strip()
    for p in (REPO + "/", "./"):
        if f.startswith(p):
            f = f[len(p):]
    return f


def pin(ev):
    """Return the evidence with its line re-pinned, or None when the quote is not there."""
    f = clean_path(ev.get("file"))
    ls = file_lines(f)
    q = norm(ev.get("quote"))
    n = ev.get("line") or 0
    if not ls or len(q) < 3:
        return None
    for d in (0, 1, -1, 2, -2, 3, -3):
        m = n + d
        if 1 <= m <= len(ls) and q in norm(ls[m - 1]):
            return {"file": f, "line": m, "quote": ev.get("quote", "").strip()[:200]}
    return None


def main():
    res = json.load(open(sys.argv[1]))
    refuted, verified = set(), set()
    if len(sys.argv) > 2 and os.path.exists(sys.argv[2]):
        for v in json.load(open(sys.argv[2])):
            (refuted if v.get("refuted") else verified).add(v["claim"])
    nodes_by_file = collections.defaultdict(list)
    for l in open(os.path.join(S0, "graph/nodes.jsonl")):
        r = json.loads(l)
        if r["kind"] not in ("impl", "trait", "use"):
            nodes_by_file[r["file"]].append(r)

    def anchor(f, a, b):
        best = None
        for r in nodes_by_file.get(f, ()):
            if r["line_start"] <= a and b <= r["line_end"]:
                if best is None or r["line_end"] - r["line_start"] < best["line_end"] - best["line_start"]:
                    best = r
        return best

    def emit(it, a_, b_, cid, f, rg, ev):
        narrow = any(rg.get(k) and "*" not in rg.get(k) for k in ("families", "routes", "devices")) or bool(
            rg.get("door") and rg["door"].get("default_reached") is False)
        rec = {"claim": cid, "file": f, "a": a_, "b": b_, "anchor_item": it["id"], "anchor_a": it["line_start"],
               "anchor_hash": it["hash"], "label": rg.get("label", "")[:200], "predicate": rg.get("predicate", "")[:240],
               "kind": rg.get("predicate_kind"), "families": rg.get("families") or ["*"], "routes": rg.get("routes") or ["*"],
               "devices": rg.get("devices") or ["*"], "formats": rg.get("formats") or ["*"], "door": rg.get("door"),
               "effects": rg.get("effects") or [], "conservative": bool(rg.get("conservative")),
               "narrowing": narrow, "basis": (rg.get("narrowing_basis") or "")[:400], "evidence": ev[:4]}
        if narrow:
            base = cid.split(".")[0]
            if base in refuted and f"{base}~fam" in verified and f"{base}~fam" not in refuted:
                # the full claim fell, its family restriction alone survived a second skeptic
                rec.update(routes=["*"], devices=["*"], door=None, verdict="verified-family-only")
                stats["regions_family_only"] += 1
            elif base in refuted:
                rec.update(families=["*"], routes=["*"], devices=["*"], door=None, narrowing=False, verdict="refuted")
                stats["regions_widened"] += 1
            elif base in verified:
                rec["verdict"] = "verified"
                stats["regions_verified"] += 1
            else:
                rec["verdict"] = "unverified"
                stats["regions_unverified_narrowing"] += 1
        out["regions"].append(rec)
        stats["regions_kept"] += 1


    stats = collections.Counter()
    out = {"schema": "memra-impact-annotations-v1",
           "annotated_at": subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "tables": {}, "regions": [], "file_defaults": [], "entries": [], "exact_items": [], "dynamic_edges": []}
    sc = res.get("scouts", {})
    packs, routes, devices = sc.get("packs") or {}, sc.get("routes") or {}, sc.get("devices") or {}
    out["tables"] = {
        "families": [{k: f.get(k) for k in ("family", "aliases", "engine_types", "traits", "formats", "kv_formats", "routes",
                                             "devices_measured", "support_state", "models", "notes")} for f in packs.get("families", [])],
        "extra_families": [{"id": x["id"], "what": x["what"]} for x in packs.get("extra_families", [])],
        "predicates": [{k: p.get(k) for k in ("predicate", "meaning", "true_for", "false_for")} for p in packs.get("predicates", [])],
        "routes_vocab": routes.get("routes_vocab", []),
        "routes": [{k: r.get(k) for k in ("route", "selection_predicate", "families", "devices", "call_chain", "default_on", "door")}
                   | {"entry_items": [e["item"] for e in r.get("entry_items", [])]} for r in routes.get("routes", [])],
        "loader_dispatch": [{k: l.get(k) for k in ("families", "engine_type", "constructor_item", "predicate")} for l in routes.get("loader_dispatch", [])],
        "binaries": [{k: b.get(k) for k in ("binary", "main_item", "serves")} for b in routes.get("binaries", [])],
        "device_predicates": [{k: d.get(k) for k in ("item", "predicate", "truth")} for d in devices.get("device_predicates", [])],
        "device_defaults": [{k: d.get(k) for k in ("file", "line", "setting", "per_device", "env_seam")} for d in devices.get("device_defaults", [])],
        "build_cfgs": [{k: c.get(k) for k in ("cfg", "set_for", "meaning")} for c in devices.get("build_cfgs", [])],
    }
    for a in res.get("annotated", []):
        if not a or a.get("failed"):
            stats["failed_slices"] += 1
            continue
        sid = a["slice"]
        for i, rg in enumerate(a.get("regions", [])):
            stats["regions_in"] += 1
            ev = [p for p in (pin(e) for e in rg.get("evidence", [])) if p]
            f = clean_path(rg.get("file"))
            if not ev or not file_lines(f):
                stats["regions_dropped_evidence"] += 1
                continue
            a_, b_ = int(rg["line_start"]), int(rg["line_end"])
            if b_ < a_:
                a_, b_ = b_, a_
            it = anchor(f, a_, b_)
            if it is not None:
                spans = [(it, a_, b_, f"{sid}:R{i}")]
            else:
                # a label over a group of items: one whole-item region per item fully inside it
                inner = [r for r in nodes_by_file.get(f, ()) if a_ - 1 <= r["line_start"] and r["line_end"] <= b_ + 1
                         and not r.get("test")]
                inner = [r for r in inner if not any(o is not r and o["line_start"] <= r["line_start"] and r["line_end"] <= o["line_end"]
                                                     for o in inner)]
                if not inner:
                    stats["regions_dropped_no_anchor"] += 1
                    continue
                spans = [(r, r["line_start"], r["line_end"], f"{sid}:R{i}.{k}") for k, r in enumerate(inner)]
                stats["regions_split_items"] += len(spans)
            for it, a_, b_, cid in spans:
                emit(it, a_, b_, cid, f, rg, ev)
            continue
        for j, fd in enumerate(a.get("file_defaults", [])):
            ev = [p for p in (pin(e) for e in fd.get("evidence", [])) if p]
            if not ev:
                stats["file_defaults_dropped"] += 1
                continue
            cid = f"{sid}:F{j}"
            rec = {"claim": cid, "file": clean_path(fd["file"]), **{k: fd.get(k) or ["*"] for k in ("families", "routes", "devices", "formats")},
                   "effects": fd.get("effects") or [], "note": fd.get("note", "")[:300], "basis": (fd.get("narrowing_basis") or "")[:400], "evidence": ev[:3]}
            if cid in refuted:
                rec.update(families=["*"], routes=["*"], devices=["*"], verdict="refuted")
            elif cid in verified:
                rec["verdict"] = "verified"
            out["file_defaults"].append(rec)
        for e in a.get("entries", []):
            ev = [p for p in (pin(x) for x in e.get("evidence", [])) if p]
            if ev:
                out["entries"].append({"item": e["item"], "families": e.get("families") or ["*"], "routes": e.get("routes") or ["*"],
                                       "devices": e.get("devices") or ["*"], "role": e.get("role", "")[:200], "evidence": ev[:2]})
        for k, x in enumerate(a.get("exact_items", [])):
            ev = [p for p in (pin(e) for e in x.get("evidence", [])) if p]
            cid = f"{sid}:X{k}"
            # an output-neutral claim narrows to "no test": honored only once a skeptic verified it
            if ev and cid in verified and cid not in refuted:
                out["exact_items"].append({"claim": cid, "item": x["item"], "why": x.get("why", "")[:300], "evidence": ev[:2]})
            else:
                stats["exact_items_not_honored"] += 1
        for d in a.get("dynamic_edges", []):
            ev = [p for p in (pin(e) for e in d.get("evidence", [])) if p]
            if ev:
                out["dynamic_edges"].append({"from_item": d["from_item"], "to": d["to"], "mechanism": d.get("mechanism", "")[:300], "evidence": ev[:2]})
    # engine-type restrictions (assemble.py) are narrowing claims too: only verified types apply
    out["verified_types"] = sorted(c.split(":", 1)[1] for c in verified if c.startswith("type:") and c not in refuted)
    out["stats"] = stats
    json.dump(out, sys.stdout, indent=0)
    print(json.dumps(stats), file=sys.stderr)


if __name__ == "__main__":
    main()
