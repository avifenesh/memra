#!/usr/bin/env python3
"""Replay real past diffs through the delta query and print the chosen test sets.

    python3 tools/impact_map/history.py --map docs/impact-map/map.json --models docs/impact-map/models.json \\
        --main 40 --range "#1025=origin/main..origin/spark-sm121" ... > docs/impact-map/history.md

Each row: the diff, how many program items changed, the cells reached (family x route x device),
the effects, and the runs the plan picks per device (and what no available model covers).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import delta  # noqa: E402


def git(repo, *a):
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True).stdout.strip()


def gate_cells(repo, base, head, use_map):
    """The push gate's (target, scope) cells for the diff, with or without the map."""
    sys.path.insert(0, str(HERE.parent))
    import change_impact as ci
    try:
        rules = (HERE.parent / "change-impact.toml").read_text()  # today's rules for every replayed diff
        arch = ci.arch_table_at(repo, head)
        for t in arch.accepted:  # a target new in the head gets a push-gated row, as rules_in_force does
            if f'[target."{t}"]' not in rules:
                rules += f'\n[target."{t}"]\nname = "sm_{t}"\npush_gate = true\n'
        impact, _, _ = ci.impact_between(repo, base, head, rules, use_map=use_map)
    except ci.ImpactError as error:
        return f"REFUSED: {error}"
    cells = sorted({f"{c['name']}:{c['scope']}" for c in impact["cells"]})
    unc = sorted(impact.get("uncovered", {}))
    return ", ".join(cells) + (f" (uncovered: {', '.join(unc)})" if unc else "") if cells or unc else "none"


def row(repo, label, base, head, m, models, dev_target=None, pushed=None):
    rep = delta.delta(repo, base, head, map_data=m)
    s = delta.summarize(rep)
    pl = delta.plan(repo, base, rep, models=models, ledger={"records": []}, device_target=dev_target, push_gated=pushed)
    changed = sum(1 for f in rep["files"] for it in f.get("items", []) if it.get("change") not in ("test-only",))
    files = len(rep["files"])
    runs = []
    for dev, ms in sorted(pl["runs"].items()):
        for model, v in sorted(ms.items()):
            runs.append(f"{dev}: {model.split('/')[-1]} [{','.join(v['routes'])}] {'+'.join(v['tests'])}")
    unc = sorted({(u["family"], u["device"]) for u in pl["uncovered"]})
    fams = s["families"]
    return {"label": label, "files": files, "items": changed, "cells": s["cells"],
            "gate_rules_only": gate_cells(repo, base, head, False), "gate_with_map": gate_cells(repo, base, head, True),
            "families": "*" if len(fams) == len(m["vocab"]["families"]) else ",".join(fams) or "-",
            "devices": ",".join(s["devices"]) or "-", "effects": ",".join(s["effects"]) or "-",
            "kernel": s["kernel"], "doors": ",".join(rep.get("doors", [])), "runs": runs,
            "uncovered": len(unc), "daily": "; ".join(f"{d}: {len(f)}" for d, f in pl.get("daily", {}).items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parent.parent)
    ap.add_argument("--map", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--main", type=int, default=40)
    ap.add_argument("--main-ref", default="origin/main")
    ap.add_argument("--range", action="append", default=[], help="LABEL=BASE..HEAD")
    ap.add_argument("--commits", type=Path, help="file of commit shas to replay (each against its first parent)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    m = json.loads(Path(a.map).read_text())
    models = json.loads(Path(a.models).read_text())
    sys.path.insert(0, str(HERE.parent))
    import change_impact as ci
    from impact_map import gate as g
    arch = ci.arch_table_at(a.repo, "HEAD")
    rules = ci.load_rules((HERE.parent / "change-impact.toml").read_text(), arch)
    dev_target = g.device_targets(rules, arch)
    pushed = {t for t, r in rules.get("target", {}).items() if r.get("push_gate")}
    import os
    os.environ["MEMRA_IMPACT_MAP_FILE"] = str(Path(a.map).resolve())
    os.environ["MEMRA_IMPACT_MODELS_FILE"] = str(Path(a.models).resolve())
    rows = []
    listing = git(a.repo, "log", f"-{a.main}", "--first-parent", "--format=%H %s", a.main_ref).splitlines() if a.main else []
    if a.commits:
        listing = [git(a.repo, "log", "-1", "--format=%H %s", c.strip()) for c in a.commits.read_text().split() if c.strip()]
    for line in listing:
        sha, subj = line.split(" ", 1)
        rows.append(row(a.repo, f"`{sha[:10]}` {subj[:70]}", f"{sha}^", sha, m, models, dev_target, pushed))
    for spec in a.range:
        label, rng = spec.split("=", 1)
        base, head = rng.split("..")
        mb = git(a.repo, "merge-base", base, head)
        rows.append(row(a.repo, label, mb, head, m, models, dev_target, pushed))
    if a.json:
        print(json.dumps(rows, indent=1))
        return
    print("| diff | files | changed items | gate cells, path rules only (#1030) | gate cells with the map | reached families | devices | effects | push runs the plan picks | daily (families per device) | reached cells no measured model covers |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        runs = "<br>".join(r["runs"]) or "none"
        eff = r["effects"] + (" +kernel" if r["kernel"] else "") + (f" doors:{r['doors']}" if r["doors"] else "")
        print(f"| {r['label']} | {r['files']} | {r['items']} | {r['gate_rules_only']} | {r['gate_with_map']} | {r['families']} | {r['devices']} | {eff} | {runs} | {r['daily'] or '-'} | {r['uncovered']} |")


if __name__ == "__main__":
    main()
