"""Adapter between the impact map and the push gate (tools/change_impact.py, tools/cell_evidence.py).

The path rules in tools/change-impact.toml stay the first word: they decide a file's class and the
build targets its path can reach. For a Rust or CUDA program file, the map then refines that
answer from the items the diff actually changed:

- no item's token hash moved (comment, doc-comment or format-only edit): the file reaches no cell;
- every changed item is known to the map: the file's build targets are intersected with the
  targets of the devices those items' reach includes (both sides are over-approximations, so the
  intersection is still one), and the perf cell is dropped when no changed item has a speed effect;
- any changed item the map cannot place (new, renamed, unknown file): the path rule's answer stands.

Files whose class is build, gate or unknown, files flagged for device-keyed defaults, and
non-program files are never narrowed here.

The map in force is the push base's (docs/impact-map/map.json at the base), like the rules, so a
change cannot narrow its own requirements; only the change that introduces the map uses the head's.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

try:  # imported as tools/impact_map/gate.py from tools/change_impact.py
    from . import delta  # type: ignore
except ImportError:  # pragma: no cover - direct execution
    import delta  # type: ignore

NARROWABLE = ("kernel", "numeric", "serving")
SPEED = {"speed"}


def map_in_force(repo, base, head):
    override = os.environ.get("MEMRA_IMPACT_MAP_FILE")
    if override:  # what-if (history replay, map refresh review): a map file instead of the base's
        return delta.Map(json.loads(Path(override).read_text())), f"file {override}"
    data = delta.load_json_at(repo, base, delta.MAP_PATH)
    origin = "base"
    if data is None:
        data = delta.load_json_at(repo, head, delta.MAP_PATH)
        origin = "head (introduced by this change)"
    if data is None:
        return None, None
    return delta.Map(data), origin


def device_targets(rules, arch):
    """Device class -> build target, from the rules' device rows and the arch table."""
    out = {}
    for name, row in rules.get("device", {}).items():
        out[name] = arch.target_for_cap(row["compute_cap"])
    if "89" in arch.accepted:
        out.setdefault("sm89", "89")
    out["cpu"] = None
    return out


class Refiner:
    def __init__(self, repo, base, head, rules, arch):
        self.repo, self.base, self.head = repo, base, head
        self.rules = rules
        self.map, self.origin = map_in_force(repo, base, head)
        self.dev_target = device_targets(rules, arch) if self.map else {}
        self.cells = {}
        self.effects = set()
        self.kernel = False
        self.notes = []

    def active(self):
        return self.map is not None

    def file(self, result):
        """Refine one classify_file() result in place."""
        if not self.map or result["class"] not in NARROWABLE or result.get("defaults"):
            return result
        path = result["path"]
        if not (path.endswith(".rs") or path.startswith(delta.CU_PREFIX + "cu/")):
            return result
        rep = delta.delta(self.repo, self.base, self.head, paths=[path], map_data=self.map.data)
        entry = next((f for f in rep["files"] if f["path"] == path), None)
        changed = [it for it in (entry or {}).get("items", []) if it.get("change") != "test-only"]
        if entry is not None and not changed:
            result.update({"class": "comment", "perf": False,
                           "why": "token-identical: no item's comment-free token stream changed"})
            return result
        for c, e in rep["cells"].items():
            slot = self.cells.setdefault(c, {"why": set(), "effects": set()})
            slot["why"].update(e["why"])
            slot["effects"].update(e["effects"])
        self.effects |= rep["effects"]
        self.kernel |= rep["kernel"]
        if any(it.get("source") not in ("map-item", "map-region") for it in changed):
            result["map"] = "not narrowed: an item the map cannot place"
            return result
        devices = {d for (_, _, d) in rep["cells"]}
        families = {f for (f, _, _) in rep["cells"]}
        covered = {name for name, pack in self.rules.get("pack", {}).items() if pack.get("roster")}
        if families and not (families & covered) and result["class"] == "numeric":
            result["uncovered_families"] = sorted(families)
        targets = {self.dev_target.get(d) for d in devices} - {None}
        before = set(result["targets"]) if result.get("targets") is not None else None
        result["targets"] = sorted(targets if before is None else (before & targets))
        effects = set().union(*(set(it.get("effects", ())) for it in changed)) if changed else set()
        if result.get("perf") and not (effects & SPEED):
            result["perf"] = False
        result["map"] = f"{len(changed)} changed item(s) reach {len(rep['cells'])} cell(s) on {','.join(sorted(devices)) or 'no device'}"
        return result

    def report(self):
        if not self.map:
            return None
        rep = {"cells": self.cells, "effects": self.effects, "kernel": self.kernel, "files": [], "doors": []}
        pushed = {t for t, row in self.rules.get("target", {}).items() if row.get("push_gate")}
        pl = delta.plan(self.repo, self.base if self.origin == "base" else self.head, rep,
                        models=_json_env("MEMRA_IMPACT_MODELS_FILE"), ledger=_json_env("MEMRA_IMPACT_LEDGER_FILE"),
                        device_target=self.dev_target, push_gated=pushed)
        return {"map_from": self.origin, "map_commit": self.map.data.get("source_commit"),
                "summary": delta.summarize(rep), "plan": pl}


def _json_env(name):
    p = os.environ.get(name)
    return json.loads(Path(p).read_text()) if p else None


def render_plan(mapinfo):
    if not mapinfo:
        return []
    s, pl = mapinfo["summary"], mapinfo["plan"]
    lines = [f"  map ({mapinfo['map_from']}, built at {str(mapinfo['map_commit'])[:12]}): {s['cells']} cell(s); "
             f"families={','.join(s['families']) or '-'} devices={','.join(s['devices']) or '-'} effects={','.join(s['effects']) or '-'}"]
    for dev, ms in sorted(pl["runs"].items()):
        for model, v in sorted(ms.items()):
            lines.append(f"    run on {dev}: {model} ({v['family']}): routes {','.join(v['routes'])}; tests {'+'.join(v['tests'])}")
    if pl.get("daily"):
        lines.append("    daily run (per device, not at push): " + "; ".join(f"{d}: {','.join(f)}" for d, f in pl["daily"].items()))
    unc = sorted({(u["family"], u["device"]) for u in pl["uncovered"]})
    if unc:
        lines.append(f"    no model on hand for {len(unc)} family x device pair(s): "
                     + ", ".join(f"{f}@{d}" for f, d in unc[:12]) + (" ..." if len(unc) > 12 else ""))
    for p in pl["proven"][:10]:
        lines.append(f"    already measured for this exact code: {p['model']} {p['route']} {p['device']} {p['test']}")
    return lines


def dumps(mapinfo):
    return json.dumps(mapinfo, default=sorted)
