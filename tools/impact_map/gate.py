"""Adapter between the impact map and the push gate (tools/change_impact.py, tools/cell_evidence.py).

The path rules in tools/change-impact.toml stay the first word: they decide a file's class and the
build targets its path can reach. For a Rust or CUDA program file, the map then refines that
answer from the items the diff actually changed:

- no item's token hash moved, including the synthetic item that holds every token outside an
  item (comment, doc-comment or format-only edit): the file reaches no cell;
- every changed item is placed by the map: the file's build targets are intersected with the
  targets of the devices those items' reach includes (both sides are over-approximations, so the
  intersection is still one), perf is dropped when no changed item has a speed effect, and a
  numeric file whose items reach only families no battery roster model belongs to is UNCOVERED;
- anything the map cannot place (a new or renamed item, tokens outside items, an item placed
  nowhere by default, a file the scanner does not read, a base the map is not an ancestor of): the
  path rule's answer stands. Device narrowing also stands down when the arch table moved since the
  map was built.

Files whose class is build, gate or unknown, files flagged for device-keyed defaults, and
non-program files are never narrowed here.

The map in force is the push base's (docs/impact-map/map.json at the base), like the rules, so a
change cannot narrow its own requirements; only the change that introduces the map uses the
head's. A change to map.json itself must leave a loadable map, or the gate refuses it.
"""
from __future__ import annotations

import json

try:  # imported as tools/impact_map/gate.py from tools/change_impact.py
    from . import delta  # type: ignore
except ImportError:  # pragma: no cover - direct execution
    import delta  # type: ignore

NARROWABLE = ("kernel", "numeric", "serving")
SPEED = {"speed"}
PLACED = ("map-item", "map-region", "map-item+drift", "map-region+drift")


class MapError(ValueError):
    pass


def load_map(repo, ref):
    data = delta.load_json_at(repo, ref, delta.MAP_PATH)
    return None if data is None else delta.Map(data)


def map_in_force(repo, base, head, map_ref=None):
    """(Map or None, origin, note). map_ref pins the map to another commit (still_proven pins the
    push base). A base map that cannot be read makes the refiner inactive (path rules only)."""
    ref = map_ref or base
    try:
        m = load_map(repo, ref)
    except (ValueError, KeyError, TypeError, delta.DeltaError) as error:
        return None, None, f"map at {str(ref)[:12]} unreadable ({error}): path rules only"
    if m is not None:
        return m, "base" if map_ref is None else f"pinned {str(ref)[:12]}", None
    if map_ref is not None:
        return None, None, f"no map at {str(ref)[:12]}: path rules only"
    try:
        m = load_map(repo, head)
    except (ValueError, KeyError, TypeError, delta.DeltaError) as error:
        raise MapError(f"the head's {delta.MAP_PATH} cannot be loaded: {error}") from error
    return (m, "head (introduced by this change)", None) if m else (None, None, None)


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
    def __init__(self, repo, base, head, rules, arch, map_ref=None, map_data=None):
        self.repo, self.base, self.head = repo, base, head
        self.rules = rules
        self.note = None
        self.whatif = map_data is not None
        if self.whatif:  # what-if replays (history.py) pass a map explicitly; never a push decision
            self.map, self.origin = delta.Map(map_data), "what-if file"
        else:
            self.map, self.origin, self.note = map_in_force(repo, base, head, map_ref)
        # a change to the map must leave a map the next push can load
        if delta.MAP_PATH in delta.changed_paths(repo, base, head):
            try:
                load_map(repo, head)
            except (ValueError, KeyError, TypeError, delta.DeltaError) as error:
                raise MapError(f"this change leaves {delta.MAP_PATH} unloadable: {error}") from error
        self.dev_target = device_targets(rules, arch) if self.map else {}
        self.cells = {}
        self.effects = set()
        self.kernel = False

    def active(self):
        return self.map is not None

    def file(self, result):
        """Refine one classify_file() result in place."""
        if not self.map or result["class"] not in NARROWABLE or result.get("defaults"):
            return result
        path = result["path"]
        if not (path.endswith(".rs") or path.startswith(delta.CU_PREFIX + "cu/")):
            return result
        if not delta.scannable(path):
            result["map"] = "not narrowed: a file the scanner does not read"
            return result
        rep = delta.delta(self.repo, self.base, self.head, paths=[path], map_obj=self.map, whatif=self.whatif)
        entry = next((f for f in rep["files"] if f["path"] == path), None)
        if entry is None or "items" not in entry:
            result["map"] = "not narrowed: the delta has no entry for this file"
            return result
        changed = [it for it in entry["items"] if it.get("change") != "test-only"]
        if not entry["items"]:
            result.update({"class": "comment", "perf": False,
                           "why": "token-identical: no item's comment-free token stream changed"})
            return result
        if not changed:
            result.update({"class": "tests", "perf": False, "why": "only test items changed"})
            return result
        for c, e in rep["cells"].items():
            slot = self.cells.setdefault(c, {"why": set(), "effects": set(), "linked": False})
            slot["why"].update(e["why"])
            slot["effects"].update(e["effects"])
            slot["linked"] |= e.get("linked", True)
        self.effects |= rep["effects"]
        self.kernel |= rep["kernel"]
        unplaced = sorted({it.get("source", "?") for it in changed if it.get("source") not in PLACED})
        if unplaced:
            result["map"] = "not narrowed: " + ", ".join(unplaced)
            return result
        devices = {d for (_, _, d) in rep["cells"]}
        families = {f for (f, _, _) in rep["cells"]}
        covered = {name for name, pack in self.rules.get("pack", {}).items() if pack.get("roster")}
        if families and not (families & covered) and result["class"] == "numeric":
            result["uncovered_families"] = sorted(families)
        if not rep["arch_moved"]:
            targets = {self.dev_target.get(d) for d in devices} - {None}
            before = set(result["targets"]) if result.get("targets") is not None else None
            result["targets"] = sorted(targets if before is None else (before & targets))
        effects = set().union(*(set(it.get("effects", ())) for it in changed))
        if result.get("perf") and not (effects & SPEED):
            result["perf"] = False
        result["map"] = (f"{len(changed)} changed item(s) reach {len(rep['cells'])} cell(s) on "
                         f"{','.join(sorted(devices)) or 'no device'}"
                         + (" (arch table moved since the map: targets kept)" if rep["arch_moved"] else ""))
        return result

    def report(self):
        if not self.map:
            return {"map_from": None, "note": self.note} if self.note else None
        rep = {"cells": self.cells, "effects": self.effects, "kernel": self.kernel, "files": [], "doors": []}
        pushed = {t for t, row in self.rules.get("target", {}).items() if row.get("push_gate")}
        ref = self.base if self.origin == "base" else self.head
        pl = delta.plan(self.repo, ref, rep, device_target=self.dev_target, push_gated=pushed)
        return {"map_from": self.origin, "map_commit": self.map.data.get("source_commit"),
                "summary": delta.summarize(rep), "plan": pl}


def render_plan(mapinfo):
    if not mapinfo:
        return []
    if mapinfo.get("map_from") is None:
        return [f"  map: {mapinfo['note']}"] if mapinfo.get("note") else []
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
        lines.append(f"    no measured model for {len(unc)} family x device pair(s): "
                     + ", ".join(f"{f}@{d}" for f, d in unc[:12]) + (" ..." if len(unc) > 12 else ""))
    for p in pl["proven"][:10]:
        lines.append(f"    already measured for this exact code: {p['model']} {p['route']} {p['device']} {p['test']}")
    return lines


def dumps(mapinfo):
    return json.dumps(mapinfo, default=sorted)
