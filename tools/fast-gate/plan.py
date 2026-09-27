#!/usr/bin/env python3
"""Explain development impact. The map seeds probes; the dependency graph expands it.

GPU decisions are advisory until shadowed on the intended model and hardware.
Only the separately isolated CPU components have admitted result reuse.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.PIPE)


def changed_paths(repo, ref):
    git(repo, "rev-parse", "--verify", ref + "^{commit}")
    paths = set()
    for args in (("diff", "--no-ext-diff", "--no-renames", "--name-only", "-z", ref, "--"),
                 ("ls-files", "--others", "--exclude-standard", "-z")):
        paths.update(p.decode() for p in git(repo, *args).split(b"\0") if p)
    hidden = []
    for raw in git(repo, "ls-files", "-v", "-z").split(b"\0"):
        if raw and (raw[:1].islower() or raw[:1] == b"S"):
            hidden.append(raw[2:].decode())
    for raw in git(repo, "ls-files", "--others", "--ignored", "--exclude-standard", "-z", "--",
                   ".cargo", "crates", "tools").split(b"\0"):
        if raw and not (b"/__pycache__/" in raw and raw.endswith(b".pyc")):
            hidden.append(raw.decode())
    return sorted(paths), sorted(set(hidden))


def registry(directory):
    rows, default = [], None
    for n, line in enumerate((directory / "map.tsv").read_text().splitlines(), 1):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 4:
            raise ValueError(f"map.tsv:{n}: expected four fields")
        rx, scope, probes, spec = fields
        row = dict(pattern=rx, scope=scope, probes=csv(probes), spec=csv(spec), line=n)
        if rx == "DEFAULT":
            if default:
                raise ValueError("duplicate DEFAULT")
            default = row
        else:
            rows.append((re.compile(rx), row))
    if not default:
        raise ValueError("missing DEFAULT")
    models = {}
    for line in (directory / "models.tsv").read_text().splitlines():
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 6 or fields[0] in models:
            raise ValueError("malformed or duplicate probe registry entry")
        models[fields[0]] = fields[1:]
    for row in [r for _, r in rows] + [default]:
        if set(row["probes"] + row["spec"]) - models.keys():
            raise ValueError(f"unknown mapped probe at map.tsv:{row['line']}")
    return rows, default, models


def csv(value):
    return [p for p in value.split(",") if p and p != "-"]


def graph_order(nodes):
    order, active = [], set()
    def visit(name):
        if name not in nodes:
            raise ValueError(f"unknown dependency: {name}")
        if name in active:
            raise ValueError(f"dependency cycle: {name}")
        if name in order:
            return
        active.add(name)
        for dep in nodes[name].get("needs", []):
            visit(dep)
        active.remove(name)
        order.append(name)
    for name in nodes:
        visit(name)
    return order


def make_plan(paths, directory=HERE, overrides=(), hidden=(), context_changes=()):
    rows, default, models = registry(directory)
    config = json.loads((directory / "dependencies.json").read_text())
    if config["schema"] != "memra-dev-dependencies-v1":
        raise ValueError("unsupported dependency schema")
    nodes = config["nodes"]
    order = graph_order(nodes)
    known_contexts = set(nodes["native-checkpoint"]["contexts"])
    if set(context_changes) - known_contexts:
        raise ValueError("unknown context category")
    scope, sections, probes, specs = "none", set(), set(), set()
    reasons, expansion, impact = [], [], {}
    def add(row):
        nonlocal scope
        if row["scope"] == "all":
            scope = "all"
        elif row["scope"] not in ("none", "synthetic"):
            sections.update(csv(row["scope"]))
            if scope != "all":
                scope = "csv"
        elif row["scope"] == "synthetic" and scope == "none":
            scope = "synthetic"
        probes.update(row["probes"])
        specs.update(row["spec"])
    for path in sorted(set(paths)):
        p = PurePosixPath(path)
        if p.is_absolute() or ".." in p.parts or "\n" in path or "\t" in path:
            raise ValueError("noncanonical changed path")
        matches = [r for rx, r in rows if rx.search(path)]
        for row in matches or [default]:
            add(row)
        direct = [n for n in order if any(fnmatch.fnmatchcase(path, g)
                  for g in nodes[n].get("paths", []))]
        for name in direct:
            impact.setdefault(name, []).append(path)
        if not direct:
            expansion.append(f"unmodelled input: {path}")
        # Broad legacy no-gate rows are only hints, never dependency declarations.
        if not matches:
            expansion.append(f"no probe mapping: {path}")
        reasons.append({"path": path, "map_lines": [r["line"] for r in matches], "direct_nodes": direct})
    for name in order:
        inherited = [d for d in nodes[name].get("needs", []) if d in impact]
        contexts = sorted(set(nodes[name].get("contexts", [])) & set(context_changes))
        if inherited or contexts:
            impact.setdefault(name, []).extend(["dependency:" + d for d in inherited] +
                                                ["context:" + c for c in contexts])
        if name in impact and nodes[name].get("expand"):
            expansion.append(nodes[name]["expand"])
    expansion.extend("hidden/index-masked input: " + p for p in hidden)
    if expansion:
        add(default)
        scope = "all"
    if overrides:
        unknown = set(overrides) - models.keys()
        if unknown:
            raise ValueError("unknown requested probes: " + ", ".join(sorted(unknown)))
        probes = {p for p in overrides if models[p][0] not in ("spec", "gspec")}
        specs = set(overrides) - probes
    components = []
    for name, item in config["components"].items():
        changed = [p for p in paths if p in item["inputs"] or p.startswith("tools/fast-gate/")]
        if changed:
            components.append({"id": name, "decision": "run", "because": changed,
                               "scope": item["scope"]})
    return {"schema": "memra-dev-plan-v1", "qualification": False,
            "gpu_selection": "shadow-only", "changed": sorted(set(paths)),
            "kernel_scope": scope, "kernel_sections": sorted(sections),
            "probes": sorted(probes), "spec_probes": sorted(specs),
            "reasons": reasons, "transitive_impact": impact,
            "decision": "expand" if expansion else ("run" if paths or overrides else "no-change"),
            "expansion": sorted(set(expansion)), "components": components,
            "required_unregistered": sorted({g for n in impact for g in nodes[n].get("required", [])}),
            "context_contract": {k: "exact identity required before native evidence reuse" for k in sorted(known_contexts)}}


def summary(plan):
    print(f"fast-gate plan: {plan['decision']} (development only; GPU selection shadow-only)")
    print(f"  kernel-check: {plan['kernel_scope']} {','.join(plan['kernel_sections'])}")
    print("  probes: " + (",".join(plan["probes"] + plan["spec_probes"]) or "none"))
    for why in plan["expansion"]:
        print("  EXPAND: " + why)
    for gate in plan["required_unregistered"]:
        print("  REQUIRED, not covered by legacy probes: " + gate)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT)
    parser.add_argument("--diff", default="HEAD")
    parser.add_argument("--probes", default="")
    parser.add_argument("--changed", action="append", help="explicit paths for offline planning only")
    parser.add_argument("--context-changed", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cache", type=Path, help="verify matching isolated CPU receipts, without execution")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--fields", type=Path, help="read saved plan as shell-safe TSV fields")
    args = parser.parse_args()
    try:
        if args.fields:
            plan = json.loads(args.fields.read_text())
            for k, v in (("scope", plan["kernel_scope"]), ("sections", ",".join(plan["kernel_sections"])),
                         ("probes", ",".join(plan["probes"])), ("spec", ",".join(plan["spec_probes"])),
                         ("decision", plan["decision"])):
                print(k + "\t" + v)
            return 0
        no_git_diagnostics = False
        if args.changed is not None:
            paths, hidden = args.changed, []
        else:
            try:
                root = Path(git(args.repo, "rev-parse", "--show-toplevel").decode().strip())
                has_git = root.resolve() == args.repo.resolve()
            except subprocess.SubprocessError:
                has_git = False
            if not has_git and args.probes:
                # Explicit diagnostics are useful in source-only rsync trees.
                # They have no change-coverage or qualification authority.
                paths, hidden, no_git_diagnostics = [], [], True
            else:
                paths, hidden = changed_paths(args.repo, args.diff)
        plan = make_plan(paths, overrides=csv(args.probes), hidden=hidden, context_changes=args.context_changed)
        if no_git_diagnostics:
            plan["decision"] = "expand"
            plan["expansion"].append("Git metadata unavailable: explicit probe diagnostics only; change coverage unknown")
        if args.cache:
            from component import lookup
            for item in plan["components"]:
                item.update(lookup(args.repo, args.cache, item["id"]))
        plan["diff"] = args.diff
        plan["offline_paths"] = args.changed is not None
        plan["no_git_diagnostics"] = no_git_diagnostics
        encoded = json.dumps(plan, indent=2, sort_keys=True) + "\n"
        if args.out:
            args.out.write_text(encoded)
        if args.json:
            print(encoded, end="")
        else:
            summary(plan)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"fast-gate: REFUSED plan: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
