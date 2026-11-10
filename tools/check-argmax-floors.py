#!/usr/bin/env python3
"""Every argmax cell the release battery runs has a floor row on every maintained device class.

The release battery runs the argmax margin gate with --require-floor, so a cell without a floor
row refuses a release. This census reads the battery's own cell list
(`tools/release-battery.sh --list-argmax-cells`: every roster model, the gate prompt, each
decode program), the maintained device classes (the classes of push-gated build targets in
`tools/change-impact.toml`, through `tools/change_impact.py`) and
`tools/argmax-margin-floors.tsv`. It fails when

  * a maintained class has no calibrated device at all (no row whose device name the class's
    `name` matches), or
  * a device that has any row in the table lacks the row of a battery cell.

So a roster model cannot be added, a device cannot be half calibrated, and a maintained class
cannot be left out, without its calibration cells. A class that is not maintained (H100, B200)
is not demanded; the gate reports its cells as not judged.

The floor row must be for the prompt's bytes: its prompt_sha256 is checked against the file.

usage: check-argmax-floors.py [--repo DIR]     exit 0 complete, 1 a row is missing, 2 unreadable
"""
import argparse
import hashlib
import importlib.util
import pathlib
import subprocess
import sys


def change_impact(repo):
    """tools/change_impact.py of a tree, imported: the one device-name mapping."""
    spec = importlib.util.spec_from_file_location("change_impact", repo / "tools/change_impact.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("change_impact", module)
    spec.loader.exec_module(module)
    return module


def battery_cells(repo, roster=None):
    """[(roster id, model, prompt name, decode)] from the battery's own list mode."""
    command = ["bash", str(repo / "tools/release-battery.sh"), "--list-argmax-cells"]
    if roster:
        command += ["--roster", str(roster)]
    out = subprocess.run(command, capture_output=True, text=True, check=True).stdout
    cells = [tuple(line.split("\t")) for line in out.splitlines() if line.strip()]
    if not cells or any(len(c) != 4 for c in cells):
        raise ValueError("release-battery.sh --list-argmax-cells printed no cell or a malformed one")
    return cells


def floor_rows(text):
    """{(model, prompt, prompt_sha256, decode, device)} of the floors table."""
    rows = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 13:
            raise ValueError(f"floors line {n}: expected 13 tab-separated fields")
        rows.add(tuple(fields[:5]))
    return rows


def missing_rows(cells, rows, prompt_sha256):
    """The (device, roster id, model, prompt, decode) of every battery cell with no floor row
    on a device the table covers. `prompt_sha256` maps a prompt name to its file's digest."""
    devices = sorted({row[4] for row in rows})
    missing = []
    for device in devices:
        for rid, model, prompt, decode in cells:
            if (model, prompt, prompt_sha256[prompt], decode, device) not in rows:
                missing.append((device, rid, model, prompt, decode))
    return missing


def uncalibrated_classes(maintained, rows, class_of):
    """The maintained device classes with no calibrated device: no row whose device name maps
    to the class. `class_of` maps a device name to its class (or None)."""
    covered = {class_of(row[4]) for row in rows}
    return [cls for cls in maintained if cls not in covered]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(pathlib.Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    repo = pathlib.Path(args.repo)
    try:
        cells = battery_cells(repo)
        rows = floor_rows((repo / "tools/argmax-margin-floors.tsv").read_text())
        shas = {}
        for _, _, prompt, _ in cells:
            path = repo / "research/e2e/prompts" / f"{prompt}.txt"
            shas[prompt] = hashlib.sha256(path.read_bytes()).hexdigest()
        ci = change_impact(repo)
        rules, arch = ci.tree_rules(repo)
        maintained = ci.maintained_classes(rules, arch)
    except (OSError, ValueError, subprocess.CalledProcessError) as err:
        print(f"check-argmax-floors: {err}", file=sys.stderr)
        return 2
    except Exception as err:  # ImpactError: the rules or the arch table could not be read
        print(f"check-argmax-floors: {err}", file=sys.stderr)
        return 2
    devices = sorted({row[4] for row in rows})
    missing = missing_rows(cells, rows, shas)
    for device, rid, model, prompt, decode in missing:
        print(f"check-argmax-floors: MISSING floor row: device={device!r} model={model} ({rid}) "
              f"prompt={prompt} decode={decode}", file=sys.stderr)
    bare = uncalibrated_classes(maintained, rows, lambda name: ci.device_class_of(rules, name))
    for cls in bare:
        print(f"check-argmax-floors: MISSING device: maintained class {cls} "
              f"({rules['device'][cls]['name']}) has no calibrated device: {len(cells)} floor rows needed",
              file=sys.stderr)
    if missing or bare:
        print("check-argmax-floors: calibrate each with tools/argmax-margin-gate.sh <model> "
              "[--decode batch] --calibrate on that device and add the row", file=sys.stderr)
        return 1
    print(f"check-argmax-floors: OK: {len(cells)} battery cells x {len(devices)} device(s) "
          f"({', '.join(devices) or 'none'}), every floor row present; maintained classes "
          f"{', '.join(maintained)} each calibrated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
