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

  * a battery cell's floor row has no stored table of main's rows for the battery's window in
    `tools/argmax-margin-rows/` (memra#1204), or the stored table does not come from that floor
    row's receipt: its receipt must be the floor row's and its rows the receipt's last rows,
    byte for byte.

So a roster model cannot be added, a device cannot be half calibrated, and a maintained class
cannot be left out, without its calibration cells. A class that is not maintained (H100, B200)
is not demanded; the gate reports its cells as not judged.

The floor row must be for the prompt's bytes: its prompt_sha256 is checked against the file.
A stored table's model_sha256 names the artifact on the device; the census checks its form only.

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


def floor_rows(text, receipts=None):
    """{(model, prompt, prompt_sha256, decode, device)} of the floors table. `receipts`, when
    given a dict, is filled with each key's receipt path."""
    rows = set()
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) != 13:
            raise ValueError(f"floors line {n}: expected 13 tab-separated fields")
        rows.add(tuple(fields[:5]))
        if receipts is not None:
            receipts[tuple(fields[:5])] = fields[12]
    return rows


def bad_receipts(receipts, repo):
    """The floor rows whose receipt is not a banked file of the tree: a path under research/,
    relative, without '..', that exists. (The judge's own parser and the recomputation of every
    row from its receipt are the engine test
    the_tree_floors_table_parses_and_every_row_is_its_receipts_calibration.)"""
    bad = []
    for key, path in sorted(receipts.items()):
        if not path.startswith("research/") or ".." in path.split("/") or not (repo / path).is_file():
            bad.append((key, path))
    return bad


STORED_FIELDS = ("model", "model_sha256", "prompt", "prompt_sha256", "decode", "device", "window",
                 "experts", "commit", "receipt")


def experts_of(log):
    """Where a probe log's MoE experts ran, as memra_engine::argmax_margin::experts_of reads it:
    `resident`, `slru` or `none` (no decision line: dense, or forced off the card)."""
    decisions = [l for l in log.splitlines() if l.startswith("[moe] resident-experts decision")]
    if not decisions:
        return "none"
    return "resident" if all(l.rstrip().endswith("-> RESIDENT") for l in decisions) else "slru"

# The window the release battery's gate cells judge: the gate's default.
BATTERY_WINDOW = 12


def table_lines(text):
    """The probe table lines of a log, trailing whitespace dropped (the lines the gate compares)."""
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 7 and f[0].isdigit() and f[1].isdigit():
            out.append(line.rstrip())
    return out


def stored_table(text, name="stored table"):
    """One `.rows` file: (header dict, table lines). The form memra_engine::argmax_margin writes."""
    header, lines = {}, text.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("#") or not line.strip():
            continue
        if line == "rows":
            rows = table_lines("\n".join(lines[i + 1:]))
            break
        field, sep, value = line.partition("\t")
        if not sep or field not in STORED_FIELDS or field in header or not value.strip():
            raise ValueError(f"{name}: header line {line!r}")
        header[field] = value
    else:
        raise ValueError(f"{name}: no `rows` line")
    missing = [f for f in STORED_FIELDS if f not in header]
    if missing:
        raise ValueError(f"{name}: missing {', '.join(missing)}")
    if not header["window"].isdigit() or int(header["window"]) != len(rows) or not rows:
        raise ValueError(f"{name}: {len(rows)} rows for a window of {header['window']}")
    if len(header["model_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in header["model_sha256"]):
        raise ValueError(f"{name}: model_sha256 is not a sha256")
    return header, rows


def stored_tables(directory):
    """[(path, header, rows)] of every `.rows` file of the stored-rows directory."""
    if not directory.is_dir():
        return []
    return [(p, *stored_table(p.read_text(), str(p))) for p in sorted(directory.glob("*.rows"))]


def stored_problems(cells, rows, receipts, prompt_sha256, tables, read_receipt):
    """What is wrong with the stored rows of the battery cells' floor rows: a cell on a covered
    device without its stored table for the battery window, two tables for one cell, or a table
    that is not its floor row's receipt's last rows. `read_receipt(path)` gives a receipt's text."""
    problems = []
    for key in sorted(rows):
        model, prompt, sha, decode, device = key
        if not any((model, prompt, prompt_sha256.get(prompt), decode) == (c[1], c[2], sha, c[3]) for c in cells):
            continue
        found = [(p, h, r) for p, h, r in tables
                 if (h["model"], h["prompt"], h["prompt_sha256"], h["decode"], h["device"]) == key
                 and int(h["window"]) == BATTERY_WINDOW]
        if not found:
            problems.append(f"MISSING stored rows: device={device!r} model={model} prompt={prompt} "
                            f"decode={decode} window={BATTERY_WINDOW}")
            continue
        if len({h["model_sha256"] for _, h, _ in found}) != len(found):
            problems.append(f"TWO stored tables for one artifact: device={device!r} model={model} decode={decode}")
        for path, header, stored in found:
            if header["receipt"] != receipts[key]:
                problems.append(f"STALE stored rows {path.name}: receipt {header['receipt']} is not the floor "
                                f"row's {receipts[key]}")
                continue
            receipt = read_receipt(header["receipt"])
            last = table_lines(receipt)[-BATTERY_WINDOW:]
            if last != stored:
                problems.append(f"STALE stored rows {path.name}: not the last {BATTERY_WINDOW} rows of "
                                f"{header['receipt']}")
            elif header["experts"] != experts_of(receipt):
                problems.append(f"STALE stored rows {path.name}: experts {header['experts']} but the receipt "
                                f"ran {experts_of(receipt)}")
    return problems


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
        receipts = {}
        rows = floor_rows((repo / "tools/argmax-margin-floors.tsv").read_text(), receipts)
        tables = stored_tables(repo / "tools/argmax-margin-rows")
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
    unbanked = bad_receipts(receipts, repo)
    for key, path in unbanked:
        print(f"check-argmax-floors: RECEIPT not banked in the tree: device={key[4]!r} model={key[0]} "
              f"decode={key[3]}: {path!r}", file=sys.stderr)
    if unbanked:
        return 1
    try:
        stored = stored_problems(cells, rows, receipts, shas, tables, lambda p: (repo / p).read_text())
    except OSError as err:
        print(f"check-argmax-floors: {err}", file=sys.stderr)
        return 2
    for problem in stored:
        print(f"check-argmax-floors: {problem}", file=sys.stderr)
    if missing or bare:
        print("check-argmax-floors: calibrate each with tools/argmax-margin-gate.sh <model> "
              "[--decode batch] --calibrate on that device and add the row", file=sys.stderr)
    if stored:
        print("check-argmax-floors: store main's rows with tools/argmax-margin-gate.sh <model> "
              "[--decode batch] --calibrate --rows-out FILE --commit C, or from the floor row's receipt "
              "with argmax-margin-probe --judge <receipt> --calibrate --rows-out (docs/TESTING.md)",
              file=sys.stderr)
    if missing or bare or stored:
        return 1
    print(f"check-argmax-floors: OK: {len(cells)} battery cells x {len(devices)} device(s) "
          f"({', '.join(devices) or 'none'}), every floor row present; maintained classes "
          f"{', '.join(maintained)} each calibrated; every battery floor row has main's stored rows "
          f"for the {BATTERY_WINDOW}-row window, from its receipt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
