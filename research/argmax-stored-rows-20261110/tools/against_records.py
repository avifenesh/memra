#!/usr/bin/env python3
"""against_records.py: each stored table against the 12-row gate tables of the cell-evidence
records of its device class (research/release-qualification/cells/<commit>-<class>-*/battery/).

For every tools/argmax-margin-rows/*.rows file: the records whose battery ran the same model and
decode program on the same device class, and whether their 12-row table equals the stored rows
byte for byte (trailing whitespace dropped), and how many distinct devices of that name (by
the record's uuid_sha256) the records come from. Prints a Markdown table; exit 0 always (a record
from a build that moved the numbers differs legitimately: the table says which).
"""
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
CLASS = {"NVIDIA GeForce RTX 5090 Laptop GPU": "rtx5090", "NVIDIA GB10": "gb10",
         "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition": "rtxpro6000"}


def table_lines(text):
    out = []
    for line in text.splitlines():
        f = line.split()
        if len(f) >= 7 and f[0].isdigit() and f[1].isdigit():
            out.append(line.rstrip())
    return out


def stored(path):
    head, _, body = path.read_text().partition("\nrows\n")
    fields = dict(line.split("\t", 1) for line in head.splitlines() if "\t" in line and not line.startswith("#"))
    return fields, table_lines(body)


def main():
    print("| stored table | records of the device with this cell | identical | devices (distinct uuid) | differ (record commits) |")
    print("|---|---|---|---|---|")
    for path in sorted((REPO / "tools/argmax-margin-rows").glob("*.rows")):
        h, rows = stored(path)
        cls = CLASS.get(h["device"])
        suffix = ".batch.log" if h["decode"] == "batch" else ".log"
        name = f"gate-{h['model']}-{h['prompt']}{suffix}"
        same, differ, units = [], [], set()
        for rec in sorted((REPO / "research/release-qualification/cells").glob(f"*-{cls}-*")):
            # A record's device must be the stored table's exact device.
            if not (rec / "record.json").is_file():
                continue
            device = json.loads((rec / "record.json").read_text()).get("device", {})
            if device.get("name") != h["device"]:
                continue
            for log in sorted(rec.glob(f"battery/argmax*/{name}")):
                if log.name.endswith(".batch.log") != (h["decode"] == "batch"):
                    continue
                lines = table_lines(log.read_text(errors="replace"))
                if len(lines) != len(rows):
                    continue
                (same if lines == rows else differ).append(rec.name.split("-")[0])
                units.add(device.get("uuid_sha256"))
        print(f"| `{path.name}` | {len(same) + len(differ)} | {len(same)} | {len(units)} | "
              f"{', '.join(sorted(set(differ))) or 'none'} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
