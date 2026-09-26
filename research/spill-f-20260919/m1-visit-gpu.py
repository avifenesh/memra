#!/usr/bin/env python3
"""Post-hoc per-visit GPU summary for recorded cells (telemetry amendment; recording only).

Usage: m1-visit-gpu.py <regime dir> [...]  -> <regime dir>/visits-gpu.json per directory

A regime dir holds either one collector cell (`command.gpu.csv`, `command.capture.json`, `visits/`)
or 5090 round cells (`round-KK/` each with its own). Collector timestamps are host-local; the
host's UTC offset is taken from the cell's `started_utc` against the CSV's first sample (rounded to
the quarter hour), so BOX27 and 5090 receipts both slice correctly. Each visit's window is its
`utc_start` plus `wall_ns`.
"""
import datetime
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("gpusampler", HERE / "m1-gpu-sampler.py")
GPU = importlib.util.module_from_spec(spec)
spec.loader.exec_module(GPU)


def cells(d):
    if (d / "command.gpu.csv").exists():
        yield d
    for c in sorted(d.glob("round-[0-9][0-9]")):
        if (c / "command.gpu.csv").exists():
            yield c


def offset_s(cell):
    cap = json.loads((cell / "command.capture.json").read_text())
    started = datetime.datetime.fromisoformat(cap["started_utc"]).timestamp()
    first = (cell / "command.gpu.csv").read_text().splitlines()[1].split(",")[0].strip()
    naive = datetime.datetime.strptime(first, "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
    return round((naive - started) / 900) * 900


def main():
    for arg in sys.argv[1:]:
        d = Path(arg)
        out = {}
        for cell in cells(d):
            off = offset_s(cell)
            local = datetime.datetime.now().astimezone().utcoffset().total_seconds()
            for vj in sorted(cell.glob("visits/r*/visit.json")):
                v = json.loads(vj.read_text())
                t0 = datetime.datetime.fromisoformat(v["utc_start"]).timestamp()
                t1 = t0 + v["wall_ns"] / 1e9
                # summarize_collector parses stamps as this machine's local time: shift the window.
                shift = off - local
                out[f"{cell.name}/{vj.parent.name}"] = {"arm": v["arm"], "round": v["round"], "tok_s": v.get("tok_s"),
                                                         **GPU.summarize_collector(cell / "command.gpu.csv",
                                                                                   t0 + shift, t1 + shift)}
        (d / "visits-gpu.json").write_text(json.dumps(out, indent=1) + "\n")
        print(f"M1-VISIT-GPU {d} visits={len(out)}")


if __name__ == "__main__":
    main()
