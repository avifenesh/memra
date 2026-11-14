#!/usr/bin/env python3
"""memra#1258 capacity A/B summary: main (dmain) vs step 2 (ds2), the freed-to-pool variant (ds2r),
the parked variant before the fixes (ds2q; on the 5090 the ds2q dirs hold the fixed build) and the
fixed variant (ds2qf: trim fix + calibration fix), and ds2h (ds2qf + the park floor: the workspace
is parked only while the device holds admission's driver floor) on the 202e1411fe base, at ~5.7k and ~8k tokens beside a decoding keeper.
Per arm and length: chunk-1 demand per prompt across the two spread boots (determinism), the
first gate pass's charge (cost, reserve, required), the boot calibration's rest-mapped bytes,
and (5090 only) the margin sweeps' seen margins against target, spec/plain admissions and OOMs
(warm = margins 0 up to 150, two reps; trim = the same with the pool drained before each prime;
desc = one rep from 150 down to 0, so the first rows run on the boot's own charge).
usage: cap-summary.py [5090|gb10]"""
import re
import sys
from pathlib import Path

B = Path.home() / ".cache/memra-1258"
RE_ROW = re.compile(r"^i (\d+): status (\d+) prompt (\d+) path (\S+) need (\d+) growth (\d+).*calls (\[.*\]) trims (\d+) oom (\d+)")
RE_SW = re.compile(r"^rep (\d+) m(\d+): status (\d+) path (\S+) seen (-?\d+|None) eager \S+ oom (\d+)")
RE_REST = re.compile(r"rest-mapped (\d+)MB")
RE_PASS = re.compile(r"\[diag1255\] gate-pass id=\S+ prompt=(\d+) path=(\w+) cost=(\d+)MB draft=(\d+)MB reserve=(\d+)MB required=(\d+)MB")


def spread_dirs(where: str, arm: str, length: str) -> list[Path]:
    if where == "gb10":
        return sorted(B.glob(f"cap-gb10/spread-{arm}-{length}-*"))
    out = []
    for d in sorted(B.glob(f"spread/{arm}-*")):
        logs = sorted(d.glob("b*-mode*.log"))
        rows = [m for l in logs[0].read_text(errors="replace").splitlines() if (m := RE_ROW.match(l))] if logs else []
        if rows and ((length == "5.7k") == (int(rows[0].group(3)) < 7000)):
            out.append(d)
    return out


def spread_line(d: Path) -> str:
    boots, passes, rests = [], [], set()
    for log in sorted(d.glob("b*-mode*.log")):
        rows = [m for l in log.read_text(errors="replace").splitlines() if (m := RE_ROW.match(l))]
        boots.append({int(m.group(3)): (int(m.group(5)), int(m.group(9)), m.group(4)) for m in rows})
        serve = log.with_suffix("") / "serve.log"
        if serve.exists():
            text = serve.read_text(errors="replace")
            rests.update(RE_REST.findall(text))
            passes += [m for l in text.splitlines() if (m := RE_PASS.search(l)) and int(m.group(1)) > 1000]
    needs = [v[0] for b in boots for v in b.values()]
    if not needs:
        return f"{d.name}: no rows"
    same = len(boots) >= 2 and all(boots[0][k][0] == boots[1].get(k, (None,))[0] for k in boots[0])
    steady = [v[0] for b in boots for i, v in enumerate(b.values()) if i >= 2]
    ooms = sum(v[1] for b in boots for v in b.values())
    paths = sorted({v[2] for b in boots for v in b.values()})
    p = passes[0] if passes else None
    charge = f"cost {p.group(3)} reserve {p.group(5)} required {p.group(6)} MB" if p else "no gate line"
    return (f"chunk-1 need {min(needs)}-{max(needs)} MiB (from the 3rd prime {min(steady)}-{max(steady)}), "
            f"identical across boots {same}, oom {ooms}, paths {paths}; {charge}; rest-mapped {sorted(rests)} ({d.name})")


def sweep_line(name: str) -> str:
    f = B / f"sweep/{name}/sweep.log"
    if not f.exists():
        return f"{name}: missing"
    rows = [m for l in f.read_text(errors="replace").splitlines() if (m := RE_SW.match(l))]
    seen = [f"m{m.group(2)}:{m.group(5)}{'P' if m.group(4) == 'plain' else ''}" for m in rows]
    spec = sum(1 for m in rows if m.group(4) == "spec")
    ooms = sum(int(m.group(6)) for m in rows)
    bad = sum(1 for m in rows if m.group(3) != "200")
    serve = B / f"sweep/{name}/serve.log"
    text = serve.read_text(errors="replace") if serve.exists() else ""
    rest = sorted(set(RE_REST.findall(text)))
    req = sorted({int(m.group(6)) for l in text.splitlines() if (m := RE_PASS.search(l)) and m.group(2) == "spec"})
    unparked = text.count("GDN scan workspace not parked")
    return (f"{name}: rows {len(rows)} spec {spec} plain {len(rows) - spec} oom {ooms} non200 {bad} not-parked {unparked} "
            f"rest-mapped {rest} spec-required {req[:1]}..{req[-1:]} seen {' '.join(seen)}")


def main() -> None:
    where = sys.argv[1] if len(sys.argv) > 1 else "5090"
    for length in ("5.7k", "8k"):
        print(f"== {where} {length}")
        for arm in ("dmain", "ds2", "ds2r", "ds2q", "ds2qf", "ds2h", "dmainh", "ds2qfh"):
            print(f"{arm}:")
            for d in spread_dirs(where, arm, length)[-1:]:
                print("   spread " + spread_line(d))
            if where == "5090":
                for mode in ("warm", "trim", "desc"):
                    if (B / f"sweep/cap-{arm}-{mode}-{length}/sweep.log").exists():
                        print("   " + sweep_line(f"cap-{arm}-{mode}-{length}"))


if __name__ == "__main__":
    main()
