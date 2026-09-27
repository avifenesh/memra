#!/usr/bin/env python3
"""Day 36 receipt census (lane C, no card): the commands behind the numbers added to the packet on day 36.

Usage: python3 day36-receipt-census.py <research/spill-a-20260919 dir> <research/spill-lead-20260919/integration-day12 dir>

Prints, for each of A's day-28 and day-29 target-card sittings: the owner-thread ledger segments over every
`demote digests landed off the tick` line of the 10 ON boots of the double-park cell (N per figure printed), the
collector regimes (samples, temperature, power, memory) of the double-park, gates and unit-cell holds, and the
replay counts, and each ON boot's first ledger line checked against the promote runs' receipt lines (A's N=100
set); then the two identity default-ON ledger lines and parked counts per card; then, per gate log of A's
day-28 and day-29 sittings on both cards and of integ45's RTX 5090 cells, the last verdict line verbatim, the `ok:`
count (`(^|[^a-z])ok: `, the packet's appendix-A rule), the `FAIL` count, the fault gate's cell names and the hit
gate's door-line census (the fault gate's ok/FAIL per cell too); last, the trees the receipts name. Every figure is read from the named receipt file; nothing is compared across cards.
"""
import csv, glob, re, statistics as st, sys

LEDGER = re.compile(
    r"hashed in ([\d.]+)ms on the hash helper, landed after (\d+) poll\(s\).*?pre-submit ([\d.]+), copy settle "
    r"([\d.]+) over (\d+) poll\(s\), hashing polls ([\d.]+), take-back bind and publish ([\d.]+); owner in-completion "
    r"([\d.]+)ms; wall ([\d.]+)ms t0 to publication(?:; (\d+) hit\(s\) parked on the Hashing entry \((\d+) re-park\(s\)\))?"
)
PAYLOADS = re.compile(r"(\d+) payloads \(([\d.]+)MB\)")


def stat(name, v, nd=2):
    if not v:
        return f"{name} N=0"
    return f"{name} N={len(v)} median={st.median(v):.{nd}f} min={min(v):.{nd}f} max={max(v):.{nd}f}"


def ledger(root, day):
    files = sorted(glob.glob(f"{root}/{day}/box/double-park/ev/o*/b*-on/server.log"))
    cols = {k: [] for k in ("hashed", "polls", "presubmit", "settle", "hpolls", "takeback", "owner", "wall")}
    first3, rest, parked, reparks, payloads = [], [], 0, 0, set()
    per_boot = []
    for f in files:
        ps = []
        for line in open(f, errors="replace"):
            if "demote digests landed off the tick" not in line:
                continue
            m = LEDGER.search(line)
            if not m:
                print(f"NOMATCH {f}: {line[:120]}")
                continue
            h, polls, pre, settle, _cp, hp, tb, own, wall, hits, rp = m.groups()
            cols["hashed"].append(float(h)); cols["polls"].append(int(polls)); cols["presubmit"].append(float(pre))
            cols["settle"].append(float(settle)); cols["hpolls"].append(float(hp)); cols["takeback"].append(float(tb))
            cols["owner"].append(float(own)); cols["wall"].append(float(wall))
            parked += int(hits or 0); reparks += int(rp or 0)
            pm = PAYLOADS.search(line); payloads.add(f"{pm.group(1)} payloads ({pm.group(2)}MB)")
            ps.append(float(pre))
        per_boot.append((f.split("/")[-2], len(ps)))
        first3 += ps[:3]; rest += ps[3:]
    print(f"{day} LEDGER ON_boots={len(files)} ledger_lines={len(cols['owner'])} per_boot={per_boot}")
    print(f"{day} LEDGER {stat('owner_in_completion', cols['owner'])}")
    print(f"{day} LEDGER {stat('pre_submit', cols['presubmit'])} | {stat('first_three_per_boot', first3)} | {stat('demotes_4_to_11_per_boot', rest)}")
    print(f"{day} LEDGER {stat('copy_settle', cols['settle'])} | {stat('hashing_polls', cols['hpolls'])} | {stat('take_back_bind_publish', cols['takeback'])}")
    print(f"{day} LEDGER {stat('helper_hashed_in', cols['hashed'])} | {stat('landed_after_polls', cols['polls'], 0)} | {stat('wall_t0_to_publication', cols['wall'], 1)}")
    print(f"{day} LEDGER payloads={sorted(payloads)} hits_parked_total={parked} reparks_total={reparks}")


def regime(root, day):
    for cell in ("double-park", "gates", "unit-cell"):
        p = f"{root}/{day}/box/{cell}/command.gpu.csv"
        rows = list(csv.reader(open(p)))
        hdr = [h.strip() for h in rows[0]]; body = rows[1:]

        def col(n):
            i = hdr.index(n)
            v = []
            for r in body:
                try:
                    v.append(float(r[i].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
            return v
        t, w, mem = col("temperature.gpu"), col("power.draw [W]"), col("memory.used [MiB]")
        print(f"{day} REGIME {cell} samples={len(body)} {body[0][0].strip()}..{body[-1][0].strip()} "
              f"temp={min(t):.0f}..{max(t):.0f}C power={min(w):.2f}..{max(w):.2f}W mem_max={max(mem):.0f}MiB")
    rl = f"{root}/{day}/box/double-park/ev/replays.log"
    txt = open(rl, errors="replace").read()
    print(f"{day} REPLAYS stall_replay_pass={txt.count('STALL REPLAY: PASS')} errors0={txt.count('errors=0')} "
          f"lock_retries_files={len(glob.glob(f'{root}/{day}/box/*/lock-retries.log'))}")


def identity(root, label, path):
    lines = open(path, errors="replace").read().splitlines()
    parked = [l for l in lines if "hit parked on a Hashing entry" in l]
    ledg = [l for l in lines if "demote digests landed off the tick" in l]
    speck = [l for l in lines if "[spec-k] model=" in l]
    cached = [re.search(r"cached=(\d+) lcp=(\d+)", l).groups() for l in speck]
    print(f"{label} IDENTITY-DEFAULT-ON parked_lines={len(parked)} ledger_lines={len(ledg)} spec_k_cached_lcp={cached}")
    for l in parked + ledg:
        print(f"{label}   {l.strip()}")


SEQ = re.compile(r"demote digests landed off the tick: ticket seq=(\d+),")


def first_demote(root, day):
    """Per ON boot: the first ledger line's seq and pre-submit, and whether any promote run's `server_log_lines` in
    `promote/receipt.json` (the lines A's day-28 reader takes its N=100 from) carries that seq's ledger line."""
    import json
    files = sorted(glob.glob(f"{root}/{day}/box/double-park/ev/o*/b*-on/server.log"))
    rows, outside = [], 0
    for f in files:
        led = [l for l in open(f, errors="replace") if "demote digests landed off the tick" in l]
        in_runs = set()
        rec = json.load(open(f.replace("server.log", "promote/receipt.json")))
        for run in rec["runs"]:
            if run["arm"] != "promote":
                continue
            for l in run.get("server_log_lines", []):
                m = SEQ.search(l)
                if m:
                    in_runs.add(int(m.group(1)))
        seq = int(SEQ.search(led[0]).group(1)); pre = float(LEDGER.search(led[0]).group(3))
        inside = seq in in_runs
        outside += 0 if inside else 1
        rows.append(f"{f.split('/')[-2]}:seq={seq},pre={pre:.2f},in_promote_runs={inside},in_run_ledgers={len(in_runs)}")
    pres = [float(r.split("pre=")[1].split(",")[0]) for r in rows]
    print(f"{day} FIRST-DEMOTE boots={len(rows)} outside_promote_runs={outside} {stat('first_pre_submit', pres)}")
    for r in rows:
        print(f"{day}   {r}")


VERDICT = re.compile(r"GATE: |-> PASS|-> FAIL|FAILURE\(S\)")
OK = re.compile(r"(^|[^a-z])ok: ")
FAIL = re.compile(r"(^|[^a-z])FAIL: ")


def gates(label, paths):
    for p in paths:
        lines = open(p, errors="replace").read().splitlines()
        verdict = [l.strip() for l in lines if VERDICT.search(l)]
        oks = sum(1 for l in lines if OK.search(l))
        fails = sum(1 for l in lines if FAIL.search(l))
        cells = [re.match(r"== cell ([\w-]+):", l).group(1) for l in lines if re.match(r"== cell [\w-]+:", l)]
        per_cell, cur = {}, None
        for l in lines:
            m = re.match(r"== cell ([\w-]+):", l)
            if m:
                cur = m.group(1); per_cell.setdefault(cur, [0, 0]); continue
            if cur is not None and OK.search(l):
                per_cell[cur][0] += 1
            if cur is not None and FAIL.search(l):
                per_cell[cur][1] += 1
        doors = [l.strip() for l in lines if "door lines:" in l]
        route = [l.strip() for l in lines if "route submission(s)" in l]
        print(f"{label} GATE {p} ok={oks} fail={fails} "
              f"verdict={verdict[-1] if verdict else 'NONE'!r}")
        if cells:
            print(f"{label}   cells={len(cells)} {cells}")
            print(f"{label}   per_cell ok/fail (from the cell header to the next) " +
                  " ".join(f"{c}={o}/{x}" for c, (o, x) in per_cell.items()))
        for l in doors + route:
            print(f"{label}   {l}")


def provenance(root, integ):
    """The trees the receipts name: A's `box/tree.sha` per target sitting, the RTX 5090 battery's `start ... tree=`
    lines, and the tree lines in integ45's RTX 5090 gate receipts."""
    for day in ("pro-single-day28", "pro-single-day29"):
        print(f"TREE {day} box/tree.sha={open(f'{root}/{day}/box/tree.sha').read().strip()[:9]}")
    for day in ("rtx5090-day28", "rtx5090-day29"):
        for l in open(f"{root}/{day}/battery.log", errors="replace"):
            m = re.search(r"start ([a-z0-9-]+) tree=([0-9a-f]{9})", l)
            if m:
                print(f"TREE {day} {m.group(1)} tree={m.group(2)}")
    for p in sorted(glob.glob(f"{integ}/integ45-*-gate-5090/**/*", recursive=True)):
        try:
            for l in open(p, errors="replace"):
                m = re.search(r"tree=([0-9a-f]{9})", l)
                if m:
                    print(f"TREE integ45 {p.split('integration-day12/')[1]} tree={m.group(1)}")
        except (IsADirectoryError, UnicodeDecodeError):
            pass


if __name__ == "__main__":
    root = sys.argv[1].rstrip("/")
    for day in ("pro-single-day28", "pro-single-day29"):
        ledger(root, day); first_demote(root, day); regime(root, day)
    identity(root, "TARGET", f"{root}/pro-single-day29/box/gates/identity-default-on/host-on-server.log")
    identity(root, "RTX5090", f"{root}/rtx5090-day29/identity-default-on/ev/host-on-server.log")
    for day in ("pro-single-day28", "pro-single-day29"):
        gates(f"TARGET {day}", sorted(p for p in glob.glob(f"{root}/{day}/box/gates/*.log") if "/command." not in p))
    for day in ("rtx5090-day28", "rtx5090-day29"):
        gates(f"RTX5090 {day}", sorted(glob.glob(f"{root}/{day}/*/gate.log")))
    integ = sys.argv[2].rstrip("/")
    gates("RTX5090 integ45", sorted(glob.glob(f"{integ}/integ45-*-gate-5090/*/gate.log")))
    provenance(root, integ)
