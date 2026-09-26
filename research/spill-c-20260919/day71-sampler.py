#!/usr/bin/env python3
"""Day 71 sampler for cell `core` (research/spill-c-20260919/DAY71.md section 1): DAY70's sampler extended, log only,
one process, started and stopped by the cell, pinned by it to a CPU outside the run's pin and its hardware-thread
siblings. Rows are tab-separated with a UTC time first; one time per pass, shared by every row of the pass.

Every 250 ms:
- `C` every CPU's `/proc/stat` line;
- `O` each `run-gen` process's main thread: pid, name, last CPU, its `schedstat` (run ns, wait ns, slices), minor and
  major faults, user and system ticks;
- `I` every `/proc/interrupts` row that moved since the last pass: its name and `cpu:delta` for the CPUs that moved
  (`IH` rows, once, give each row's description);
- `S` the same for `/proc/softirqs`;
- `D` every CPU whose `cpuidle` states moved: `state:time_us_delta/usage_delta` for the states that moved (`DH` rows,
  once, give each state's name);
- `V` every `/proc/vmstat` counter that moved: `name:delta`;
- `E` every powercap zone's `energy_uj` (`EH` rows, once, give each zone's name, or that it cannot be read);
- `H` every hwmon temperature and power input: `chip/input=value`;
- `N` (DAY71 section 3) the first number of `/proc/stat`'s `intr` line (the host's interrupt total) and its `ctxt`.
Every 1 s:
- `T` every host thread whose user or system time moved since the last pass (DAY70's rows).

usage: day71-sampler.py <out.tsv>
"""
import glob
import os
import sys
import time
from datetime import datetime, timezone


def stamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return None


def stat_fields(text):
    """`/proc/.../stat`: (name, fields after the name); the name may hold spaces and parentheses."""
    left, right = text.find("("), text.rfind(")")
    return text[left + 1:right], text[right + 2:].split()


def run_gens():
    out = []
    for pid in os.listdir("/proc"):
        if pid.isdigit():
            comm = read(f"/proc/{pid}/comm")
            if comm and comm.startswith("run-gen"):
                out.append((pid, comm.strip()))
    return out


def per_cpu_table(text):
    """`/proc/interrupts` or `/proc/softirqs`: {row: ([count per CPU column], description)} and the CPU numbers."""
    lines = text.splitlines()
    cpus = [int(c[3:]) for c in lines[0].split()]
    rows = {}
    for line in lines[1:]:
        name, _, rest = line.partition(":")
        parts = rest.split()
        counts = []
        for p in parts[:len(cpus)]:
            if not p.isdigit():
                break
            counts.append(int(p))
        rows[name.strip()] = (counts, " ".join(parts[len(counts):]))
    return rows, cpus


class Deltas:
    """The per-CPU table rows that moved since the last pass, as `I`/`S` rows; each row's description once."""

    def __init__(self, path, tag):
        self.path, self.tag, self.last, self.described = path, tag, None, set()

    def rows(self, t):
        text = read(self.path)
        if not text:
            return []
        table, cpus = per_cpu_table(text)
        out = []
        for name, (counts, desc) in table.items():
            if name not in self.described:
                self.described.add(name)
                out.append(f"{t}\t{self.tag}H\t{name}\t{desc}")
        if self.last is not None:
            for name, (counts, _) in table.items():
                before = self.last.get(name, ([0] * len(counts), ""))[0]
                label = cpus if len(counts) == len(cpus) else ["all"] * len(counts)
                moved = [f"{label[i]}:{c - b}" for i, (c, b) in enumerate(zip(counts, before)) if c != b]
                if moved:
                    out.append(f"{t}\t{self.tag}\t{name}\t{','.join(moved)}")
        self.last = table
        return out


class Idle:
    """Every CPU's `cpuidle` state times (us) and usage: the moved ones as `D` rows, the names once as `DH` rows."""

    def __init__(self):
        self.states = {}
        for d in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpuidle/state[0-9]*")):
            cpu = int(d.split("/")[5][3:])
            self.states.setdefault(cpu, []).append((int(d.rsplit("state", 1)[1]), d))
        self.last, self.named = {}, False

    def rows(self, t):
        out = []
        if not self.named:
            self.named = True
            for cpu, states in sorted(self.states.items()):
                for k, d in states:
                    out.append(f"{t}\tDH\t{cpu}\t{k}\t{(read(d + '/name') or '?').strip()}")
        for cpu, states in sorted(self.states.items()):
            moved = []
            for k, d in states:
                tm, us = read(d + "/time"), read(d + "/usage")
                if tm is None or us is None:
                    continue
                now = (int(tm), int(us))
                before = self.last.get((cpu, k))
                self.last[(cpu, k)] = now
                if before is not None and now != before:
                    moved.append(f"{k}:{now[0] - before[0]}/{now[1] - before[1]}")
            if moved:
                out.append(f"{t}\tD\t{cpu}\t{','.join(moved)}")
        return out


class Vm:
    def __init__(self):
        self.last = None

    def rows(self, t):
        text = read("/proc/vmstat") or ""
        now = {}
        for line in text.splitlines():
            k, _, v = line.partition(" ")
            if v.strip().lstrip("-").isdigit():
                now[k] = int(v)
        out = []
        if self.last is not None:
            moved = [f"{k}:{v - self.last.get(k, v)}" for k, v in now.items() if v != self.last.get(k, v)]
            if moved:
                out.append(f"{t}\tV\t{','.join(moved)}")
        self.last = now
        return out


class Power:
    """Powercap zones' energy and hwmon inputs."""

    def __init__(self):
        self.zones = sorted(glob.glob("/sys/class/powercap/*/energy_uj"))
        self.inputs = []
        for chip in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
            name = (read(chip + "/name") or "?").strip()
            for f in sorted(glob.glob(chip + "/temp*_input") + glob.glob(chip + "/power*_input")):
                self.inputs.append((f"{os.path.basename(chip)}:{name}/{os.path.basename(f)}", f))
        self.named = False

    def rows(self, t):
        out = []
        if not self.named:
            self.named = True
            for z in self.zones:
                d = os.path.dirname(z)
                ok = "readable" if read(z) is not None else "unreadable"
                out.append(f"{t}\tEH\t{os.path.basename(d)}\t{(read(d + '/name') or '?').strip()}\t{ok}")
        for z in self.zones:
            v = read(z)
            if v is not None:
                out.append(f"{t}\tE\t{os.path.basename(os.path.dirname(z))}\t{v.strip()}")
        vals = []
        for label, f in self.inputs:
            v = read(f)
            if v is not None:
                vals.append(f"{label}={v.strip()}")
        if vals:
            out.append(f"{t}\tH\t{','.join(vals)}")
        return out


def main():
    out = open(sys.argv[1], "a", buffering=1)
    last = {}
    next_threads = 0.0
    irqs, softirqs = Deltas("/proc/interrupts", "I"), Deltas("/proc/softirqs", "S")
    idle, vm, power = Idle(), Vm(), Power()
    while True:
        now = time.monotonic()
        t = stamp()
        rows = []
        text = read("/proc/stat") or ""
        intr = ctxt = None
        for line in text.splitlines():
            if line.startswith("cpu") and line[3:4].isdigit():
                rows.append(f"{t}\tC\t{line}")
            elif line.startswith("intr "):
                intr = line.split()[1]
            elif line.startswith("ctxt "):
                ctxt = line.split()[1]
        if intr is not None and ctxt is not None:
            rows.append(f"{t}\tN\t{intr}\t{ctxt}")
        for pid, comm in run_gens():
            sched = read(f"/proc/{pid}/task/{pid}/schedstat")
            st = read(f"/proc/{pid}/task/{pid}/stat")
            if sched and st:
                _, f = stat_fields(st)
                rows.append(f"{t}\tO\t{pid}\t{comm}\t{f[36]}\t{sched.strip()}\t{f[7]}\t{f[9]}\t{f[11]}\t{f[12]}")
        for source in (irqs, softirqs, idle, vm, power):
            try:
                rows.extend(source.rows(t))
            except (OSError, ValueError, IndexError) as e:
                rows.append(f"{t}\tX\t{type(source).__name__}\t{e!r}")
        if now >= next_threads:
            next_threads = now + 1.0
            seen = {}
            for pid in os.listdir("/proc"):
                if not pid.isdigit():
                    continue
                try:
                    tids = os.listdir(f"/proc/{pid}/task")
                except OSError:
                    continue
                for tid in tids:
                    st = read(f"/proc/{pid}/task/{tid}/stat")
                    if not st:
                        continue
                    name, fields = stat_fields(st)
                    ticks = int(fields[11]) + int(fields[12])
                    key = (pid, tid)
                    seen[key] = ticks
                    moved = ticks - last.get(key, ticks)
                    if moved > 0:
                        rows.append(f"{t}\tT\t{pid}\t{tid}\t{name}\t{fields[36]}\t{moved}")
            last = seen
        out.write("\n".join(rows) + "\n")
        time.sleep(max(0.0, 0.25 - (time.monotonic() - now)))


if __name__ == "__main__":
    main()
