#!/usr/bin/env python3
"""Day 78 page census for cell `pages` (research/spill-c-20260919/DAY78.md section 1): log only, run by the cell as a
separate process pinned outside the run's CPUs. It reads one process's mappings twice, `--gap` seconds apart, and
writes per pass and per mapping: the range, permissions, name, size, resident and locked kB and huge pages from
`smaps`, and from `pagemap` the present, swapped, file-or-shared-anonymous and exclusively-mapped page counts; where
the kernel shows page frame numbers (only with CAP_SYS_ADMIN), the mapping's frame numbers hashed (so a mapping whose
frames changed between the passes had pages migrated), the count of frames that changed, and per-flag counts from
`/proc/kpageflags` for its frames. `access` rows record what the process could read.

usage: day78-pages.py <pid> <out.tsv> [--gap SECONDS]
"""
import hashlib
import os
import struct
import sys
import time
from datetime import datetime, timezone

PAGE = 4096
# /proc/kpageflags bits (Documentation/admin-guide/mm/pagemap.rst).
KPF = {0: "LOCKED", 4: "DIRTY", 5: "LRU", 6: "ACTIVE", 10: "BUDDY", 11: "MMAP", 12: "ANON", 15: "COMPOUND_HEAD",
       16: "COMPOUND_TAIL", 17: "HUGE", 18: "UNEVICTABLE", 22: "THP", 25: "OFFLINE", 26: "ZERO_PAGE", 33: "MLOCKED"}


# Translation tables: a byte maps to 1 when its bit `b` is set, else 0.
BIT = {b: bytes((x >> b) & 1 for x in range(256)) for b in range(8)}
SHOW_PFN = False


def stamp():
    return datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]


def cap_eff():
    for line in open("/proc/self/status"):
        if line.startswith("CapEff:"):
            return int(line.split()[1], 16)
    return 0


def mappings(pid):
    """(start, end, perms, name, smaps fields) per mapping, from /proc/<pid>/smaps."""
    out, cur = [], None
    with open(f"/proc/{pid}/smaps") as f:
        for line in f:
            head = line.split()
            if "-" in head[0] and len(head) >= 5 and ":" not in head[0]:
                start, end = (int(x, 16) for x in head[0].split("-"))
                cur = [start, end, head[1], " ".join(head[5:]) if len(head) > 5 else "[anon]", {}]
                out.append(cur)
            elif cur is not None and head and head[0].endswith(":") and len(head) >= 2:
                key = head[0][:-1]
                if key in ("Rss", "Locked", "AnonHugePages", "Anonymous", "Swap", "Shared_Hugetlb", "Private_Hugetlb"):
                    cur[4][key] = int(head[1])
                elif key == "VmFlags":
                    cur[4]["VmFlags"] = " ".join(head[1:])
    return out


def census(pid, flags_fd):
    rows = []
    pm = os.open(f"/proc/{pid}/pagemap", os.O_RDONLY)
    try:
        for start, end, perms, name, fields in mappings(pid):
            n = (end - start) // PAGE
            present = swapped = filemap = exclusive = 0
            pfns = []
            off = (start // PAGE) * 8
            # A mapping with nothing resident (the driver's large address reservations) is not scanned.
            done = 0 if fields.get("Rss", 0) or fields.get("Swap", 0) else n
            while done < n:
                take = min(n - done, 1 << 18)
                try:
                    raw = os.pread(pm, take * 8, off + done * 8)
                except OSError:
                    break
                # The flag bits all sit in each entry's top byte: count them at C speed.
                top = raw[7::8]
                present += top.translate(BIT[7]).count(1)
                swapped += top.translate(BIT[6]).count(1)
                filemap += top.translate(BIT[5]).count(1)
                exclusive += top.translate(BIT[0]).count(1)
                if SHOW_PFN:
                    for (v,) in struct.iter_unpack("<Q", raw):
                        pfn = v & ((1 << 55) - 1)
                        if v >> 63 & 1 and pfn:
                            pfns.append(pfn)
                done += take
            flagcount = {}
            if pfns and flags_fd is not None:
                for pfn in pfns:
                    try:
                        (fl,) = struct.unpack("<Q", os.pread(flags_fd, 8, pfn * 8))
                    except OSError:
                        break
                    for bit, label in KPF.items():
                        if fl >> bit & 1:
                            flagcount[label] = flagcount.get(label, 0) + 1
            rows.append((start, end, perms, name, n, present, swapped, filemap, exclusive, fields, pfns, flagcount))
    finally:
        os.close(pm)
    return rows


def main():
    pid, out = sys.argv[1], sys.argv[2]
    gap = float(sys.argv[sys.argv.index("--gap") + 1]) if "--gap" in sys.argv else 0.5
    w = open(out, "a", buffering=1)
    global SHOW_PFN
    cap = cap_eff()
    # Frame numbers read nonzero only with CAP_SYS_ADMIN; without it, skip the per-entry pass.
    SHOW_PFN = bool(cap >> 21 & 1)
    try:
        flags_fd = os.open("/proc/kpageflags", os.O_RDONLY)
        os.pread(flags_fd, 8, 0)
        kpf = "readable"
    except OSError as e:
        flags_fd, kpf = None, f"unreadable ({e.strerror})"
    w.write(f"{stamp()}\taccess\tpid={pid}\tCapEff={cap:#x}\tcap_sys_admin={'yes' if cap >> 21 & 1 else 'no'}"
            f"\tkpageflags={kpf}\tpage_owner={'present' if os.path.exists('/sys/kernel/debug/page_owner') else 'absent'}"
            f"\ttracefs={'present' if os.path.isdir('/sys/kernel/tracing/events/compaction') else 'absent'}\n")
    passes = []
    for n in range(2):
        t = stamp()
        rows = census(pid, flags_fd)
        passes.append(rows)
        pfn_rows = sum(1 for r in rows if r[10])
        w.write(f"{t}\tpass\t{n}\tmappings={len(rows)}\tmappings_with_pfns={pfn_rows}\n")
        for (start, end, perms, name, pages, present, swapped, filemap, exclusive, fields, pfns, flagcount) in rows:
            digest = hashlib.sha256(struct.pack(f"<{len(pfns)}Q", *pfns)).hexdigest()[:16] if pfns else "-"
            w.write(f"{t}\tmap\t{n}\t{start:x}-{end:x}\t{perms}\t{name}\tpages={pages}\tpresent={present}"
                    f"\tswapped={swapped}\tfile_or_shared={filemap}\texclusive={exclusive}"
                    f"\trss_kb={fields.get('Rss', 0)}\tlocked_kb={fields.get('Locked', 0)}"
                    f"\tanon_kb={fields.get('Anonymous', 0)}\tthp_kb={fields.get('AnonHugePages', 0)}"
                    f"\tvmflags={fields.get('VmFlags', '')}\tpfn_sha={digest}"
                    f"\tflags={','.join(f'{k}:{v}' for k, v in sorted(flagcount.items())) or '-'}\n")
        if n == 0:
            time.sleep(gap)
    # Frames that changed between the passes, per mapping (only where frame numbers are visible).
    first = {(r[0], r[1]): r[10] for r in passes[0]}
    for r in passes[1]:
        before = first.get((r[0], r[1]))
        if before and r[10]:
            moved = sum(1 for a, b in zip(before, r[10]) if a != b) + abs(len(before) - len(r[10]))
            w.write(f"{stamp()}\tmoved\t{r[0]:x}-{r[1]:x}\t{r[3]}\tchanged_frames={moved}\n")
    w.write(f"{stamp()}\tdone\n")


if __name__ == "__main__":
    main()
