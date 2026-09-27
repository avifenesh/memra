#!/usr/bin/env python3
"""M1 proof: is a writable directory on physical local NVMe? (M1-PREREG.md section A)

Fail-closed chain, standard library only:

  A1 kernel      not a guest: no `hypervisor` CPU flag, DMI not a virtual platform, no
                 /sys/hypervisor type, no emulated platform device on the PCI bus
  A2 path        existing, writable directory
  A3 mount       statx dev + mount id == mountinfo entry == fdinfo mount id; ext4 or xfs only
  A4 block graph /sys/dev/block/<maj:min> -> partition parent -> slaves (dm, md) recursively;
                 every leaf an NVMe namespace (native multipath heads expanded to their paths)
  A5 controller  transport pcie, under /sys/devices/pci*, class 0x010802, vendor outside the
                 emulated set, non-empty non-emulator model, readable link fields
  A6 I/O binding lane-owned file: N bytes written O_DIRECT + fdatasync, read back O_DIRECT,
                 SHA-256 identity; leaf and top-device sector deltas each cover the payload
  A7 capacity    free space covers --reserve-bytes
  A8 identity    {device, filesystem_id, mount_id} exactly as tools/tier-battery.py produces

Only A6 writes, and only one lane-owned file inside --path, removed before exit. Nothing is
mounted, created in /dev, dropped from a cache, or run with privilege.

Exit 0 PASS (class nvme-local-direct), 3 FAIL (every failing reason listed), 2 refused usage.
"""
import argparse
import ctypes
import ctypes.util
import datetime
import hashlib
import json
import mmap
import os
import re
import struct
import sys
import time
from pathlib import Path

SCHEMA = "m1-nvme-proof-v1"
ADMITTED_FS = {"ext4", "xfs"}
NVME_CLASS = 0x010802
SECTOR = 512
CHUNK = 4 << 20

EMULATED_VENDORS = {
    0x1B36: "Red Hat/QEMU", 0x1AF4: "virtio", 0x15AD: "VMware", 0x1414: "Microsoft Hyper-V",
    0x1AE0: "Google virtual", 0x1D0F: "cloud virtual-device vendor (every host exposing it, VM or bare metal, fails)", 0x80EE: "VirtualBox", 0x1AB8: "Parallels",
    0x1234: "bochs/QEMU display",
}
# (vendor, device) pairs that only exist in emulated platforms.
EMULATED_PAIRS = {
    (0x8086, 0x5845): "QEMU legacy NVMe",
    (0x8086, 0x29C0): "Q35 host bridge", (0x8086, 0x2918): "ICH9 LPC (Q35)",
    (0x8086, 0x2922): "ICH9 AHCI (Q35)", (0x8086, 0x2930): "ICH9 SMBus (Q35)",
    (0x8086, 0x1237): "i440FX host bridge", (0x8086, 0x7000): "PIIX3 ISA",
    (0x8086, 0x7010): "PIIX3 IDE", (0x8086, 0x7113): "PIIX4 ACPI",
}
VIRTUAL_DMI = ("qemu", "kvm", "vmware", "virtualbox", "innotek", "xen", "bochs", "parallels",
               "bhyve", "openstack", "google compute engine", "virtual machine",
               "cloud hypervisor", "firecracker", "apple virtualization", "red hat")
EMULATOR_MODELS = ("qemu", "vmware", "virtual", "elastic block store", "nvme instance storage",
                   "google", "microsoft", "msft", "vbox", "bochs")
NON_NVME_LEAVES = (("vd", "virtio block"), ("xvd", "xen block"), ("sd", "SCSI/SATA disk"),
                   ("nbd", "network block device"), ("rbd", "ceph rbd"), ("zram", "zram"),
                   ("ram", "ram disk"), ("pmem", "persistent memory"), ("mmcblk", "MMC/SD"),
                   ("sr", "optical"), ("drbd", "drbd replicated"), ("md", "md without members"),
                   ("dm-", "dm without members"), ("loop", "loop device"))


class Env:
    """sysfs/proc roots; fixtures substitute directory trees with the same relative links."""

    def __init__(self, sys_root="/sys", proc_root="/proc"):
        self.sys_root = Path(sys_root).resolve()
        self.proc_root = Path(proc_root)

    def sys(self, rel):
        return self.sys_root / rel.lstrip("/")

    def proc(self, rel):
        return self.proc_root / rel.lstrip("/")

    def inside(self, p):
        try:
            p.relative_to(self.sys_root)
            return True
        except ValueError:
            return False

    def rel(self, p):
        return "/sys/" + str(p.relative_to(self.sys_root)) if self.inside(p) else str(p)


def read(p):
    try:
        return p.read_text().strip()
    except (OSError, UnicodeDecodeError):
        return None


def hexint(text):
    try:
        return int(text, 16)
    except (TypeError, ValueError):
        return None


def listdir(p):
    try:
        return sorted(os.listdir(p))
    except OSError:
        return []


# ---------------------------------------------------------------- A1 kernel

def check_kernel(env):
    reasons, rec = [], {}
    cpuinfo = read(env.proc("cpuinfo"))
    if cpuinfo is None:
        reasons.append("A1: /proc/cpuinfo unreadable")
    else:
        flags = set()
        for line in cpuinfo.splitlines():
            if line.startswith(("flags", "Features")):
                flags.update(line.split(":", 1)[-1].split())
        rec["cpu_flag_lines"] = bool(flags)
        rec["hypervisor_flag"] = "hypervisor" in flags
        if not flags:
            reasons.append("A1: no CPU flag line in /proc/cpuinfo")
        elif "hypervisor" in flags:
            reasons.append("A1: guest kernel (CPU flag `hypervisor` present)")
    dmi = {}
    for key in ("sys_vendor", "product_name", "board_vendor", "board_name"):
        dmi[key] = read(env.sys(f"class/dmi/id/{key}"))
    rec["dmi_private"] = dmi
    joined = " ".join(v.lower() for v in dmi.values() if v)
    hits = [w for w in VIRTUAL_DMI if w in joined]
    rec["dmi_virtual_markers"] = hits
    if hits:
        reasons.append(f"A1: guest kernel (DMI names a virtual platform: {', '.join(hits)})")
    hv = read(env.sys("hypervisor/type"))
    rec["sys_hypervisor_type"] = hv
    if hv:
        reasons.append(f"A1: guest kernel (/sys/hypervisor/type = {hv})")
    pci_root = env.sys("bus/pci/devices")
    devices = listdir(pci_root)
    rec["pci_functions"] = len(devices)
    emulated = []
    for d in devices:
        vendor = hexint(read(pci_root / d / "vendor"))
        device = hexint(read(pci_root / d / "device"))
        if vendor is None or device is None:
            emulated.append("a PCI function with unreadable vendor/device")
            continue
        if vendor in EMULATED_VENDORS:
            emulated.append(f"{vendor:04x}:{device:04x} {EMULATED_VENDORS[vendor]}")
        elif (vendor, device) in EMULATED_PAIRS:
            emulated.append(f"{vendor:04x}:{device:04x} {EMULATED_PAIRS[(vendor, device)]}")
    rec["pci_emulated_devices"] = sorted(set(emulated))
    if not devices:
        reasons.append("A1: /sys/bus/pci/devices empty or unreadable (platform census impossible)")
    if emulated:
        reasons.append("A1: emulated platform device(s) on the PCI bus: "
                       + "; ".join(sorted(set(emulated))[:8]))
    return rec, reasons


# ---------------------------------------------------------------- A3 mount

def parse_mountinfo(text):
    rows = []
    for line in text.splitlines():
        left, _, right = line.partition(" - ")
        f = left.split()
        r = right.split()
        if len(f) < 5 or len(r) < 2:
            continue
        unescape = lambda s: re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), s)
        rows.append({"mount_id": int(f[0]), "parent_id": int(f[1]), "major_minor": f[2],
                     "root": unescape(f[3]), "mount_point": unescape(f[4]),
                     "fstype": r[0], "source": unescape(r[1]),
                     "super_options": r[2] if len(r) > 2 else ""})
    return rows


def covering_mount(rows, real_path, mount_id):
    by_id = [r for r in rows if r["mount_id"] == mount_id]
    prefix = [r for r in rows
              if real_path == r["mount_point"]
              or real_path.startswith(r["mount_point"].rstrip("/") + "/")]
    longest = None
    for r in prefix:  # the later entry of equal length is the visible overmount
        if longest is None or len(r["mount_point"]) >= len(longest["mount_point"]):
            longest = r
    return (by_id[0] if by_id else None), longest


class Statx(ctypes.Structure):
    _fields_ = [("stx_mask", ctypes.c_uint32), ("stx_blksize", ctypes.c_uint32),
                ("stx_attributes", ctypes.c_uint64), ("stx_nlink", ctypes.c_uint32),
                ("stx_uid", ctypes.c_uint32), ("stx_gid", ctypes.c_uint32),
                ("stx_mode", ctypes.c_uint16), ("_spare0", ctypes.c_uint16),
                ("stx_ino", ctypes.c_uint64), ("stx_size", ctypes.c_uint64),
                ("stx_blocks", ctypes.c_uint64), ("stx_attributes_mask", ctypes.c_uint64),
                ("_times", ctypes.c_uint8 * 64),
                ("stx_rdev_major", ctypes.c_uint32), ("stx_rdev_minor", ctypes.c_uint32),
                ("stx_dev_major", ctypes.c_uint32), ("stx_dev_minor", ctypes.c_uint32),
                ("stx_mnt_id", ctypes.c_uint64), ("stx_dio_mem_align", ctypes.c_uint32),
                ("stx_dio_offset_align", ctypes.c_uint32), ("_spare3", ctypes.c_uint8 * 96)]


STATX_MNT_ID = 0x1000
STATX_DIOALIGN = 0x2000
AT_FDCWD = -100


def statx(path):
    libc = ctypes.CDLL(ctypes.util.find_library("c") or None, use_errno=True)
    fn = getattr(libc, "statx", None)
    if fn is None:
        return None, "libc has no statx"
    fn.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_uint, ctypes.POINTER(Statx)]
    fn.restype = ctypes.c_int
    buf = Statx()
    rc = fn(AT_FDCWD, os.fsencode(str(path)), 0, STATX_MNT_ID | STATX_DIOALIGN, ctypes.byref(buf))
    if rc != 0:
        e = ctypes.get_errno()
        return None, f"statx failed: {os.strerror(e)} (errno {e})"
    return {"mask": buf.stx_mask, "dev_major": buf.stx_dev_major, "dev_minor": buf.stx_dev_minor,
            "mnt_id": buf.stx_mnt_id if buf.stx_mask & STATX_MNT_ID else None,
            "dio_mem_align": buf.stx_dio_mem_align if buf.stx_mask & STATX_DIOALIGN else None,
            "dio_offset_align": buf.stx_dio_offset_align if buf.stx_mask & STATX_DIOALIGN else None}, None


def filesystem_identity(path):
    """Same shape and method as tools/tier-battery.py::filesystem_identity."""
    path = path.resolve(strict=True)
    st = path.stat()
    identity = {"device": st.st_dev, "filesystem_id": os.statvfs(path).f_fsid}
    fd = os.open(path, os.O_RDONLY)
    try:
        info = Path(f"/proc/self/fdinfo/{fd}")
        if info.exists():
            match = re.search(r"^mnt_id:\s*(\d+)$", info.read_text(), re.MULTILINE)
            if match:
                identity["mount_id"] = int(match[1])
    finally:
        os.close(fd)
    return identity


def check_path_and_mount(env, path):
    reasons, rec = [], {}
    try:
        real = path.resolve(strict=True)
    except OSError as e:
        return rec, [f"A2: path does not resolve: {e}"], None
    rec["realpath_private"] = str(real)
    if not real.is_dir():
        reasons.append("A2: path is not a directory")
    if not os.access(real, os.W_OK | os.X_OK):
        reasons.append("A2: path is not writable")
    st = real.stat()
    maj, mnr = os.major(st.st_dev), os.minor(st.st_dev)
    rec["st_dev"] = f"{maj}:{mnr}"
    sx, err = statx(real)
    rec["statx"] = sx
    if err:
        reasons.append(f"A3: {err}")
    identity = filesystem_identity(real)
    rec["identity"] = identity
    text = read(env.proc("self/mountinfo"))
    if text is None:
        reasons.append("A3: /proc/self/mountinfo unreadable")
        return rec, reasons, None
    rows = parse_mountinfo(text)
    by_id, longest = covering_mount(rows, str(real), identity.get("mount_id", -1))
    if "mount_id" not in identity:
        reasons.append("A3: fdinfo carries no mnt_id")
    if by_id is None:
        reasons.append("A3: fdinfo mount id has no mountinfo entry")
    if longest is None:
        reasons.append("A3: no mountinfo entry covers the path")
    if by_id and longest and by_id["mount_id"] != longest["mount_id"]:
        reasons.append(f"A3: fdinfo mount {by_id['mount_id']} != covering mount {longest['mount_id']}")
    if sx and sx["mnt_id"] is not None and identity.get("mount_id") not in (None, sx["mnt_id"]):
        reasons.append(f"A3: statx mnt_id {sx['mnt_id']} != fdinfo mnt_id {identity['mount_id']}")
    if sx and (sx["dev_major"], sx["dev_minor"]) != (maj, mnr):
        reasons.append("A3: statx device differs from stat device")
    mount = by_id or longest
    if mount:
        rec["fstype"] = mount["fstype"]
        rec["mount_private"] = mount
        if mount["major_minor"] != f"{maj}:{mnr}":
            reasons.append(f"A3: mountinfo major:minor {mount['major_minor']} != stat {maj}:{mnr}")
        if mount["fstype"] not in ADMITTED_FS:
            reasons.append(f"A3: filesystem type {mount['fstype']!r} is not block-traceable here "
                           f"(admitted: {', '.join(sorted(ADMITTED_FS))})")
    if maj == 0:
        reasons.append(f"A3: anonymous device {maj}:{mnr} (no block device behind this filesystem)")
    return rec, reasons, (maj, mnr)


# ---------------------------------------------------------------- A4 block graph

def trace_block(env, major, minor):
    reasons, nodes, leaves = [], [], []
    seen = set()

    def visit(p, via, depth):
        if depth > 32:
            reasons.append("A4: block graph deeper than 32 levels")
            return
        try:
            real = p.resolve(strict=True)
        except OSError:
            reasons.append(f"A4: block node {env.rel(p)} does not resolve")
            return
        if not env.inside(real):
            reasons.append(f"A4: sysfs link escapes the sysfs root: {real}")
            return
        if real in seen:
            return
        seen.add(real)
        name = real.name
        node = {"name": name, "via": via, "sysfs_private": env.rel(real)}
        dev = read(real / "dev")
        if dev is None:
            reasons.append(f"A4: block node {name} has no readable dev attribute")
            nodes.append(node)
            return
        node["major_minor"] = dev
        node["rotational"] = read(real / "queue" / "rotational")
        if (real / "partition").exists():
            node["kind"] = "partition"
            nodes.append(node)
            visit(real.parent, name, depth + 1)
            return
        if name.startswith("loop") or (real / "loop").is_dir():
            node["kind"] = "loop"
            nodes.append(node)
            reasons.append(f"A4: {name} is a loop device (a loop layer is not the spill path's "
                           "native storage, whatever backs it)")
            return
        slaves = listdir(real / "slaves")
        if slaves:
            if (real / "dm").is_dir():
                node["kind"] = "dm"
                uuid = read(real / "dm" / "uuid") or ""
                node["dm_uuid_prefix"] = uuid.split("-", 1)[0] if "-" in uuid else uuid[:8]
                node["dm_name_private"] = read(real / "dm" / "name")
            elif (real / "md").is_dir():
                node["kind"] = "md"
                node["md_level"] = read(real / "md" / "level")
            else:
                node["kind"] = "stacked"
            node["members"] = slaves
            nodes.append(node)
            for s in slaves:
                visit(real / "slaves" / s, name, depth + 1)
            return
        if re.fullmatch(r"nvme\d+n\d+", name) or re.fullmatch(r"nvme\d+c\d+n\d+", name):
            paths = listdir(real / "multipath")
            if paths:
                node["kind"] = "nvme-multipath-head"
                node["paths"] = paths
                nodes.append(node)
                leaves.append({"name": name, "stat_node_private": env.rel(real),
                               "controllers_private": [env.rel((real / "multipath" / q).resolve().parent)
                                                       for q in paths]})
                return
            node["kind"] = "nvme-namespace"
            nodes.append(node)
            if "/devices/virtual/" in env.rel(real):
                reasons.append(f"A4: NVMe namespace {name} sits under /sys/devices/virtual "
                               "without multipath paths")
                return
            leaves.append({"name": name, "stat_node_private": env.rel(real),
                           "controllers_private": [env.rel(real.parent)]})
            return
        node["kind"] = "leaf-other"
        nodes.append(node)
        label = next((desc for pre, desc in NON_NVME_LEAVES if name.startswith(pre)),
                     "virtual device" if "/devices/virtual/" in env.rel(real) else "unknown class")
        reasons.append(f"A4: leaf {name} is not an NVMe namespace ({label})")

    visit(env.sys(f"dev/block/{major}:{minor}"), "mount", 0)
    if not leaves and not reasons:
        reasons.append("A4: no NVMe leaf found")
    return {"nodes": nodes, "leaves": leaves}, reasons


# ---------------------------------------------------------------- A5 controller

def check_controllers(env, leaves):
    reasons, ctrls = [], []
    seen = set()
    for leaf in leaves:
        for c in leaf["controllers_private"]:
            if c in seen:
                continue
            seen.add(c)
            cdir = env.sys(c[len("/sys/"):]) if c.startswith("/sys/") else Path(c)
            name = cdir.name
            rec = {"controller": name, "leaf": leaf["name"]}
            transport = read(cdir / "transport")
            model = read(cdir / "model")
            rec.update(transport=transport, model=model, firmware_rev=read(cdir / "firmware_rev"))
            serial = read(cdir / "serial")
            rec["serial_sha256_16"] = hashlib.sha256(serial.encode()).hexdigest()[:16] if serial else None
            rec["serial_private"] = serial
            if transport != "pcie":
                reasons.append(f"A5: {name} transport is {transport!r}, not 'pcie' (NVMe-oF or unknown)")
            if not model:
                reasons.append(f"A5: {name} model string empty or unreadable")
            elif any(w in model.lower() for w in EMULATOR_MODELS):
                reasons.append(f"A5: {name} model {model!r} names an emulator or virtual disk")
            try:
                pci = (cdir / "device").resolve(strict=True)
            except OSError:
                reasons.append(f"A5: {name} has no resolvable device link")
                ctrls.append(rec)
                continue
            prel = env.rel(pci)
            rec["pci_path_private"] = prel
            rec["pci_depth"] = prel.count("/") - 2
            if not env.inside(pci) or not re.match(r"^/sys/devices/pci[0-9a-f]{4}:[0-9a-f]{2}/", prel):
                reasons.append(f"A5: {name} controller does not sit under /sys/devices/pci*")
            cls = hexint(read(pci / "class"))
            vendor = hexint(read(pci / "vendor"))
            device = hexint(read(pci / "device"))
            svendor = hexint(read(pci / "subsystem_vendor"))
            rec.update(pci_class=f"0x{cls:06x}" if cls is not None else None,
                       vendor=f"0x{vendor:04x}" if vendor is not None else None,
                       device=f"0x{device:04x}" if device is not None else None,
                       subsystem_vendor=f"0x{svendor:04x}" if svendor is not None else None)
            if cls != NVME_CLASS:
                reasons.append(f"A5: {name} PCI class {rec['pci_class']} is not NVMe (0x010802)")
            if vendor is None or device is None:
                reasons.append(f"A5: {name} PCI vendor/device unreadable")
            elif vendor in EMULATED_VENDORS:
                reasons.append(f"A5: {name} PCI vendor 0x{vendor:04x} is {EMULATED_VENDORS[vendor]}")
            elif (vendor, device) in EMULATED_PAIRS:
                reasons.append(f"A5: {name} PCI id is {EMULATED_PAIRS[(vendor, device)]}")
            link = {k: read(pci / k) for k in ("current_link_speed", "current_link_width",
                                                "max_link_speed", "max_link_width")}
            rec["link"] = link
            rec["numa_node"] = read(pci / "numa_node")
            bad = [k for k, v in link.items() if not v or v.lower().startswith("unknown")]
            if bad:
                reasons.append(f"A5: {name} PCIe link fields missing or unknown: {', '.join(bad)}")
            ctrls.append(rec)
    if not ctrls and not reasons:
        reasons.append("A5: no controller to check")
    return ctrls, reasons


# ---------------------------------------------------------------- A6 I/O binding

def read_stat(env, stat_node):
    text = read(env.sys(stat_node[len("/sys/"):]) / "stat")
    if text is None:
        return None
    f = text.split()
    if len(f) < 7:
        return None
    return {"read_ios": int(f[0]), "read_sectors": int(f[2]),
            "write_ios": int(f[4]), "write_sectors": int(f[6])}


_BASES = {}


def fill_chunk(buf, chunk_index, seed):
    base = _BASES.get(seed)
    if base is None:
        base = _BASES[seed] = bytes((i * 131 + seed) & 0xFF for i in range(4096))
    for off in range(0, len(buf), 4096):
        buf[off:off + 4096] = base
        struct.pack_into("<QQ", buf, off, chunk_index, off)


def io_binding(env, real, nbytes, top_node, leaves, dio_mem_align, dio_offset_align):
    reasons, rec = [], {"payload_bytes": nbytes}
    nodes = {"top": top_node, **{f"leaf:{leaf['name']}": leaf["stat_node_private"] for leaf in leaves}}
    rec["dio_mem_align"], rec["dio_offset_align"] = dio_mem_align, dio_offset_align
    if nbytes % CHUNK or nbytes < 16 * CHUNK:
        return rec, ["A6: --bind-bytes must be a multiple of 4 MiB and at least 64 MiB"]
    if (dio_mem_align or 0) > mmap.PAGESIZE:
        return rec, [f"A6: DIO memory alignment {dio_mem_align} exceeds the page-aligned buffer"]
    if dio_offset_align and CHUNK % dio_offset_align:
        return rec, [f"A6: DIO offset alignment {dio_offset_align} does not divide the 4 MiB chunk"]
    name = real / f".m1-proof-{os.getpid()}-{time.time_ns()}.bin"
    rec["file"] = name.name
    seed = int.from_bytes(os.urandom(1), "little")
    buf = mmap.mmap(-1, CHUNK)
    snaps = {}

    def snap(label):
        snaps[label] = {k: read_stat(env, v) for k, v in nodes.items()}
    fd = None
    try:
        try:
            fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_DIRECT", 0), 0o600)
        except OSError as e:
            return rec, [f"A6: O_DIRECT create refused: {e.strerror} (errno {e.errno})"]
        wsha = hashlib.sha256()
        snap("before_write")
        t0 = time.monotonic_ns()
        for i in range(nbytes // CHUNK):
            fill_chunk(buf, i, seed)
            wsha.update(buf)
            n = os.pwrite(fd, buf, i * CHUNK)
            if n != CHUNK:
                reasons.append(f"A6: short O_DIRECT write {n} of {CHUNK} at chunk {i}")
                break
        os.fdatasync(fd)
        rec["write_fdatasync_ns"] = time.monotonic_ns() - t0
        snap("after_write")
        rsha = hashlib.sha256()
        mismatch = None
        t1 = time.monotonic_ns()
        expect = mmap.mmap(-1, CHUNK)
        for i in range(nbytes // CHUNK):
            n = os.preadv(fd, [buf], i * CHUNK)
            if n != CHUNK:
                reasons.append(f"A6: short O_DIRECT read {n} of {CHUNK} at chunk {i}")
                break
            rsha.update(buf)
            if mismatch is None:
                fill_chunk(expect, i, seed)
                if buf[:] != expect[:]:
                    mismatch = i
        rec["read_ns"] = time.monotonic_ns() - t1
        snap("after_read")
        rec["write_sha256"] = wsha.hexdigest()
        rec["read_sha256"] = rsha.hexdigest()
        if mismatch is not None or rec["write_sha256"] != rec["read_sha256"]:
            reasons.append(f"A6: byte mismatch (first differing chunk {mismatch})")
    except OSError as e:
        reasons.append(f"A6: I/O error: {e.strerror} (errno {e.errno})")
    finally:
        if fd is not None:
            os.close(fd)
        try:
            name.unlink()
            rec["removed"] = True
        except OSError as e:
            rec["removed"] = False
            reasons.append(f"A6: could not remove the proof file: {e.strerror}")
    need = nbytes // SECTOR
    rec["required_sectors"] = need
    rec["snapshots"] = snaps
    if {"before_write", "after_write", "after_read"} <= snaps.keys():
        deltas = {}
        for k in nodes:
            b, w, r = (snaps[s][k] for s in ("before_write", "after_write", "after_read"))
            if None in (b, w, r):
                reasons.append(f"A6: stat counters unreadable for {k}")
                continue
            deltas[k] = {"write_sectors": w["write_sectors"] - b["write_sectors"],
                         "read_sectors": r["read_sectors"] - w["read_sectors"]}
        rec["deltas"] = deltas
        leaf_w = sum(v["write_sectors"] for k, v in deltas.items() if k.startswith("leaf:"))
        leaf_r = sum(v["read_sectors"] for k, v in deltas.items() if k.startswith("leaf:"))
        rec["leaf_write_sectors"], rec["leaf_read_sectors"] = leaf_w, leaf_r
        rec["foreign_write_sectors_upper"] = max(0, leaf_w - need)
        rec["foreign_read_sectors_upper"] = max(0, leaf_r - need)
        if leaf_w < need:
            reasons.append(f"A6: leaf write sectors {leaf_w} < payload {need}")
        if leaf_r < need:
            reasons.append(f"A6: leaf read sectors {leaf_r} < payload {need}")
        top = deltas.get("top")
        if top and top["write_sectors"] < need:
            reasons.append(f"A6: top device write sectors {top['write_sectors']} < payload {need}")
        if top and top["read_sectors"] < need:
            reasons.append(f"A6: top device read sectors {top['read_sectors']} < payload {need}")
    return rec, reasons


# ---------------------------------------------------------------- report

PRIVATE_KEYS = ("_private",)


def public_identity(identity):
    if not isinstance(identity, dict):
        return identity
    out = dict(identity)
    if "filesystem_id" in out:
        out["filesystem_id_sha256_16"] = hashlib.sha256(str(out.pop("filesystem_id")).encode()).hexdigest()[:16]
    return out


def sanitize(obj):
    if isinstance(obj, dict):
        return {k: public_identity(sanitize(v)) if k in ("identity", "A8_identity") else sanitize(v)
                for k, v in obj.items() if not k.endswith(PRIVATE_KEYS)}
    if isinstance(obj, list):
        return [sanitize(v) for v in obj]
    return obj


def run(args, env):
    reasons, report = [], {"schema": SCHEMA,
                           "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                           "registration": "research/spill-f-20260919/M1-PREREG.md section A",
                           "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    kernel, r = check_kernel(env)
    report["A1_kernel"] = kernel
    reasons += r
    mount, r, majmin = check_path_and_mount(env, Path(args.path))
    report["A2_A3_mount"] = mount
    reasons += r
    leaves = []
    if majmin and majmin[0] != 0:
        graph, r = trace_block(env, *majmin)
        report["A4_block_graph"] = graph
        reasons += r
        leaves = graph["leaves"]
        ctrls, r = check_controllers(env, leaves)
        report["A5_controllers"] = ctrls
        reasons += r
    else:
        reasons.append("A4: skipped (no block device)")
    try:
        vfs = os.statvfs(args.path)
        free = vfs.f_bavail * vfs.f_frsize
        report["A7_capacity"] = {"free_bytes": free, "reserve_bytes": args.reserve_bytes}
        if free < args.reserve_bytes:
            reasons.append(f"A7: free {free} bytes < reserve {args.reserve_bytes}")
    except OSError as e:
        reasons.append(f"A7: statvfs failed: {e}")
    if args.bind_bytes == 0:
        reasons.append("A6: not run (--bind-bytes 0); a PASS requires the I/O binding")
    elif reasons:
        report["A6_io_binding"] = {"skipped": "earlier steps failed; no write attempted"}
        reasons.append("A6: not run because earlier steps failed")
    else:
        top = f"/sys/dev/block/{majmin[0]}:{majmin[1]}"
        real = Path(args.path).resolve()
        sx = mount.get("statx") or {}
        binding, r = io_binding(env, real, args.bind_bytes, top, leaves,
                                sx.get("dio_mem_align"), sx.get("dio_offset_align"))
        report["A6_io_binding"] = binding
        reasons += r
        after = filesystem_identity(real)
        if after != mount.get("identity"):
            reasons.append("A8: filesystem identity changed during the proof")
    report["A8_identity"] = mount.get("identity")
    report["reasons"] = reasons
    report["verdict"] = "PASS" if not reasons else "FAIL"
    report["class"] = "nvme-local-direct" if not reasons else None
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--path", required=True)
    ap.add_argument("--bind-bytes", type=int, default=1 << 30)
    ap.add_argument("--reserve-bytes", type=int, default=20 << 30)
    ap.add_argument("--private-out", type=Path, required=True)
    ap.add_argument("--public-out", type=Path, required=True)
    ap.add_argument("--sys-root", default="/sys", help="fixture seam; real runs use /sys")
    ap.add_argument("--proc-root", default="/proc", help="fixture seam; real runs use /proc")
    try:
        args = ap.parse_args(argv)
    except SystemExit:
        return 2
    for out in (args.private_out, args.public_out):
        if out.exists():
            print(f"REFUSED: {out} exists; receipts are never overwritten", file=sys.stderr)
            return 2
    report = run(args, Env(args.sys_root, args.proc_root))
    private = json.dumps(report, indent=1, sort_keys=True) + "\n"
    args.private_out.parent.mkdir(parents=True, exist_ok=True)
    args.private_out.write_text(private)
    public = sanitize(report)
    public["private_capture_sha256"] = hashlib.sha256(private.encode()).hexdigest()
    args.public_out.parent.mkdir(parents=True, exist_ok=True)
    args.public_out.write_text(json.dumps(public, indent=1, sort_keys=True) + "\n")
    print(f"M1-PROOF verdict={report['verdict']} class={report['class']} reasons={len(report['reasons'])}")
    for r in report["reasons"]:
        print(f"  {r}")
    return 0 if report["verdict"] == "PASS" else 3


if __name__ == "__main__":
    sys.exit(main())
