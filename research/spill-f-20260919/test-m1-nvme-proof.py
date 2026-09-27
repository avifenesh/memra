#!/usr/bin/env python3
"""Fixture red/green controls for m1-nvme-proof.py (CPU only, no writes outside a temp dir).

Every fail-closed branch in M1-PREREG.md section A has a fixture that must FAIL, and each
admitted topology has a fixture that must PASS. Fixture sysfs trees use the same relative
symlinks as the kernel, so the tool's resolution code runs unchanged.
"""
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("proof", HERE / "m1-nvme-proof.py")
proof = importlib.util.module_from_spec(spec)
spec.loader.exec_module(proof)

PCI_NVME = "devices/pci0000:00/0000:00:01.0/0000:01:00.0"
STAT = "100 0 800 5 200 0 1600 7 0 10 12 0 0 0 0 0 0"


class Tree:
    def __init__(self, root):
        self.root = Path(root)
        self.sys = self.root / "sys"
        self.proc = self.root / "proc"
        self.put("proc/cpuinfo", "processor\t: 0\nflags\t\t: fpu sse2 avx2\n")
        self.put("sys/class/dmi/id/sys_vendor", "Example Board Co")
        self.put("sys/class/dmi/id/product_name", "Workstation 9")
        self.pci("devices/pci0000:00/0000:00:00.0", 0x1022, 0x14D8, 0x060000)

    def put(self, rel, text):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text + "\n")

    def link(self, rel, target):
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(os.path.relpath(self.root / target, p.parent), p)

    def pci(self, rel, vendor, device, cls, link=True):
        self.put(f"sys/{rel}/vendor", f"0x{vendor:04x}")
        self.put(f"sys/{rel}/device", f"0x{device:04x}")
        self.put(f"sys/{rel}/subsystem_vendor", f"0x{vendor:04x}")
        self.put(f"sys/{rel}/class", f"0x{cls:06x}")
        for k, v in (("current_link_speed", "16.0 GT/s PCIe"), ("current_link_width", "4"),
                     ("max_link_speed", "16.0 GT/s PCIe"), ("max_link_width", "4"),
                     ("numa_node", "0")):
            self.put(f"sys/{rel}/{k}", v)
        if link:
            self.link(f"sys/bus/pci/devices/{Path(rel).name}", f"sys/{rel}")

    def nvme(self, ctrl="nvme0", ns="nvme0n1", pci=PCI_NVME, vendor=0x15B7, device=0x5030,
             model="EXAMPLE NVMe 2TB", transport="pcie", majmin="259:0"):
        self.pci(pci, vendor, device, 0x010802)
        c = f"sys/{pci}/nvme/{ctrl}"
        self.put(f"{c}/transport", transport)
        self.put(f"{c}/model", model)
        self.put(f"{c}/firmware_rev", "1.0")
        self.put(f"{c}/serial", "SERIAL123")
        self.link(f"{c}/device", f"sys/{pci}")
        self.link(f"sys/class/nvme/{ctrl}", c)
        self.put(f"{c}/{ns}/dev", majmin)
        self.put(f"{c}/{ns}/stat", STAT)
        self.put(f"{c}/{ns}/queue/rotational", "0")
        self.link(f"sys/dev/block/{majmin}", f"{c}/{ns}")
        return f"{c}/{ns}"

    def partition(self, disk, name, majmin):
        self.put(f"{disk}/{name}/dev", majmin)
        self.put(f"{disk}/{name}/partition", "2")
        self.put(f"{disk}/{name}/stat", STAT)
        self.link(f"sys/dev/block/{majmin}", f"{disk}/{name}")
        return f"{disk}/{name}"

    def virtual(self, name, majmin, members, kind=None):
        d = f"sys/devices/virtual/block/{name}"
        self.put(f"{d}/dev", majmin)
        self.put(f"{d}/stat", STAT)
        if kind == "dm":
            self.put(f"{d}/dm/name", "vg-data")
            self.put(f"{d}/dm/uuid", "LVM-abcdef")
        if kind == "md":
            self.put(f"{d}/md/level", "raid0")
        for rel in members:
            self.link(f"{d}/slaves/{Path(rel).name}", rel)
        self.link(f"sys/dev/block/{majmin}", d)
        return d

    def env(self):
        return proof.Env(self.sys, self.proc)


def trace(tree, majmin):
    env = tree.env()
    graph, reasons = proof.trace_block(env, *map(int, majmin.split(":")))
    ctrls, creasons = proof.check_controllers(env, graph["leaves"])
    return graph, reasons + creasons, ctrls


class Green(unittest.TestCase):
    def test_bare_metal_partition(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            disk = t.nvme()
            t.partition(disk, "nvme0n1p2", "259:3")
            rec, kr = proof.check_kernel(t.env())
            self.assertEqual(kr, [])
            graph, reasons, ctrls = trace(t, "259:3")
            self.assertEqual(reasons, [])
            self.assertEqual([leaf["name"] for leaf in graph["leaves"]], ["nvme0n1"])
            self.assertEqual(ctrls[0]["pci_class"], "0x010802")

    def test_lvm_over_nvme(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            part = t.partition(t.nvme(), "nvme0n1p1", "259:1")
            t.virtual("dm-0", "252:0", [part], kind="dm")
            graph, reasons, _ = trace(t, "252:0")
            self.assertEqual(reasons, [])
            self.assertEqual(graph["nodes"][0]["kind"], "dm")
            self.assertEqual(graph["nodes"][0]["dm_uuid_prefix"], "LVM")

    def test_md_raid0_two_nvme(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            a = t.nvme()
            b = t.nvme("nvme1", "nvme1n1", "devices/pci0000:00/0000:00:02.0/0000:02:00.0",
                       majmin="259:5")
            t.virtual("md0", "9:0", [a, b], kind="md")
            graph, reasons, ctrls = trace(t, "9:0")
            self.assertEqual(reasons, [])
            self.assertEqual(sorted(leaf["name"] for leaf in graph["leaves"]), ["nvme0n1", "nvme1n1"])
            self.assertEqual(len(ctrls), 2)

    def test_native_multipath_head_with_pcie_path(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.nvme(ns="nvme0c0n1", majmin="259:9")
            head = "sys/devices/virtual/nvme-subsystem/nvme-subsys0/nvme0n1"
            t.put(f"{head}/dev", "259:0")
            t.put(f"{head}/stat", STAT)
            t.link(f"{head}/multipath/nvme0c0n1", f"sys/{PCI_NVME}/nvme/nvme0/nvme0c0n1")
            os.unlink(t.root / "sys/dev/block/259:0") if (t.root / "sys/dev/block/259:0").is_symlink() else None
            t.link("sys/dev/block/259:0", head)
            graph, reasons, ctrls = trace(t, "259:0")
            self.assertEqual(reasons, [])
            self.assertEqual(ctrls[0]["controller"], "nvme0")


class Red(unittest.TestCase):
    def assertReason(self, reasons, needle):
        self.assertTrue(any(needle in r for r in reasons), f"{needle!r} not in {reasons}")

    def test_hypervisor_flag(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.put("proc/cpuinfo", "flags\t\t: fpu sse2 hypervisor\n")
            _, r = proof.check_kernel(t.env())
            self.assertReason(r, "CPU flag `hypervisor`")

    def test_dmi_virtual_platform(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.put("sys/class/dmi/id/sys_vendor", "QEMU")
            _, r = proof.check_kernel(t.env())
            self.assertReason(r, "DMI names a virtual platform")

    def test_hidden_flag_caught_by_pci_census(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.pci("devices/pci0000:00/0000:00:1f.0", 0x8086, 0x2918, 0x060100)
            _, r = proof.check_kernel(t.env())
            self.assertReason(r, "ICH9 LPC (Q35)")

    def test_empty_pci_census(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            (t.root / "sys/bus/pci/devices/0000:00:00.0").unlink()
            _, r = proof.check_kernel(t.env())
            self.assertReason(r, "platform census impossible")

    def test_virtio_block(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            vd = "sys/devices/pci0000:00/0000:00:04.0/virtio1/block/vda"
            t.pci("devices/pci0000:00/0000:00:04.0", 0x1AF4, 0x1042, 0x010000)
            t.put(f"{vd}/dev", "253:0")
            t.link("sys/dev/block/253:0", vd)
            _, reasons, _ = trace(t, "253:0")
            self.assertReason(reasons, "leaf vda is not an NVMe namespace (virtio block)")
            _, kr = proof.check_kernel(t.env())
            self.assertReason(kr, "virtio")

    def test_emulated_nvme_controller(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.nvme(vendor=0x1B36, device=0x0010, model="QEMU NVMe Ctrl")
            _, reasons, _ = trace(t, "259:0")
            self.assertReason(reasons, "PCI vendor 0x1b36 is Red Hat/QEMU")
            self.assertReason(reasons, "names an emulator")

    def test_nvme_over_tcp_namespace(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            c = "sys/devices/virtual/nvme-fabrics/ctl/nvme2"
            t.put(f"{c}/transport", "tcp")
            t.put(f"{c}/nvme2n1/dev", "259:20")
            t.link("sys/dev/block/259:20", f"{c}/nvme2n1")
            _, reasons, _ = trace(t, "259:20")
            self.assertReason(reasons, "under /sys/devices/virtual")

    def test_nvme_over_tcp_multipath_path(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            c = "sys/devices/virtual/nvme-fabrics/ctl/nvme2"
            t.put(f"{c}/transport", "tcp")
            t.put(f"{c}/model", "Linux")
            t.put(f"{c}/nvme2c2n1/dev", "259:21")
            head = "sys/devices/virtual/nvme-subsystem/nvme-subsys2/nvme2n1"
            t.put(f"{head}/dev", "259:20")
            t.link(f"{head}/multipath/nvme2c2n1", f"{c}/nvme2c2n1")
            t.link("sys/dev/block/259:20", head)
            _, reasons, _ = trace(t, "259:20")
            self.assertReason(reasons, "transport is 'tcp'")

    def test_raid1_with_sata_member(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            a = t.nvme()
            sd = "sys/devices/pci0000:00/0000:00:17.0/ata1/host0/target0:0:0/0:0:0:0/block/sda"
            t.put(f"{sd}/dev", "8:0")
            t.virtual("md1", "9:1", [a, sd], kind="md")
            _, reasons, _ = trace(t, "9:1")
            self.assertReason(reasons, "leaf sda is not an NVMe namespace (SCSI/SATA disk)")

    def test_loop_device(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            lp = "sys/devices/virtual/block/loop3"
            t.put(f"{lp}/dev", "7:3")
            t.put(f"{lp}/loop/backing_file", "/var/lib/volumes/v1.img")
            t.link("sys/dev/block/7:3", lp)
            _, reasons, _ = trace(t, "7:3")
            self.assertReason(reasons, "loop device")

    def test_unreadable_node(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            _, reasons, _ = trace(t, "259:77")
            self.assertReason(reasons, "does not resolve")

    def test_missing_dev_attribute(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            disk = t.nvme()
            (t.root / disk / "dev").unlink()
            _, reasons, _ = trace(t, "259:0")
            self.assertReason(reasons, "no readable dev attribute")

    def test_cycle_terminates(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            a = "sys/devices/virtual/block/dm-5"
            b = "sys/devices/virtual/block/dm-6"
            t.put(f"{a}/dev", "252:5")
            t.put(f"{b}/dev", "252:6")
            t.put(f"{a}/dm/uuid", "LVM-x")
            t.put(f"{b}/dm/uuid", "LVM-y")
            t.link(f"{a}/slaves/dm-6", b)
            t.link(f"{b}/slaves/dm-5", a)
            t.link("sys/dev/block/252:5", a)
            _, reasons, _ = trace(t, "252:5")
            self.assertReason(reasons, "no NVMe leaf found")

    def test_missing_link_fields(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            t.nvme()
            (t.root / "sys" / PCI_NVME / "current_link_speed").write_text("Unknown\n")
            _, reasons, _ = trace(t, "259:0")
            self.assertReason(reasons, "PCIe link fields missing or unknown: current_link_speed")


class Mountinfo(unittest.TestCase):
    TEXT = ("24 1 259:3 / / rw,relatime shared:1 - ext4 /dev/nvme0n1p2 rw\n"
            "30 24 0:40 / /scratch rw - tmpfs tmpfs rw\n"
            "31 24 259:3 /var/lib/v1 /scratch rw - ext4 /dev/nvme0n1p2 rw\n"
            "32 24 0:41 / /with\\040space rw - overlay overlay rw\n")

    def test_overmount_and_escape(self):
        rows = proof.parse_mountinfo(self.TEXT)
        by_id, longest = proof.covering_mount(rows, "/scratch/spill-f", 31)
        self.assertEqual(longest["mount_id"], 31)
        self.assertEqual(by_id["root"], "/var/lib/v1")
        self.assertEqual(rows[3]["mount_point"], "/with space")
        _, root = proof.covering_mount(rows, "/etc", 24)
        self.assertEqual(root["mount_id"], 24)

    def test_prefix_is_component_aware(self):
        rows = proof.parse_mountinfo(self.TEXT)
        _, m = proof.covering_mount(rows, "/scratchy/x", -1)
        self.assertEqual(m["mount_id"], 24)


class Cli(unittest.TestCase):
    def run_cli(self, path, extra=()):
        with tempfile.TemporaryDirectory() as d:
            priv, pub = Path(d, "priv.json"), Path(d, "pub.json")
            p = subprocess.run([sys.executable, str(HERE / "m1-nvme-proof.py"), "--path", str(path),
                                "--private-out", str(priv), "--public-out", str(pub), *extra],
                               capture_output=True, text=True)
            return p, (json.loads(priv.read_text()) if priv.exists() else None), \
                (json.loads(pub.read_text()) if pub.exists() else None)

    def test_tmpfs_fails_closed_and_public_is_sanitized(self):
        with tempfile.TemporaryDirectory(dir="/dev/shm") as d:
            p, priv, pub = self.run_cli(d, ("--bind-bytes", "0", "--reserve-bytes", "0"))
        self.assertEqual(p.returncode, 3, p.stdout + p.stderr)
        self.assertEqual(pub["verdict"], "FAIL")
        self.assertTrue(any("'tmpfs' is not block-traceable" in r for r in pub["reasons"]))
        self.assertTrue(any("anonymous device" in r for r in pub["reasons"]))
        text = json.dumps(pub)
        self.assertNotIn("_private", text)
        self.assertNotIn("filesystem_id\"", text)
        self.assertIn("filesystem_id_sha256_16", text)
        self.assertIn("realpath_private", json.dumps(priv))

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            existing = Path(d, "pub.json")
            existing.write_text("{}")
            p = subprocess.run([sys.executable, str(HERE / "m1-nvme-proof.py"), "--path", d,
                                "--private-out", str(Path(d, "priv.json")),
                                "--public-out", str(existing)], capture_output=True, text=True)
            self.assertEqual(p.returncode, 2)
            self.assertIn("never overwritten", p.stderr)


class Unit(unittest.TestCase):
    def test_read_stat(self):
        with tempfile.TemporaryDirectory() as d:
            t = Tree(d)
            ns = t.nvme()
            s = proof.read_stat(t.env(), "/sys/" + ns[len("sys/"):])
            self.assertEqual(s, {"read_ios": 100, "read_sectors": 800,
                                 "write_ios": 200, "write_sectors": 1600})

    def test_sanitize_nested(self):
        obj = {"a_private": 1, "b": [{"c_private": 2, "d": 3}],
               "identity": {"device": 1, "filesystem_id": 99, "mount_id": 5}}
        s = proof.sanitize(obj)
        self.assertEqual(s["b"], [{"d": 3}])
        self.assertNotIn("a_private", s)
        self.assertNotIn("filesystem_id", s["identity"])
        self.assertEqual(s["identity"]["mount_id"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
