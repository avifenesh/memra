#!/usr/bin/env python3
"""Read-only BOX2 facts; emit hardware/path classes, never deployment identities.

Run remotely via the operator's existing SSH connection. Does not open artifacts,
credentials, environment files, source trees or other lanes' receipts. du reads
metadata only; subprocesses and directory census are individually bounded.
"""
import datetime
import glob
import json
from pathlib import Path
import subprocess


def command(argv, timeout=20):
    try:
        p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
        return {"argv": argv, "exit_code": p.returncode, "stdout": p.stdout,
                "stderr": p.stderr}
    except subprocess.TimeoutExpired:
        return {"argv": argv, "status": "timeout", "seconds": timeout}


def main():
    out = {"schema_version": 1, "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "rig": "BOX2", "evidence_class": "read-only-inventory-not-performance"}
    # Never include mount OPTIONS (overlay lowerdirs disclose container identities).
    out["mounts"] = [command(["findmnt", "-J", "-T", p, "-o", "TARGET,SOURCE,FSTYPE,MAJ:MIN"])
                     for p in ["/", "/root", "/scratch", "/workspace", "/dev/shm"]]
    out["lsblk"] = command(["lsblk", "-J", "-o", "NAME,MAJ:MIN,TYPE,SIZE,ROTA,TRAN,FSTYPE,MOUNTPOINTS"])
    out["df"] = command(["df", "-T", "-B1", "/", "/root", "/dev/shm"])
    out["statfs"] = command(["stat", "-f", "-c", "%T %s %b %f %a", "/root"])
    # Only major:minor, type and mountpoint, not options/host paths/ids.
    out["proc_mount_classes"] = sorted({line.split()[2] for line in Path("/proc/mounts").read_text().splitlines()})
    out["sys_block"] = []
    for name in sorted(glob.glob("/sys/block/*")):
        p = Path(name)
        if p.name.startswith(("loop", "ram", "zram")):
            continue
        row = {"name": p.name}
        for key in ["dev", "size", "queue/rotational", "queue/logical_block_size"]:
            f = p / key
            if f.is_file():
                row[key] = f.read_text().strip()
        row["device_node_visible"] = (Path("/dev") / p.name).exists()
        out["sys_block"].append(row)
    # Aggregate sibling targets without disclosing or modifying their contents.
    groups = {"artifacts": ["/root/artifacts"], "base_target": ["/root/memra-spill/target"],
              "receipts": ["/root/spill-receipts"],
              "lane_targets": sorted(glob.glob("/root/wt-*/target")),
              "cargo_registry": ["/root/.cargo/registry"], "cargo_git": ["/root/.cargo/git"]}
    out["disk_usage"] = {}
    for group, paths in groups.items():
        paths = [p for p in paths if Path(p).exists()]
        total = 0
        statuses = []
        for p in paths:
            r = command(["du", "-s", "-x", "-B1", p], timeout=25)
            if r.get("exit_code") == 0:
                total += int(r["stdout"].split()[0])
                statuses.append("ok")
            else:
                statuses.append("timeout" if r.get("status") == "timeout" else "error")
        out["disk_usage"][group] = {"paths_count": len(paths), "allocated_bytes_sum": total,
                                    "statuses": statuses, "complete": all(s == "ok" for s in statuses)}
    out["gpu"] = command(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,power.limit,power.max_limit,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max", "--format=csv"])
    out["compute_apps"] = command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"])
    out["memory"] = command(["free", "-b"])
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
