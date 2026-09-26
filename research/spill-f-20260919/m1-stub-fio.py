#!/usr/bin/env python3
"""CPU stand-in for fio in m1-fio-envelope tests: writes fio json+ shaped output. Never a measurement.

Bandwidth per engine comes from M1_STUB_FIO_BW (JSON {engine: bytes/s}); the prepare step creates
the file at --size. --version prints a stub version.
"""
import json
import os
import sys

args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
if "--version" in sys.argv:
    print("fio-stub-3.41")
    sys.exit(0)
units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30}
size = args["size"]
size = int(size[:-1]) * units[size[-1]] if size[-1] in units else int(size)
if args["name"] == "m1-prep":
    with open(args["filename"], "wb") as f:
        f.truncate(size)
if os.environ.get("M1_STUB_FIO_REFUSE_URING") == "1" and args["ioengine"] == "io_uring":
    open(args["output"], "w").write('{"jobs": [{"error": 1}]}\nfio: pid=1, err=1/file:engines/io_uring.c:1049, '
                                     'func=io_queue_init, error=Operation not permitted\n')
    sys.exit(1)
bw = json.loads(os.environ.get("M1_STUB_FIO_BW", "{}")).get(args["ioengine"], 1_000_000_000)
side = {"io_bytes": bw * 10, "bw_bytes": bw, "iops": bw / int(args["bs"]), "runtime": 10000,
        "clat_ns": {"percentile": {"50.000000": 80000, "99.000000": 300000}}}
empty = {"io_bytes": 0, "bw_bytes": 0, "iops": 0, "runtime": 0}
write = args["rw"] == "write"
out = {"fio version": "fio-stub-3.41", "jobs": [{"jobname": args["name"], "read": empty if write else side,
                                                   "write": side if write else empty, "usr_cpu": 1.5, "sys_cpu": 4.0}]}
open(args["output"], "w").write(json.dumps(out))
