#!/usr/bin/env python3
"""OWED 20 storage-bound cells on Step-3.7-Flash IQ4_XS (M1-PREREG.md section H).

A wrapper over `m1-spill-runner.py` (loaded as a module, its file untouched) for the split,
two-card artifact:

- all three shards: cold start (`DONTNEED` + `mincore` 0) and residency cover every shard, and each
  shard's size and sha256 are checked against the arms lock before any visit;
- every `[spill-pread]` totals line a visit prints (one per PP stage engine) is summed field by field;
- the per-visit GPU sampler covers both cards;
- `reference`: one resident visit of the qualified PP-2 program (no disk tier, same binary, prompt
  and env otherwise); its token ids become the oracle file every disk-tier arm must match;
- `run`: the B3 runner's `run` with this lock, `--oracle-tokens` from the reference, and a refusal
  unless host MemTotal is below the lock's expert-bank bytes.

  m1-step-runner.py reference --arms-lock LOCK --binary RUN_GEN --shard-dir DIR --mtp-draft FILE \
      --out DIR --lock-fd N
  m1-step-runner.py run --shard-dir DIR --mtp-draft FILE -- <m1-spill-runner run arguments>
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("runner", HERE.parent / "m1-spill-runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
DROP = re.compile(R.PATTERNS["drop"])
SPILL_KEYS = ("MEMRA_SPILL_DISK", "MEMRA_SPILL_PINNED_FRAC", "MEMRA_MOE_CACHE", "MEMRA_MOE_RESIDENT",
              "MEMRA_MOE_SLOTS", "MEMRA_SPILL_STATS", "MEMRA_SPILL_IO", "MEMRA_SPILL_PREAD_DEPTH",
              "MEMRA_MOE_MMAP_ADVICE")


def shard_paths(lock, shard_dir):
    return [str(Path(shard_dir) / Path(f["path"]).name) for f in lock["artifact"]["files"] if f.get("shard")]


def install(lock, shard_dir):
    """Point the B3 runner's cache regime, log parser and GPU sampler at the split two-card artifact."""
    shards = shard_paths(lock, shard_dir)
    cache = R.CACHE

    class SplitCache:
        def __getattr__(self, name):
            return getattr(cache, name)

        def cold(self, files, retries=5):
            return cache.cold(shards if list(files) == shards[:1] else files, retries)

        def residency(self, path):
            if path != shards[0]:
                return cache.residency(path)
            got = [cache.residency(p) for p in shards]
            return sum(r for r, _ in got), sum(p for _, p in got)
    R.CACHE = SplitCache()
    base_parse = R.parse_log

    def parse_log(text):
        out = base_parse(text)
        lines = DROP.findall(text)
        out["drop_lines"] = len(lines)
        if len(lines) > 1:
            out["drop"] = [str(sum(int(line[i]) for line in lines)) for i in range(len(lines[0]))]
        return out
    R.parse_log = parse_log
    base_sampler = R.GPU.GpuSampler

    class BothCards(base_sampler):
        def __init__(self, path, device="0,1"):
            super().__init__(path, device)
    R.GPU.GpuSampler = BothCards
    return shards


def check_shards(lock, shards):
    for path, f in zip(shards, [f for f in lock["artifact"]["files"] if f.get("shard")]):
        R.B.require(os.path.getsize(path) == f["bytes"], f"{path}: size != the pinned {f['bytes']}")
        R.B.require(R.sha(path) == f["sha256"], f"{path}: sha256 != the pinned {f['sha256']}")


def check_ram(lock):
    total = int(next(l for l in Path("/proc/meminfo").read_text().splitlines()
                     if l.startswith("MemTotal:")).split()[1]) * 1024
    bank = lock["artifact"]["expert_bank_bytes"]
    R.B.require(total < bank, f"MemTotal {total} is not below the expert bank {bank}: not storage-bound")
    return {"mem_total_bytes": total, "expert_bank_bytes": bank, "cache_ceiling_fraction": total / bank}


def reference(a):
    lock = json.loads(Path(a.arms_lock).read_text())
    shards = install(lock, a.shard_dir)
    proc = subprocess.run([sys.executable, str(R.ROOT / "tools/tier-lock-proof.py"), "--fd", str(a.lock_fd),
                           "--lock", R.B.LOCKS["pro-pair"]], pass_fds=(a.lock_fd,), capture_output=True, text=True)
    R.B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
    check_shards(lock, shards)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    env = R.arm_env(lock, {"env": {}})
    for key in SPILL_KEYS:
        env.pop(key, None)
    env["MEMRA_MTP_DRAFT"] = a.mtp_draft
    R.CACHE.cold(shards[:1])
    with (out / "run.log").open("xb") as log:
        rc = subprocess.run([a.binary, shards[0]], stdout=log, stderr=subprocess.STDOUT, env=env, cwd=R.ROOT).returncode
    parsed = R.parse_log((out / "run.log").read_text(errors="replace"))
    ok = bool(rc == 0 and parsed["gate"] and parsed["gate"][2] == "MATCH" and parsed["tokens"]
              and parsed["placed"] is None and parsed["pread_enabled"] is None)
    rec = {"rc": rc, "gate": parsed["gate"], "tokens": len(parsed["tokens"] or []),
           "disk_tier_absent": parsed["placed"] is None and parsed["pread_enabled"] is None,
           "env": {k: v for k, v in env.items() if k.startswith("MEMRA_")}, "ok": ok, "qualified": False}
    (out / "reference.json").write_text(json.dumps(rec, indent=1) + "\n")
    if ok:
        (out / "oracle-tokens.json").write_text(json.dumps(parsed["tokens"]) + "\n")
    print(f"M1-STEP-REFERENCE ok={ok} rc={rc} gate={parsed['gate']} tokens={rec['tokens']}", flush=True)
    return 0 if ok else 3


def run(a, rest):
    lock_path = rest[rest.index("--arms-lock") + 1]
    lock = json.loads(Path(lock_path).read_text())
    shards = install(lock, a.shard_dir)
    R.B.require(rest[rest.index("--artifact") + 1] == shards[0], "--artifact must be shard 1 of --shard-dir")
    os.environ["MEMRA_MTP_DRAFT"] = a.mtp_draft
    ram = check_ram(lock)
    out = Path(rest[rest.index("--out") + 1])
    out.parent.mkdir(parents=True, exist_ok=True)
    (out.parent / "storage-bound.json").write_text(json.dumps(ram, indent=1) + "\n")
    print("M1-STEP-RAM " + json.dumps(ram), flush=True)
    return R.main(["run", *rest])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("reference")
    for name in ("--arms-lock", "--binary", "--shard-dir", "--mtp-draft", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--lock-fd", type=int, required=True)
    g = sub.add_parser("run")
    g.add_argument("--shard-dir", required=True)
    g.add_argument("--mtp-draft", required=True)
    argv = sys.argv[1:]
    if argv and argv[0] == "run" and "--" in argv:
        split = argv.index("--")
        a = ap.parse_args(argv[:split])
        return run(a, argv[split + 1:])
    a = ap.parse_args(argv)
    if a.cmd == "reference":
        return reference(a)
    ap.error("run needs `--` before the m1-spill-runner arguments")


if __name__ == "__main__":
    sys.exit(main())
