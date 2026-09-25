#!/usr/bin/env python3
"""M1 B2 driver: KV host-tier handoff export/import on the proven path (OWED 13).

  run --gate kv-handoff-gate --server memra-server --artifact FILE --prompts b2-prompts.jsonl
      --proof PRIVATE_PROOF.json --scratch DIR --out DIR --size-bytes N [--cycles 5]
      --rig pro-single|rtx5090 --lock-fd N [--port 18119] [--stub-no-lock]

Registration: M1-PREREG.md "B2 amendment". One reference boot of the stock server with the host
tier off gives the probes' cold texts. Each cycle: boot the gate, fill the host tier with
prompts until /metrics prefix_host_bytes reaches --size-bytes, SIGUSR1 export (sampler on the
proof's leaves over the export window), SIGTERM, hash the file, cold regime on it, boot the gate
again (sampler over the import window), wait for the import DONE line, send the four probes,
SIGTERM. Server stdout and stderr go straight to per-boot log files; parsing reads only those.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "tools/tier-battery.py").exists())


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B = load("battery", ROOT / "tools/tier-battery.py")
CACHE = load("regime", HERE / "m1-cache-regime.py")
RUNNER = load("runner", HERE / "m1-spill-runner.py")
SAMPLER = HERE / "m1-host-sampler.py"
SAMPLE = load("sampler", SAMPLER)
PROBE_SUFFIX = " Finally, state which single item most often fails first and why."

EXPORT = re.compile(r"\[prefix-host\] handoff export: (\d+) entries / ([\d.]+)MB to (\S+) in (\d+)ms "
                    r"write_ms=([\d.]+) fsync_ms=([\d.]+) \(drain-demoted (\d+) device entries first; "
                    r"(\d+) skipped over")
ARMED_BOOT = re.compile(r"\[prefix-host\] handoff import armed at boot: (\d+) entries / ([\d.]+)MB")
DONE = re.compile(r"\[prefix-host\] handoff import DONE: (\d+) entries / ([\d.]+)MB re-materialized, "
                  r"(\d+) skipped, in ([\d.]+)s from")
ABORTED = "[prefix-host] handoff import ABORTED"


def parse_export(text):
    m = EXPORT.search(text)
    if not m:
        return None
    n, mb, path, ms, wms, fms, demoted, skipped = m.groups()
    return {"entries": int(n), "mb": float(mb), "ms": int(ms), "write_ms": float(wms),
            "fsync_ms": float(fms), "drain_demoted": int(demoted), "skipped_over_cap": int(skipped)}


def parse_import(text):
    armed, done = ARMED_BOOT.search(text), DONE.search(text)
    return {"armed": [int(armed[1]), float(armed[2])] if armed else None,
            "done": {"entries": int(done[1]), "mb": float(done[2]), "skipped": int(done[3]),
                     "seconds": float(done[4])} if done else None,
            "aborted": ABORTED in text}


class Server:
    def __init__(self, binary, env, log, port):
        self.log = Path(log)
        self.port = port
        with self.log.open("xb") as out:
            self.proc = subprocess.Popen([str(binary)], stdout=out, stderr=subprocess.STDOUT, env=env, cwd=ROOT)

    def text(self):
        return self.log.read_text(errors="replace")

    def wait_for(self, predicate, timeout, what):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            B.require(self.proc.poll() is None, f"server exited while waiting for {what}; see {self.log.name}")
            time.sleep(0.1)
        raise ValueError(f"timed out after {timeout}s waiting for {what}")

    def ready(self):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{self.port}/v1/models", timeout=2)
            return True
        except OSError:
            return False

    def post(self, prompt, max_tokens):
        body = {"model": "gate", "prompt": prompt, "max_tokens": max_tokens, "temperature": 0}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/completions",
                                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=900))

    def metrics(self):
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/metrics", timeout=30))

    def stop(self, timeout=120):
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        return self.proc.returncode


class Sampler:
    def __init__(self, devices, pid, out):
        self.out = out
        self.devices = devices
        with Path(str(out) + ".err").open("xb") as err:
            self.proc = subprocess.Popen([sys.executable, str(SAMPLER), "sample", "--devices", ",".join(devices),
                                          "--pid", str(pid), "--out", str(out)], stdout=subprocess.DEVNULL, stderr=err)
        time.sleep(0.3)

    def stop(self, leaves):
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=10)
        problems = SAMPLE.validate(self.out, self.devices, 500)
        rows = [json.loads(l) for l in Path(self.out).read_text().splitlines() if l.strip()]
        ticks = [r for r in rows[1:] if r.get("kind") == "tick"]
        delta = None
        if len(ticks) >= 2:
            first, last = ticks[0]["diskstats"], ticks[-1]["diskstats"]
            delta = {"read_bytes": 512 * sum(last[d]["read_sectors"] - first[d]["read_sectors"] for d in leaves),
                     "write_bytes": 512 * sum(last[d]["write_sectors"] - first[d]["write_sectors"] for d in leaves)}
        return {"telemetry_ok": not problems, "problems": problems[:5], "device": delta}


def base_env(args, host_mb, handoff):
    env = dict(os.environ)
    env.update({"MEMRA_COMPAT": "openai", "MEMRA_MODELS": f"gate={args.artifact}",
                "MEMRA_ADDR": f"127.0.0.1:{args.port}", "MEMRA_CTX": "8192", "MEMRA_MAX_SESSIONS": "4",
                "MEMRA_PREFIX_CACHE_MB": "1024", "MEMRA_KV_HOST_MB": str(host_mb),
                # B2 amendment 2: the prompt-end seed publishes only for plain sessions; spec off
                # is the documented production posture for prefix-cache shapes (FLAGS.md).
                "MEMRA_SERVE_SPEC": "0"})
    env.pop("MEMRA_KV_HOST_HANDOFF", None)
    if handoff:
        env["MEMRA_KV_HOST_HANDOFF"] = str(handoff)
    return env


def probes(prompts):
    return [p + PROBE_SUFFIX for p in prompts[:4]]


def cycle(args, i, prompts, reference, leaves, top, identity):
    cdir = args.out / f"cycle-{i + 1:02d}"
    cdir.mkdir()
    rec = {"cycle": i + 1, "size_bytes": args.size_bytes, "problems": []}
    handoff = Path(args.scratch) / "handoff.bin"
    B.require(not handoff.exists(), "a stale handoff file exists before the cycle")
    B.require(B.filesystem_identity(Path(args.scratch)) == identity, "scratch left the proven filesystem")
    env = base_env(args, 16384, handoff)
    a = Server(args.gate, env, cdir / "export-boot.log", args.port)
    try:
        a.wait_for(a.ready, 900, "export boot ready")
        a.wait_for(lambda: "[handoff-gate] armed" in a.text(), 60, "gate armed")
        sent = 0
        for prompt in prompts:
            a.post(prompt, 1)
            sent += 1
            if a.metrics().get("prefix_host_bytes", 0) >= args.size_bytes:
                break
        rec["prompts_sent"] = sent
        settle = time.monotonic() + 10  # demotion lands on a later tick
        while a.metrics().get("prefix_host_bytes", 0) < args.size_bytes and time.monotonic() < settle:
            time.sleep(0.5)
        rec["host_bytes_before_export"] = a.metrics().get("prefix_host_bytes")
        if rec["host_bytes_before_export"] < args.size_bytes:
            rec["problems"].append("prompts exhausted before the host tier reached the size")
        s = Sampler(leaves + [top], a.proc.pid, cdir / "export-host.jsonl")
        a.proc.send_signal(signal.SIGUSR1)
        a.wait_for(lambda: "[handoff-gate] export ok" in a.text() or "[handoff-gate] export refused" in a.text(),
                   900, "export answer")
        rec["export_window"] = s.stop(leaves)
        rec["export"] = parse_export(a.text())
        if "[handoff-gate] export refused" in a.text() or rec["export"] is None:
            rec["problems"].append("export refused or its line is missing")
    finally:
        rec["export_boot_exit"] = a.stop()
    if handoff.exists():
        rec["file_bytes"] = handoff.stat().st_size
        rec["file_sha256"] = RUNNER.sha(handoff)
        rec["cold_ok"] = CACHE.cold([str(handoff)])
        if not rec["cold_ok"]:
            rec["problems"].append("handoff file stayed resident in the page cache")
    else:
        rec["problems"].append("no handoff file after the export")
        rec["passed"] = False
        (cdir / "cycle.json").write_text(json.dumps(rec, indent=1) + "\n")
        print(f"M1-B2-CYCLE {i + 1} passed=False problems={rec['problems']} (import phase skipped)", flush=True)
        return rec
    b = Server(args.gate, env, cdir / "import-boot.log", args.port)
    try:
        # Model load reads the artifact from the same drive: open the import window at the
        # armed line, not at process start, so the window's device reads are the handoff's.
        b.wait_for(lambda: ARMED_BOOT.search(b.text()) or DONE.search(b.text()) or ABORTED in b.text(),
                   900, "import armed at boot")
        rec["import_window_opened_after_done"] = bool(DONE.search(b.text()))
        s = Sampler(leaves + [top], b.proc.pid, cdir / "import-host.jsonl")
        b.wait_for(lambda: DONE.search(b.text()) or ABORTED in b.text(), 1800, "import DONE")
        rec["import_window"] = s.stop(leaves)
        rec["import"] = parse_import(b.text())
        done = rec["import"]["done"]
        if rec["import"]["aborted"] or done is None:
            rec["problems"].append("import aborted or never finished")
        elif done["skipped"]:
            rec["problems"].append(f"import skipped {done['skipped']} entries")
        b.wait_for(b.ready, 900, "import boot ready")
        rec["probes"] = []
        for p, ref in zip(probes(prompts), reference):
            r = b.post(p, 48)
            cached = r["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)
            same = r["choices"][0]["text"] == ref
            rec["probes"].append({"cached_tokens": cached, "identical": same})
            if cached <= 0:
                rec["problems"].append("a probe missed the restored prefix (cached_tokens 0)")
            if not same:
                rec["problems"].append("a probe's text differs from the cold reference")
    finally:
        rec["import_boot_exit"] = b.stop()
    rec["passed"] = not rec["problems"]
    (cdir / "cycle.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(f"M1-B2-CYCLE {i + 1} passed={rec['passed']} problems={rec['problems']}", flush=True)
    return rec


def run(args):
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=False)
    if args.stub_no_lock:
        B.require(Path(args.gate).name.startswith("m1-stub") and Path(args.server).name.startswith("m1-stub"),
                  "--stub-no-lock is only for stub servers")
        lock = {"stub": True}
    else:
        proc = subprocess.run([sys.executable, str(ROOT / "tools/tier-lock-proof.py"), "--fd", str(args.lock_fd),
                               "--lock", B.LOCKS[args.rig]], pass_fds=(args.lock_fd,), capture_output=True, text=True)
        B.require(proc.returncode == 0, f"lock proof failed: {proc.stderr.strip()}")
        lock = json.loads(proc.stdout)
    proof, identity, leaves, top = RUNNER.proof_view(args.proof)
    B.require(B.filesystem_identity(Path(args.scratch)) == identity, "scratch is not on the proven filesystem")
    prompts = [json.loads(l)["prompt"] for l in Path(args.prompts).read_text().splitlines() if l.strip()]
    if not args.stub_no_lock:
        manifest = json.loads((HERE / "m1-prereg/b2-prompts.manifest.json").read_text())
        B.require(RUNNER.sha(args.prompts) == manifest["sha256"], "prompt set differs from the registered manifest")
    ident = {"driver_sha256": RUNNER.sha(__file__), "gate_sha256": RUNNER.sha(args.gate),
             "server_sha256": RUNNER.sha(args.server), "prompts_sha256": RUNNER.sha(args.prompts),
             "proof_sha256": RUNNER.sha(args.proof), "size_bytes": args.size_bytes, "cycles": args.cycles,
             "lock": lock, "leaves": leaves, "top": top, "qualified": False}
    (args.out / "identity.json").write_text(json.dumps(ident, indent=1) + "\n")
    ref = Server(args.server, base_env(args, 0, None), args.out / "reference-boot.log", args.port)
    try:
        ref.wait_for(ref.ready, 900, "reference boot ready")
        reference = [ref.post(p, 48)["choices"][0]["text"] for p in probes(prompts)]
    finally:
        ref.stop()
    (args.out / "reference.json").write_text(json.dumps(reference, indent=1) + "\n")
    cycles = [cycle(args, i, prompts, reference, leaves, top, identity) for i in range(args.cycles)]
    ok = [c for c in cycles if c["passed"]]

    def med(key, sub):
        values = [c[key][sub] for c in ok if c.get(key) and c[key].get(sub) is not None]
        return statistics.median(values) if values else None
    summary = {"cycles": len(cycles), "passed": len(ok), "size_bytes": args.size_bytes,
               "median_export_ms": med("export", "ms"), "median_fsync_ms": med("export", "fsync_ms"),
               "median_import_s": (statistics.median([c["import"]["done"]["seconds"] for c in ok]) if ok else None),
               "n": len(ok), "qualified": False}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("M1-B2-SUMMARY " + json.dumps(summary), flush=True)
    return 0 if len(ok) == len(cycles) else 3


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for name in ("--gate", "--server", "--artifact", "--prompts", "--proof", "--scratch", "--out"):
        r.add_argument(name, required=True)
    r.add_argument("--size-bytes", type=int, required=True)
    r.add_argument("--cycles", type=int, default=5)
    r.add_argument("--rig", choices=["pro-single", "rtx5090"], default="pro-single")
    r.add_argument("--lock-fd", type=int)
    r.add_argument("--port", type=int, default=18119)
    r.add_argument("--stub-no-lock", action="store_true")
    args = ap.parse_args(argv)
    B.require(args.stub_no_lock or args.lock_fd is not None, "--lock-fd (inherited canonical lock) required")
    B.require(args.cycles >= 5 or args.stub_no_lock, "registered protocol is 5 cycles per size")
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
