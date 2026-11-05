#!/usr/bin/env python3
"""The DAY38 addendum D reading of one card root, with the rule of DAY38 1.6 printed as one VERDICT line.

usage: maxq-verdict.py ROOT --card LABEL [--reader PATH] [--kit KIT --card-index N] [--regime gap|timed|workstation]

1. Runs the registered reader, day38d-read.py (unchanged; it writes ROOT/SUMMARY.txt), and prints its lines.
2. With --kit, the kit's validity checks (none of them is a clause of the rule): each boot ran the binary its arm
   names (kit/bins/SHA256SUMS), the off boots wrote no park-compact line and the on boots did (the arm took
   effect on this card), each boot's card was the Max-Q and the server's pid was on the pinned card at ready,
   the model and kit files matched their hashes. A failed check makes the verdict INVALID.
3. The rule (DAY38.md 1.6, re-registered by addendum D 1.11): PROMOTE-ELIGIBLE when P1' passes in both orders
   and P2' passes on both green fault boots; P3 and P4 go with it. FAIL (no reading) when either fails.
   INCOMPLETE when a boot the rule reads is absent (a part not yet run). The red arm is reported, not ruled on:
   its expected reading is FAIL; a PASS there means it did not exercise the defect, which is recorded.
"""
import argparse, glob, json, os, re, subprocess, sys

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--card", required=True)
ap.add_argument("--reader", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                 "research", "spill-b-20260919", "day38d-read.py"))
ap.add_argument("--kit", default=None)
ap.add_argument("--card-index", default=None)
ap.add_argument("--regime", default="gap", choices=("gap", "timed", "workstation"))
a = ap.parse_args()

MAXQ = "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition"
MODEL_SHA = "1facf36c2db359dcf9c2475cf8f85fe84a528d10aaaaff20f7c0db3d561e024a"
ARM_OF = {"main-O1-off": "off", "main-O1-on": "on", "main-O2-on": "on", "main-O2-off": "off",
          "fault-batch": "on", "fault-nobatch": "on", "fault-nobatch-red": "red"}
RULE_BOOTS = ["main-O1-off", "main-O1-on", "main-O2-on", "main-O2-off", "fault-batch", "fault-nobatch"]

r = subprocess.run([sys.executable, "-I", a.reader, a.card, a.root], capture_output=True, text=True)
sys.stdout.write(r.stdout)
if r.returncode != 0:
    sys.stdout.write(r.stderr)
    print(f"VERDICT DAY38D card={a.card} -> INVALID (the reader exited rc={r.returncode})")
    sys.exit(1)
lines = r.stdout.splitlines()


def bdir(n):
    return os.path.join(a.root, "boots", n)


def have(n):
    return os.path.exists(os.path.join(bdir(n), "client.jsonl"))


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def status(prefix):
    for l in lines:
        if l.startswith(prefix):
            return "PASS" if " -> PASS" in l else "FAIL"
    return None


p1 = {o: status(f"DAY38D P1 card={a.card} order={o} ") for o in ("O1", "O2")}
p2 = {b: status(f"DAY38D P2 card={a.card} boot={b} ") for b in ("fault-batch", "fault-nobatch")}
red = status(f"DAY38D P2 card={a.card} boot=fault-nobatch-red ")

problems = []
if a.kit:
    sums = {}
    for l in read(os.path.join(a.kit, "bins", "SHA256SUMS")).splitlines():
        if l.strip():
            h, p = l.split()
            sums[p.split("/")[0]] = h
    for n, arm in ARM_OF.items():
        if not have(n):
            continue
        at = dict(x.split("=", 1) for x in read(os.path.join(a.root, "boots", f"{n}.arm.txt")).splitlines() if "=" in x)
        if at.get("bin_sha256") != sums.get(arm):
            problems.append(f"{n}: binary {at.get('bin_sha256', '?')[:16]} is not arm {arm}")
        parks = len(re.findall(r"\[kv-reuse\] park-compact: ", read(os.path.join(bdir(n), "server.log"))))
        if arm == "off" and parks:
            problems.append(f"{n}: off arm wrote {parks} park-compact lines")
        if arm != "off" and not parks:
            problems.append(f"{n}: {arm} arm wrote no park-compact line")
        gpu = read(os.path.join(bdir(n), "gpu-before.csv"))
        if MAXQ not in gpu:
            problems.append(f"{n}: card is not the Max-Q ({gpu.strip().splitlines()[-1:]})")
        pid = read(os.path.join(bdir(n), "server.pid")).split()
        apps = read(os.path.join(bdir(n), "compute-apps-ready.csv"))
        if not pid or not any(l.split(",")[0].strip() == pid[0] for l in apps.splitlines()[1:]):
            problems.append(f"{n}: server pid {pid[:1]} not on the pinned card at ready")
        if MODEL_SHA not in read(os.path.join(bdir(n), "model.sha256")):
            problems.append(f"{n}: model sha256 mismatch")
        if read(os.path.join(bdir(n), "dirty.txt")).strip():
            problems.append(f"{n}: kit files differ from KIT-MANIFEST.sha256")
    ref = read(os.path.join(a.kit, "reference", "box-d", "stream.json")).strip()
    streams = {n: read(os.path.join(bdir(n), "stream.json")).strip() for n in ARM_OF if n.startswith("main-") and have(n)}
    if ref and streams:
        same = sorted(n for n, s in streams.items() if s and json.loads(s) == json.loads(ref))
        print(f"DAY38D STREAM card={a.card} same_prompt_stream_as_workstation_box_d={len(same)}/{len(streams)} "
              f"ref={json.loads(ref)['sha256'][:16]} (informational: the rule compares arms on one card only)")
    for t in sorted(glob.glob(os.path.join(a.root, "THERMAL-STOP*.txt")) + glob.glob(os.path.join(a.root, "part*-THERMAL-STOP.txt"))):
        problems.append(f"thermal stop recorded: {os.path.basename(t)}")

if a.regime == "gap":
    print(f"DAY38D REGIME card={a.card} co-tenant: other cards of the box were training; the P3 and P4 lines above "
          "are co-tenant readings, not readings for the record (run-timed.sh carries them)")
elif a.regime == "timed":
    print(f"DAY38D REGIME card={a.card} quiet box (run-timed.sh): P3 and P4 above are the cell's readings")

missing = [n for n in RULE_BOOTS if not have(n)]
red_txt = ("absent" if red is None else
           "FAIL, red as expected" if red == "FAIL" else
           "PASS, the red arm did not exercise the defect (recorded)")
head = (f"VERDICT DAY38D card={a.card} rule=DAY38-1.6 P1'=[O1:{p1['O1'] or 'absent'},O2:{p1['O2'] or 'absent'}] "
        f"P2'=[fault-batch:{p2['fault-batch'] or 'absent'},fault-nobatch:{p2['fault-nobatch'] or 'absent'}] "
        f"red=[fault-nobatch-red:{red_txt}]")
if problems:
    print(f"{head} -> INVALID ({'; '.join(problems[:8])})")
elif a.regime == "timed":
    print(f"{head} -> TIMED (the P3 and P4 readings only; the rule's verdict comes from the gap runs)")
elif missing:
    print(f"{head} -> INCOMPLETE (absent boots: {' '.join(missing)})")
elif all(v == "PASS" for v in list(p1.values()) + list(p2.values())):
    print(f"{head} -> PROMOTE-ELIGIBLE (P1' and P2' pass; P3 and P4 go with it)")
else:
    failed = [f"P1' {o}" for o, v in p1.items() if v != "PASS"] + [f"P2' {b}" for b, v in p2.items() if v != "PASS"]
    print(f"{head} -> FAIL (no reading: {', '.join(failed)} failed; the cause is quoted from the receipts)")
