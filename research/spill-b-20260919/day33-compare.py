#!/usr/bin/env python3
"""Day 33 memra#680 verdicts (DAY33.md section 1.6). Reads per-boot rows.jsonl, shape.txt, REPORT.txt and
server.log written by run-day26-cell.sh + day31-parse.py (the client is day33-client.py) and prints the
pre-registered lines. Fixed before the first boot; never edits its inputs.

usage: day33-compare.py --card <label> <role>:<shape>:<boot-dir> ...
       role  = red (main) | green (the fix)
       shape = off | G1 | G2 | R32 | R64 | ... (DAY33.md 1.3); an ON shape's v is read from shape.txt
"""
import argparse, collections, gzip, json, os, re

ap = argparse.ArgumentParser()
ap.add_argument("--card", required=True)
ap.add_argument("boots", nargs="+")
a = ap.parse_args()
OPEN = ("i", "iii", "iv-a", "iv-b")
BOUNDED = ("warmup", "ii")
CRASH = re.compile(r"panicked|\[worker\] PANIC|\[worker\] FATAL|\[worker\] respawn|argmax sentinel|spec verify refused")
OOM = re.compile(r"CUDA_ERROR_OUT_OF_MEMORY")
KV = re.compile(r"(\w+)=(\S+)")


def log_lines(d):
    p = os.path.join(d, "server.log")
    if os.path.exists(p):
        return "server.log", open(p, encoding="utf-8", errors="replace")
    return "server.log.gz", gzip.open(p + ".gz", "rt", encoding="utf-8", errors="replace")


def first_cut(line):
    return line.rstrip().split(chr(0x2014))[0].strip()[:220]


boots = []
for spec in a.boots:
    role, shape_name, d = spec.split(":", 2)
    shape = dict(KV.findall(open(os.path.join(d, "shape.txt")).read()))
    on = shape.get("admit_by_memory") == "1"
    v = int(shape["open_output_tokens"]) if on else None
    rep = open(os.path.join(d, "REPORT.txt")).read()
    vb = re.search(r"^DAY31 V-BOOT .*$", rep, re.M)
    rows = [json.loads(l) for l in open(os.path.join(d, "rows.jsonl"))]
    marks = json.load(open(os.path.join(d, "marks.json"))) if os.path.exists(os.path.join(d, "marks.json")) else {}
    b0, b1 = marks.get("burst_start_ms"), marks.get("burst_end_ms")
    name, f = log_lines(d)
    crash = oom = 0; first_crash = first_oom = None
    admits = []; refuse_burst = 0; mem_id_lines = 0
    with f:
        for n, line in enumerate(f, 1):
            if CRASH.search(line):
                crash += 1; first_crash = first_crash or f"{name}:{n} {first_cut(line)}"
            if OOM.search(line):
                oom += 1; first_oom = first_oom or f"{name}:{n} {first_cut(line)}"
            if "[admit-mem] id=" in line:
                mem_id_lines += 1
                m = re.match(r"(\d+) ", line); t = int(m.group(1)) if m else None
                kv = dict(KV.findall(line))
                in_burst = bool(b0 and b1 and t is not None and b0 <= t <= b1 + 60000)
                if kv.get("verdict") == "admit":
                    admits.append(dict(kv, t=t, burst=in_burst, line=f"{name}:{n}"))
                if kv.get("verdict") == "refuse" and in_burst:
                    refuse_burst += 1
    boots.append(dict(role=role, shape=shape_name, dir=d, name=os.path.basename(d.rstrip("/")), on=on, v=v,
                      boot_pass=bool(vb and vb.group(0).endswith("PASS")), vboot=vb.group(0) if vb else None,
                      rows=rows, crash=crash, first_crash=first_crash, oom=oom, first_oom=first_oom,
                      admits=admits, refuse_burst=refuse_burst, mem_id_lines=mem_id_lines))

print(f"== card {a.card}")
for b in boots:
    print((b["vboot"] or f"DAY31 V-BOOT boot={b['name']} -> FAIL (no V-BOOT line)").replace("DAY31", "DAY33", 1)
          + f" role={b['role']} shape={b['shape']}")

print("== R-OOM / G-NOOM / G-BOOK (ON boots)")
green_ok = True
for b in [x for x in boots if x["on"]]:
    burst = [r for r in b["rows"] if r["class"] == "burst"]
    st = collections.Counter(r["status"] for r in burst)
    n429 = [r for r in burst if r["status"] == 429]
    ra_ok = all(r.get("retry_after") and str(r["retry_after"]).isdigit() and 1 <= int(r["retry_after"]) <= 60 for r in n429)
    other = [r for r in burst if r["status"] not in (200, 429)]
    red = b["oom"] > 0 or st.get(503, 0) > 0
    print(f"DAY33 R-OOM card={a.card} boot={b['name']} role={b['role']} shape={b['shape']} v={b['v']} "
          f"oom_lines={b['oom']} burst={dict(st)} first_oom={b['first_oom'] or 'none'} -> {'RED' if red else 'NOT-RED'}")
    noom = (b["oom"] == 0 and st.get(503, 0) == 0 and b["crash"] == 0 and st.get(200, 0) >= 1 and not other
            and ra_ok and len(n429) == b["refuse_burst"])
    adm = b["admits"]
    bad = [x["line"] for x in adm if not (x.get("est_bytes", "").isdigit() and x.get("device_free", "").isdigit()
                                          and int(x["est_bytes"]) <= int(x["device_free"]) and "pending_prime" in x)]
    adm_burst = sum(1 for x in adm if x["burst"])
    book = bool(adm) and not bad and adm_burst >= st.get(200, 0)
    print(f"DAY33 G-NOOM card={a.card} boot={b['name']} role={b['role']} shape={b['shape']} oom_lines={b['oom']} "
          f"burst_503={st.get(503, 0)} crash_lines={b['crash']} burst_200={st.get(200, 0)} other_non200={len(other)} "
          f"r429={len(n429)} refuse_lines={b['refuse_burst']} retry_after_in_1_60={ra_ok} -> {'PASS' if noom else 'FAIL'}")
    print(f"DAY33 G-BOOK card={a.card} boot={b['name']} role={b['role']} shape={b['shape']} admit_lines={len(adm)} "
          f"admit_lines_in_burst={adm_burst} est_over_booked_free={len(bad)}{' ' + ','.join(bad[:4]) if bad else ''} "
          f"-> {'PASS' if book else 'FAIL'}")
    if b["role"] == "green":
        green_ok &= noom and book


def seq(b):
    return {r["tag"]: r for r in b["rows"] if r["class"] != "burst"}


def same(r0, r1):
    return (r0["message_sha256"] == r1["message_sha256"] and r0["G"] == r1["G"]
            and r0.get("finish_reason") == r1.get("finish_reason"))


print("== V-ID-FIX (green ON against red ON, same shape, sequential rows)")
idfix_ok = True
for g in [x for x in boots if x["on"] and x["role"] == "green"]:
    for r in [x for x in boots if x["on"] and x["role"] == "red" and x["shape"] == g["shape"]]:
        s0, s1 = seq(r), seq(g); elig = eq = 0; diff = []
        for t, r1 in s1.items():
            r0 = s0.get(t)
            if r0 and r0["status"] == 200 and r1["status"] == 200 and r0["prompt_sha256"] == r1["prompt_sha256"]:
                elig += 1
                if same(r0, r1):
                    eq += 1
                else:
                    diff.append(t)
        ok = elig > 0 and eq == elig; idfix_ok &= ok
        print(f"DAY33 V-ID-FIX card={a.card} shape={g['shape']} green={g['name']} red={r['name']} eligible={elig} "
              f"equal={eq} differ={len(diff)}{' ' + ','.join(diff[:8]) if diff else ''} -> {'PASS' if ok else 'FAIL'}")

print("== V-ID (the day-32 term: green ON against green OFF)")
id_ok = True
offs = [x for x in boots if not x["on"] and x["role"] == "green"]
for g in [x for x in boots if x["on"] and x["role"] == "green"]:
    for o in offs:
        s0, s1 = seq(o), seq(g); elig = eq = 0; diff = []
        for t, r1 in s1.items():
            r0 = s0.get(t)
            if not (r0 and r0["status"] == 200 and r1["status"] == 200 and r0["prompt_sha256"] == r1["prompt_sha256"]):
                continue
            if r1["class"] in BOUNDED or (r1["class"] in OPEN and r0.get("finish_reason") == "stop" and r0["G"] < g["v"]):
                elig += 1
                if same(r0, r1):
                    eq += 1
                else:
                    diff.append(t)
        ok = elig > 0 and eq == elig; id_ok &= ok
        print(f"DAY33 V-ID card={a.card} shape={g['shape']} on={g['name']} off={o['name']} eligible={elig} equal={eq} "
              f"differ={len(diff)}{' ' + ','.join(diff[:8]) if diff else ''} -> {'PASS' if ok else 'FAIL'}")

print("== V-OFF (green OFF against red OFF: the default-OFF program unchanged)")
off_ok = True
for g in offs:
    for r in [x for x in boots if not x["on"] and x["role"] == "red"]:
        s0, s1 = seq(r), seq(g); n = eq = 0; diff = []
        for t, r1 in s1.items():
            r0 = s0.get(t)
            n += 1
            if r0 and r0["status"] == r1["status"] == 200 and r0["prompt_sha256"] == r1["prompt_sha256"] and same(r0, r1):
                eq += 1
            else:
                diff.append(t)
        ok = n > 0 and eq == n and len(s0) == len(s1) and g["mem_id_lines"] == 0 and g["boot_pass"]
        off_ok &= ok
        print(f"DAY33 V-OFF card={a.card} green={g['name']} red={r['name']} rows={n} equal={eq} differ={len(diff)}"
              f"{' ' + ','.join(diff[:8]) if diff else ''} admit_mem_lines={g['mem_id_lines']} -> {'PASS' if ok else 'FAIL'}")

boot_ok = all(b["boot_pass"] for b in boots)
greens = [b for b in boots if b["role"] == "green"]
print(f"DAY33 VERDICT card={a.card} boots={len(boots)} v_boot_all={boot_ok} green_noom_book_all={green_ok} "
      f"v_id_fix_all={idfix_ok} v_id_all={id_ok} v_off_all={off_ok} -> "
      f"{'GREEN' if greens and boot_ok and green_ok and idfix_ok and id_ok and off_ok else 'NOT-GREEN'}")
