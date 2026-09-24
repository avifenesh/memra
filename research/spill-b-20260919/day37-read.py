#!/usr/bin/env python3
"""WP-B day 37 reader (DAY37.md 1.6 and addendum B): the deciding cell's clauses A1 to A7 from the committed
receipts of one card. Written before the serving cells ran; every verdict line is printed verbatim for section 2.

usage: day37-read.py <card> <card root>
  <card root>/grow-*/receipt/GROW.txt                      A2
  <card root>/gates-pooled/, gates-vmm/                    A1 gate set (battery.log, <cell>/gate.log, CELL.txt)
  <card root>/boots/<name>/ with <name>.arm.txt            the serving boots (mix, stream, burst, fault, main)

Boot names (fixed by the chain): mix-<kind>-<order>-<arm> for the mix (kind spec|plain, order O1|O2), stream-<order>-
<k>-<arm> for the stream pairs, burst-<shape>-<arm> for the A7 bursts (shape g2|l64|boff), fault-<what> for A5,
off-<lane|main> for A6.
"""
import glob, json, math, os, re, statistics, sys

card, root = sys.argv[1], sys.argv[2]
# The gate-set directory prefix: `gates-r3` (addenda C and D, the default) or `gates` (design v1's banked set).
GATES = sys.argv[3] if len(sys.argv) > 3 else "gates-r3"
GRAN = 2 * 1024 * 1024
out = []


def say(line):
    out.append(line)
    print(line)


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def boots():
    b = {}
    for arm_txt in glob.glob(os.path.join(root, "boots", "*.arm.txt")):
        name = os.path.basename(arm_txt)[: -len(".arm.txt")]
        meta = dict(l.split("=", 1) for l in read(arm_txt).splitlines() if "=" in l)
        b[name] = dict(meta=meta, dir=os.path.join(root, "boots", name))
    return b


B = boots()


def rows(name):
    out_rows = {}
    for l in read(os.path.join(B[name]["dir"], "client.jsonl")).splitlines():
        r = json.loads(l)
        if r.get("arm") == "warmup":
            continue
        out_rows[r["tag"]] = r
    return out_rows


def digest(r):
    return r.get("content_sha256") or r.get("message_sha256") or r.get("text_sha256")


def comparable(r):
    """A row with a completion the program produced (HTTP 200, a finish that is not an error)."""
    return r.get("status") == 200 and r.get("finish_reason") not in (None, "error") and digest(r) is not None


def compare(a, b, skip=()):
    ra, rb = rows(a), rows(b)
    tags = sorted(set(ra) | set(rb))
    equal = differ = excluded = 0
    diffs = []
    for t in tags:
        if t in skip:
            excluded += 1
            continue
        x, y = ra.get(t), rb.get(t)
        if x is None or y is None or not comparable(x) or not comparable(y):
            excluded += 1
            continue
        if digest(x) == digest(y):
            equal += 1
        else:
            differ += 1
            diffs.append(t)
    return equal, differ, excluded, diffs


def log(name):
    return read(os.path.join(B[name]["dir"], "server.log"))


FAULT_RE = re.compile(r"panicked at|CUDA_ERROR_ILLEGAL_ADDRESS|illegal memory access|\bFATAL\b|respawn")
OOM_RE = re.compile(r"CUDA_ERROR_OUT_OF_MEMORY|out of memory")


def faults(name):
    t = log(name)
    return len(FAULT_RE.findall(t)), len(OOM_RE.findall(t))


def door(name):
    t = log(name)
    return (len(re.findall(r"\[kv-vmm\] door=ON", t)), len(re.findall(r"\[kv-vmm\] door=OFF", t)))


def pct(v, q):
    v = sorted(v)
    if not v:
        return float("nan")
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


# ---- A2 --------------------------------------------------------------------------------------------------------
for g in sorted(glob.glob(os.path.join(root, "grow-*", "receipt", "GROW.txt"))):
    say(f"DAY37 A2 card={card} receipt={os.path.relpath(g, root)} " + read(g).splitlines()[0])

# ---- A1 gates --------------------------------------------------------------------------------------------------
VERDICT = re.compile(r"(ALL GREEN[^\n]*|serve-smoke: \d+ failed|PREFIX-NEWEST-TURN-FITS:[^\n]*-> (?:PASS|FAIL)|FAIL[^\n]*GATE[^\n]*)")
cells = sorted({os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, f"{GATES}-pooled", "*", "CELL.txt"))}
               | {os.path.basename(os.path.dirname(p)) for p in glob.glob(os.path.join(root, f"{GATES}-vmm", "*", "CELL.txt"))})
gate_pass = True
for c in cells:
    v = {}
    for arm in ("pooled", "vmm"):
        d = os.path.join(root, f"{GATES}-{arm}", c)
        gl = read(os.path.join(d, "gate.log"))
        m = VERDICT.findall(gl)
        cell = dict(re.findall(r"(door_on_lines|door_off_lines)=(\d+)", read(os.path.join(d, "CELL.txt"))))
        rc = read(os.path.join(d, "gate.exit")).strip()
        oks = len(re.findall(r"^\s*ok\b|\bok\s*$|: ok\b", gl, re.M))
        v[arm] = dict(verdict=m[-1] if m else "no-verdict", rc=rc, door=cell.get("door_on_lines", "?") + "/" + cell.get("door_off_lines", "?"), oks=oks)
    green = all(x["rc"] == "0" and ("ALL GREEN" in x["verdict"] or "serve-smoke: 0 failed" in x["verdict"] or "-> PASS" in x["verdict"]) for x in v.values())
    doors = v["vmm"]["door"].split("/")[0] not in ("0", "?") and v["pooled"]["door"].split("/")[0] in ("0",)
    same = v["pooled"]["oks"] == v["vmm"]["oks"]
    ok = green and doors and same
    gate_pass &= ok
    say(f"DAY37 A1-GATE card={card} cell={c} pooled=[rc={v['pooled']['rc']} {v['pooled']['verdict'][:90]}] "
        f"vmm=[rc={v['vmm']['rc']} {v['vmm']['verdict'][:90]}] ok_lines={v['pooled']['oks']}/{v['vmm']['oks']} "
        f"door_on/off pooled={v['pooled']['door']} vmm={v['vmm']['door']} -> {'PASS' if ok else 'FAIL'}")

# ---- A1 mix, A5 mapper path, A6 ---------------------------------------------------------------------------------
mix_pass = True
for kind in ("spec", "plain"):
    for order in ("O1", "O2"):
        a, b = f"mix-{kind}-{order}-pooled", f"mix-{kind}-{order}-vmm"
        if a not in B or b not in B:
            continue
        e, d, x, diffs = compare(a, b)
        fa, fb = faults(a), faults(b)
        dn = door(b)
        ok = d == 0 and e > 0 and fa[0] == 0 and fb[0] == 0 and dn[0] >= 1
        mix_pass &= ok
        say(f"DAY37 A1-MIX card={card} kind={kind} order={order} compared={e + d} equal={e} differ={d} "
            f"excluded={x} differ_tags={diffs[:6]} faults_pooled={fa[0]} faults_vmm={fb[0]} door_on_vmm={dn[0]} "
            f"-> {'PASS' if ok else 'FAIL'}")

# ---- A1 stream digests and A3 -----------------------------------------------------------------------------------
stream = {}
for n in B:
    m = re.match(r"stream-(O[12])-(\d+)-(pooled|vmm)$", n)
    if m:
        s = read(os.path.join(B[n]["dir"], "SUMMARY.txt")).strip()
        kv = dict(re.findall(r"(\w+)=([-\d.na]+)", s))
        stream[n] = dict(order=m.group(1), k=m.group(2), arm=m.group(3), kv=kv)
if stream:
    diff_rows = 0
    pairs = 0
    for n, st in stream.items():
        if st["arm"] != "vmm":
            continue
        twin = f"stream-{st['order']}-{st['k']}-pooled"
        if twin in B:
            e, d, x, _ = compare(twin, n)
            diff_rows += d
            pairs += 1
    say(f"DAY37 A1-STREAM card={card} pairs={pairs} differing_rows={diff_rows} -> {'PASS' if pairs and diff_rows == 0 else 'FAIL'}")

    def med(arm, key, order=None):
        v = [float(st["kv"][key]) for st in stream.values()
             if st["arm"] == arm and (order is None or st["order"] == order) and key in st["kv"]]
        return statistics.median(v) if v else float("nan"), len(v)

    rules = [("ttft_p50", "<=", 1.05), ("itl_p99", "<=", 1.05), ("tpot_p50", "<=", 1.02), ("out_tok_per_s", ">=", 0.98)]
    a3ii = True
    for order in (None, "O1", "O2"):
        parts = []
        for key, op, bound in rules:
            (p, n_p), (v, n_v) = med("pooled", key, order), med("vmm", key, order)
            r = v / p if p else float("nan")
            ok = (r <= bound) if op == "<=" else (r >= bound)
            if order is None:
                a3ii &= ok
            parts.append(f"{key} pooled={p:.2f}(N={n_p}) vmm={v:.2f}(N={n_v}) ratio={r:.4f} rule{op}{bound} {'ok' if ok else 'FAIL'}")
        say(f"DAY37 A3-ii card={card} order={order or 'both'} " + "; ".join(parts)
            + (f" -> {'PASS' if a3ii else 'FAIL'}" if order is None else ""))
    # A3 (i): the tick-top ensure walls and the owner grows that waited on a behind mapper.
    walls, waited = [], []
    for n, st in stream.items():
        if st["arm"] != "vmm":
            continue
        t = log(n)
        for m in re.finditer(r"\[kv-vmm\] ensure-walls n=\d+ us=([\d,]+)", t):
            walls += [int(x) for x in m.group(1).split(",")]
        waited += [int(m.group(1)) for m in re.finditer(r"\[kv-vmm\] grow .* owner_us=(\d+) waited=1", t)]
    ok = len(walls) >= 20 and pct(walls, 0.99) <= 500
    say(f"DAY37 A3-i card={card} placement=helper tick_ensure_walls N={len(walls)} p50_us={pct(walls, .5):.1f} "
        f"p99_us={pct(walls, .99):.1f} max_us={max(walls) if walls else float('nan')} owner_grows_waited N={len(waited)} "
        f"p99_us={pct(waited, .99):.1f} rule N>=20 p99<=500 -> {'PASS' if ok else 'FAIL'}")

# ---- A4 ---------------------------------------------------------------------------------------------------------
retire_ok = retire_n = 0
worst = 0
for n in B:
    if not n.endswith("-vmm"):
        continue
    for m in re.finditer(r"\[kv-vmm\] retire id=\S+ planes=(\d+) mapped=(\d+) reserved=(\d+) used=(\d+) slack_bytes=(\d+) booked=(\d+)", log(n)):
        planes, mapped, reserved, used, slack, booked = map(int, m.groups())
        bound = planes * GRAN + slack
        bound = math.ceil(bound / GRAN) * GRAN
        retire_n += 1
        retire_ok += mapped - used <= bound
        worst = max(worst, (mapped - used) - bound)
say(f"DAY37 A4 card={card} retires={retire_n} within_bound={retire_ok} worst_over_bound_bytes={max(worst, 0)} "
    f"rule mapped-used<=planes*granule+slack -> {'PASS' if retire_n and retire_ok == retire_n else 'FAIL'}")
for n in sorted(B):
    if n.startswith("mix-"):
        me = {}
        try:
            me = json.load(open(os.path.join(B[n]["dir"], "metrics-end.json")))
        except (OSError, ValueError):
            pass
        say(f"DAY37 A4-IDLE card={card} boot={n} driver_free={me.get('cuda_driver_free_bytes')} "
            f"pool_cached={me.get('cuda_pool_cached_bytes')} pool_reserved={me.get('cuda_pool_reserved_bytes')} "
            f"vmm_mapped={me.get('kv_vmm_mapped_bytes')} vmm_reserved={me.get('kv_vmm_reserved_bytes')} "
            f"spec_pool={me.get('spec_pool_entries')} continuation_pool={me.get('continuation_pool_entries')}")

# ---- A5 ---------------------------------------------------------------------------------------------------------
if "fault-mapper" in B and "mix-spec-O1-pooled" in B:
    e, d, x, diffs = compare("mix-spec-O1-pooled", "fault-mapper")
    t = log("fault-mapper")
    waited = len(re.findall(r"\[kv-vmm\] grow .* waited=1", t))
    f = faults("fault-mapper")
    ok = d == 0 and e > 0 and f[0] == 0 and waited > 0
    say(f"DAY37 A5-MAPPER card={card} compared={e + d} equal={e} differ={d} owner_grows_waited={waited} faults={f[0]} -> {'PASS' if ok else 'FAIL'}")
if "fault-ensure" in B and "burst-g2-vmm" in B:
    t = log("fault-ensure")
    failed = re.findall(r"\[kv-vmm\] ensure failed: (parked|not parked)[^\n]*", t)
    retry = len(re.findall(r"\[kv-vmm\] grow failed .*reclaim-retry", t))
    victims = set(re.findall(r"\[kv-vmm\] ensure failed:.*?\(model", t))
    ra = rows("fault-ensure")
    errored = sorted(tag for tag, r in ra.items() if r.get("status") != 200 or r.get("finish_reason") == "error")
    e, d, x, diffs = compare("burst-g2-vmm", "fault-ensure", skip=set(errored))
    f = faults("fault-ensure")
    ok = len(failed) >= 1 and retry >= 1 and d == 0 and f[0] == 0
    say(f"DAY37 A5-ENSURE card={card} reclaim_retry_lines={retry} outcomes={[x[0] for x in failed]} errored_rows={errored[:6]} "
        f"peers_compared={e + d} equal={e} differ={d} faults={f[0]} -> {'PASS' if ok else 'FAIL'}")
for which in ("build1", "build64"):
    n = f"fault-{which}"
    if n not in B:
        continue
    t = log(n)
    retry = len(re.findall(r"after cache alloc|evicted .* after alloc|spec pool evicted|reclaim-on-alloc-oom|retrying", t))
    refused = len(re.findall(r"cache alloc failed", t))
    ra = rows(n)
    non200 = sum(1 for r in ra.values() if r.get("status") != 200)
    f = faults(n)
    ok = f[0] == 0 and (retry >= 1 if which == "build1" else (refused >= 1 or non200 >= 1))
    say(f"DAY37 A5-{which.upper()} card={card} retry_lines={retry} cache_alloc_failed_lines={refused} non200_rows={non200} "
        f"faults={f[0]} -> {'PASS' if ok else 'FAIL'}")

# ---- A6 ---------------------------------------------------------------------------------------------------------
# The lane's OFF boot is the O1 pooled spec mix boot (the same binary, door unset, the same workload).
lane_off = "off-lane" if "off-lane" in B else "mix-spec-O1-pooled"
if lane_off in B and "off-main" in B:
    e, d, x, diffs = compare("off-main", lane_off)
    lane = log(lane_off)
    lines = [l for l in lane.splitlines() if "[kv-vmm]" in l]
    only_boot = len(lines) == 1 and "door=OFF" in lines[0]
    main_lines = sum(1 for l in log("off-main").splitlines() if "[kv-vmm]" in l)
    ok = d == 0 and e > 0 and only_boot and main_lines == 0
    say(f"DAY37 A6 card={card} lane_boot={lane_off} compared={e + d} equal={e} differ={d} lane_kv_vmm_lines={len(lines)} "
        f"main_kv_vmm_lines={main_lines} -> {'PASS' if ok else 'FAIL'}")

# ---- A7 ---------------------------------------------------------------------------------------------------------
a7 = True
for shape in ("g2", "l64", "boff"):
    n = f"burst-{shape}-vmm"
    if n not in B:
        continue
    t = log(n)
    f = faults(n)
    ra = rows(n)
    r503 = sum(1 for r in ra.values() if r.get("status") == 503)
    grow_fail = len(re.findall(r"\[kv-vmm\] (?:grow failed|ensure failed)", t))
    over = 0
    admits = 0
    for line in t.splitlines():
        if "[admit-mem] id=" not in line or " verdict=admit " not in line:
            continue
        kv = dict(re.findall(r" (\w+)=(\S+)", line))
        admits += 1
        over += int(kv["est_bytes"]) > int(kv["device_free"])
    status = {}
    for r in ra.values():
        status[r.get("status")] = status.get(r.get("status"), 0) + 1
    ok = f[1] == 0 and r503 == 0 and grow_fail == 0 and over == 0 and f[0] == 0
    a7 &= ok
    say(f"DAY37 A7 card={card} shape={shape} oom_lines={f[1]} r503={r503} grow_failures={grow_fail} admit_lines={admits} "
        f"est_over_booked_free={over} status={status} faults={f[0]} -> {'PASS' if ok else 'FAIL'}")

with open(os.path.join(root, "SUMMARY.txt"), "w") as fh:
    fh.write("\n".join(out) + "\n")
