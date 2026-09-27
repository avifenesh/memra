#!/usr/bin/env python3
"""Day 29: apply the V3-premise typed refusal to tools/prefix-newest-turn-fits-gate.py (exact-string edits, each once)."""
import sys
from pathlib import Path

path = Path(sys.argv[1])
src = path.read_text()

def rep(old: str, new: str) -> None:
    global src
    assert src.count(old) == 1, f"expected exactly one occurrence:\n{old}"
    src = src.replace(old, new)

# 1. docstring: the premise and the refusal.
rep(
"""  Recorded, not judged: the per-turn `[primeseg] call start=.. grid_off=..` receipts
  (MEMRA_DEBUG_PRIMESEG=1, an existing diagnostic) and the count of off-grid prime-call starts.
""",
"""  Recorded, not judged: the per-turn `[primeseg] call start=.. grid_off=..` receipts
  (MEMRA_DEBUG_PRIMESEG=1, an existing diagnostic) and the count of off-grid prime-call starts.
  V3 premise:     V3 compares the two boots' states, so it presumes they retain the SAME
                  non-prefix-cache device state after every send. The admission's
                  `[admit-oom] reclaim-on-defer` releases parked plain/spec/dspark sessions when
                  a request's cost plus the reserve floor exceeds effective free; when it fires
                  in one boot's window and not in the other's (the cache-on boot carries the
                  cache's resident bytes, so on a card close to the floor it crosses first), the
                  premise does not hold and V3 cannot be decided. The gate then REFUSES (exit 2),
                  typed `REFUSED: V3 premise: ...`, naming the windows, the released sessions
                  per boot and the card at each boot (driver free, compute-apps); the verdict
                  line V1..V6 would have printed is kept in summary.json under
                  `verdict_under_broken_premise`, never printed as the verdict. Two boots that
                  release identically keep V3 evaluated exactly as above; V3's clause, form and
                  slack are unchanged (spill-b day 29: the local RTX 5090 read `V3=FAIL` in both
                  door arms with a constant `-410352980` B error, one cohort-shaped parked plain
                  session released by the cache-on boot's turn-2 reclaim beside a 1.4 GB
                  co-tenant, while the 96 GB card read `-> PASS`).
""")

# 2. the reclaim line, parsed (RE_RECLAIM stays for the counter).
rep(
"""RE_RECLAIM = re.compile(r"\\[admit-oom\\] reclaim-on-defer: ")
""",
"""RE_RECLAIM = re.compile(r"\\[admit-oom\\] reclaim-on-defer: ")
# The same line, with its counts: parked sessions released per pool and the effective free it moved.
RE_RECLAIM_PARKED = re.compile(
    r"\\[admit-oom\\] reclaim-on-defer: evicted (\\d+) prefix entries \\+ (\\d+) plain \\+ (\\d+) spec \\+ (\\d+) dspark "
    r"parked sessions \\(global LRU\\); effective free (\\d+)MB -> (\\d+)MB"
)
PARKED_POOLS = ("plain", "spec", "dspark")
""")

# 3. parse_window records the releases.
rep(
"""    ev: dict = {"inserts": [], "hits": [], "evicts": [], "refused": [], "skipped": [], "demotes": 0, "reclaims": 0, "lines": [], "route": [], "primeseg": []}
""",
"""    ev: dict = {"inserts": [], "hits": [], "evicts": [], "refused": [], "skipped": [], "demotes": 0, "reclaims": 0, "lines": [], "route": [], "primeseg": [], "parked_releases": []}
""")
rep(
"""        if RE_RECLAIM.search(ln):
            ev["reclaims"] += 1
    return ev
""",
"""        if RE_RECLAIM.search(ln):
            ev["reclaims"] += 1
            m = RE_RECLAIM_PARKED.search(ln)
            if m:
                ev["parked_releases"].append({
                    "prefix": int(m.group(1)), "plain": int(m.group(2)), "spec": int(m.group(3)), "dspark": int(m.group(4)),
                    "effective_free_before_mb": int(m.group(5)), "effective_free_after_mb": int(m.group(6)), "line": ln.strip(),
                })
    return ev


def parked_released(window: dict) -> tuple[int, int, int]:
    \"\"\"Parked sessions a window's reclaim-on-defer lines released, per pool (plain, spec, dspark).\"\"\"
    return tuple(sum(r[pool] for r in window.get("parked_releases", [])) for pool in PARKED_POOLS)


def v3_premise_rows(cal: dict, rec: dict) -> list[dict]:
    \"\"\"Window by window (each cohort send, then each turn), what each boot's reclaim released.

    V3's state equation presumes the two boots retain the same parked sessions after every send;
    a window whose releases differ between the boots breaks that premise for every later state.
    \"\"\"
    rows = []
    for c, m in zip(cal["cohort"], rec["cohort"]):
        for j, (cw, mw) in enumerate(zip(c.get("windows", []), (m["window_first"], m["window_second"])), start=1):
            rows.append({"who": f"cohort {c['tokens']} send {j}", "calibration": parked_released(cw), "measured": parked_released(mw)})
    for ct, mt in zip(cal["turns"], rec["turns"]):
        rows.append({"who": f"turn {mt['turn']}", "calibration": parked_released(ct["window"]), "measured": parked_released(mt["window"])})
    for r in rows:
        r["equal"] = tuple(r["calibration"]) == tuple(r["measured"])
    return rows


def card_listing() -> dict:
    \"\"\"The card right now: driver free of total (MiB) and the compute-apps listing (read only).\"\"\"
    def q(*args: str) -> str:
        return subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()
    apps = [ln.strip() for ln in q("--query-compute-apps=pid,process_name,used_memory").splitlines() if ln.strip()]
    return {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "memory_free_total_mib": q("--query-gpu=memory.free,memory.total"), "compute_apps": apps}


def card_brief(card: dict | None) -> str:
    if not card:
        return "card not sampled"
    apps = []
    for a in card["compute_apps"]:
        parts = [p.strip() for p in a.split(",")]
        if len(parts) >= 3:
            apps.append(f"{Path(parts[1]).name} {parts[2]} MiB")
    return f"free/total MiB {card['memory_free_total_mib']}, compute-apps {len(apps)} [{'; '.join(apps)}]"


def v3_premise_refusal(rows: list[dict], cal_card: dict | None, meas_card: dict | None, budget: int) -> str:
    broken = [r for r in rows if not r["equal"]]
    detail = "; ".join(
        f"{r['who']}: calibration released plain/spec/dspark {'/'.join(map(str, r['calibration']))}, measured {'/'.join(map(str, r['measured']))}"
        for r in broken
    )
    return (
        f"V3 premise: `[admit-oom] reclaim-on-defer` released parked sessions in one boot without the counterpart "
        f"in the other on {len(broken)} window(s) ({detail}); the state equation compares boots whose retained "
        f"parked-session sets differ, so V3 is not decidable on this card shape (budget {budget} B, MEMRA_CTX 16384, "
        f"MEMRA_MAX_SESSIONS 4); card at the calibration boot: {card_brief(cal_card)}; at the measured boot: "
        f"{card_brief(meas_card)}; the clause verdicts are kept in summary.json (verdict_under_broken_premise), not printed as a verdict"
    )
""")

# 4. the server samples the card right before it boots.
rep(
"""        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.offset = 0
""",
"""        self.proc: subprocess.Popen | None = None
        self.base = f"http://127.0.0.1:{port}"
        self.offset = 0
        self.card_at_boot: dict | None = None
""")
rep(
"""        self.log.parent.mkdir(parents=True, exist_ok=True)
        with self.log.open("wb") as log:
""",
"""        self.log.parent.mkdir(parents=True, exist_ok=True)
        self.card_at_boot = card_listing()
        with self.log.open("wb") as log:
""")

# 5. the calibration boot parses its cohort windows and records its card.
rep(
"""    cal: dict = {"boot_lines": [], "cohort": [], "turns": []}
    try:
""",
"""    cal: dict = {"boot_lines": [], "cohort": [], "turns": [], "card_at_boot": srv.card_at_boot}
    try:
""")
rep(
"""            sends = []
            for _ in range(2):
                r = srv.complete(ids, "cohort")
                time.sleep(0.5)
                srv.new_log_lines()
                if r["status"] != 200:
                    refuse(f"calibration: cohort prompt of {n} tokens was not served: {r['error']}")
                sends.append(r)
            cal["cohort"].append({"tokens": n, "first": sends[0], "second": sends[1], "metrics_after": srv.settled_metrics()})
""",
"""            sends = []
            windows = []
            for _ in range(2):
                r = srv.complete(ids, "cohort")
                time.sleep(0.5)
                windows.append(parse_window(srv.new_log_lines()))
                if r["status"] != 200:
                    refuse(f"calibration: cohort prompt of {n} tokens was not served: {r['error']}")
                sends.append(r)
            cal["cohort"].append({"tokens": n, "first": sends[0], "second": sends[1], "windows": windows, "metrics_after": srv.settled_metrics()})
""")

# 6. the measured boot records its card.
rep(
"""        rec["boot"] = {"budget_bytes": budget, "policy": policy, "line": next(ln.strip() for ln in boot_lines if RE_ON.search(ln))}
""",
"""        rec["boot"] = {"budget_bytes": budget, "policy": policy, "line": next(ln.strip() for ln in boot_lines if RE_ON.search(ln)), "card_at_boot": srv.card_at_boot}
""")

# 7. the premise decides between the verdict line and the typed refusal; receipts are written either way.
rep(
"""    summary = {
        "verdict": verdict,
        "pass": ok,
""",
"""    premise_rows = v3_premise_rows(cal, rec)
    premise_ok = all(r["equal"] for r in premise_rows)
    refusal = None if premise_ok else v3_premise_refusal(premise_rows, cal.get("card_at_boot"), rec["boot"].get("card_at_boot"), rec["boot"]["budget_bytes"])
    summary = {
        "verdict": verdict if premise_ok else f"REFUSED: {refusal}",
        "verdict_under_broken_premise": None if premise_ok else verdict,
        "pass": ok and premise_ok,
        "v3_premise": {"holds": premise_ok, "rows": premise_rows},
""")
rep(
"""    (args.out / "VERDICT.txt").write_text(verdict + "\\n")
""",
"""    (args.out / "VERDICT.txt").write_text(summary["verdict"] + "\\n")
""")
rep(
"""    print("\\n".join(table))
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)
""",
"""    print("\\n".join(table))
    for r in premise_rows:
        if not r["equal"]:
            print(f"premise: {r['who']}: calibration released plain/spec/dspark {r['calibration']}, measured {r['measured']}")
    if not premise_ok:
        refuse(refusal)
    print(verdict, flush=True)
    sys.exit(0 if ok else 1)
""")

path.write_text(src)
print(f"patched {path}")
