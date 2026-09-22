#!/usr/bin/env python3
"""Day 23 reading of the promote stall cell on main's tree (DAY23.md pre-registration, fixed before the run).

Replays lane A's rule line from every receipt.json (the harness's own `--replay`), then applies the
pre-registered thresholds below and prints one `DAY23 STALL READING` line per question plus one summary
line. Thresholds are constants here; nothing is tuned after the run. References (same box, other sittings):
A day 16 OFF promote 85.0 / demote 117.5, ON promote 162.8 / demote 193.5; A day 18 run 2 ON promote 81.9
(server_promote_ms first pair 60.8, 61.9; steady 25.9 to 26.4, median 26.1), ON demote 149.7. The parked-only
wait of #627 is bounded at 2 ms per tick, so the growth the claim allows is 2.0 ms.

    day23-stall-reading.py <ev_dir>    # ev_dir holds off/ and on/, each with demote/ promote/ receipt.json and server.log
"""
import json
import os
import re
import statistics
import subprocess
import sys

DAY18_PROMOTE_STALL = 81.9
DAY18_PROMOTE_STEADY = 26.1
DAY18_PROMOTE_FIRST_PAIR_MAX = 61.9
DAY18_DEMOTE_STALL = 149.7
DAY16_OFF_PROMOTE_STALL = 85.0
DAY16_OFF_DEMOTE_STALL = 117.5
WAIT_MS = 2.0
HARNESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "spill-a-20260919", "stall_cell.py")


def load(ev, boot, arm):
    p = os.path.join(ev, boot, arm, "receipt.json")
    rec = json.load(open(p))
    rep = subprocess.run([sys.executable, HARNESS, "--replay", p], capture_output=True, text=True)
    replay_ok = "STALL REPLAY: PASS" in rep.stdout
    return rec, replay_ok


def line(q, verdict, detail):
    print(f"DAY23 STALL READING {q}: {verdict} ({detail})")
    return verdict


def main():
    ev = sys.argv[1]
    out = {}
    recs = {}
    admissible = True
    for boot in ("off", "on"):
        for arm in ("demote", "promote"):
            rec, ok = load(ev, boot, arm)
            recs[(boot, arm)] = rec
            s = rec["summary"]
            print(rec["rule_line"])
            print(f"  replay={'PASS' if ok else 'FAIL'} errors={len(s['errors'])} tenant_text_identical={len(s['tenant_text_shas']) == 1}")
            admissible &= ok and not s["errors"] and len(s["tenant_text_shas"]) == 1
    # admissibility of the ON promote arm (A's day-18 list)
    on_log = open(os.path.join(ev, "on", "server.log"), errors="replace").read()
    on_prom = recs[("on", "promote")]
    prom_runs = [r for r in on_prom["runs"] if r["arm"] == "promote"]
    joined = "\n".join(ln for r in prom_runs for ln in r["server_log_lines"])
    n_sub = joined.count("promote submitted off the tick")
    n_pub = joined.count("promote published off the tick")
    n_rcpt = joined.count("contracts door H2D receipt")
    cached = [r["intruder"].get("cached_tokens") for r in prom_runs]
    bad = re.findall(r"demote failed|promote failed|promote refused|TIER DISABLED", on_log)
    sync = on_log.count("settled synchronously by a promote")
    polls = re.findall(r"promote published off the tick: ticket complete after (\d+) poll\(s\), ([0-9.]+)ms", on_log)
    print(f"  ON promote admissibility: submitted={n_sub} published={n_pub} h2d_receipts={n_rcpt} cached_tokens={cached} "
          f"bad_lines={len(bad)} settled_synchronously_by_a_promote={sync}")
    print(f"  ON promote publish polls (recorded, no rule): {[int(p) for p, _ in polls]} submission_to_completion_ms={[float(m) for _, m in polls]}")
    adm_on = n_sub == 10 and n_pub == 10 and n_rcpt == 10 and all(c == 64 for c in cached) and not bad and sync == 0
    admissible &= adm_on
    print(f"  admissible={admissible}")
    sm = lambda b, a: recs[(b, a)]["summary"]["stall_ms"]["median"]
    p_on, p_off, d_on, d_off = sm("on", "promote"), sm("off", "promote"), sm("on", "demote"), sm("off", "demote")
    sp = recs[("on", "promote")]["summary"]["server_promote_ms"]
    steady = statistics.median(sp[2:]) if len(sp) >= 3 else float("nan")
    first_pair = max(sp[:2]) if len(sp) >= 2 else float("nan")
    out["P1 promote ON stall against day 18 (81.9)"] = line(
        "P1 promote ON stall against day 18 (81.9)",
        "at_or_under_day18" if p_on <= DAY18_PROMOTE_STALL else ("within_wait" if p_on <= DAY18_PROMOTE_STALL + WAIT_MS else "grew"),
        f"stall_median={p_on:.1f} bounds <=81.9 | <=83.9")
    out["P2 promote ON server_promote_ms steady against day 18 (26.1)"] = line(
        "P2 promote ON server_promote_ms steady against day 18 (26.1)",
        "within_wait" if steady <= DAY18_PROMOTE_STEADY + WAIT_MS else "grew",
        f"steady_median={steady:.1f} bound <=28.1; first_pair_max={first_pair:.1f} "
        f"{'first_pair_within_wait' if first_pair <= DAY18_PROMOTE_FIRST_PAIR_MAX + WAIT_MS else 'first_pair_grew'} bound <=63.9")
    out["P3 promote ON against OFF, same window"] = line(
        "P3 promote ON against OFF, same window",
        "on_at_off" if p_on <= 1.10 * p_off else "on_above_off",
        f"on={p_on:.1f} off={p_off:.1f} bound on<={1.10 * p_off:.1f}")
    out["P4 promote OFF regime against day 16 (85.0)"] = line(
        "P4 promote OFF regime against day 16 (85.0)",
        "off_stable" if abs(p_off - DAY16_OFF_PROMOTE_STALL) <= 0.10 * DAY16_OFF_PROMOTE_STALL else "off_moved",
        f"off={p_off:.1f} band 76.5..93.5")
    out["P5 demote ON against day 18 (149.7)"] = line(
        "P5 demote ON against day 18 (149.7)",
        "demote_half_unchanged" if abs(d_on - DAY18_DEMOTE_STALL) <= 0.10 * DAY18_DEMOTE_STALL else "demote_half_moved",
        f"on={d_on:.1f} band 134.7..164.7")
    out["P6 demote OFF regime against day 16 (117.5)"] = line(
        "P6 demote OFF regime against day 16 (117.5)",
        "off_stable" if abs(d_off - DAY16_OFF_DEMOTE_STALL) <= 0.10 * DAY16_OFF_DEMOTE_STALL else "off_moved",
        f"off={d_off:.1f} band 105.8..129.3")
    print(f"DAY23 STALL READING summary: admissible={admissible} " + " ".join(f"{k.split()[0]}={v}" for k, v in out.items()))
    return 0 if admissible else 1


if __name__ == "__main__":
    sys.exit(main())
