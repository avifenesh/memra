#!/usr/bin/env python3
"""r6_parts.py prof.sqlite TTFT_S PROMPT_TOKENS: the decode window of the last request (as r5_decode.py)
as one round's parts, in the round 5 scout table's rows: ms per round of each part and its floor at
226 GB/s. A round is one verify launch of the output head (qmatvec_q5_K_mmvq_b4_r2).

Rows: trunk matvecs (NVFP4 verify), output head, draft (the MTP block's matvecs and its 32k head),
verify attention (walk and combine apart), draft attention, GDN scan and conv, rollback copies (device
copies and copy_batch_uniform_f32), glue kernels, and launch gaps and host (idle between kernels that
no copy fills).

The prefix cache publishes the request's prime boundary once, at the first burst boundary: per KV
plane a zeroed allocation and a device copy (copies and memsets of 64 MB or more). That is one event
per cold request, so it is listed apart and its time is taken out of the window before the per-round
division."""
import sqlite3
import sys

db, ttft, ptok = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
con = sqlite3.connect(db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
start = 0
prev = rows[0][1]
for i in range(1, len(rows)):
    if rows[i][0] - prev > 1e9:
        start = i
    prev = max(prev, rows[i][1])
t0 = rows[start][0] + int(ttft * 1e9)
t_end = max(r[1] for r in rows[start:])
win = [(s, e, n.split("(")[0]) for s, e, n in rows[start:] if s >= t0]


def part(name):
    if name.startswith("qmatvec_nvfp4") and "_mr2" not in name:
        return "trunk"
    if name.startswith("qmatvec_q5_K"):
        return "head"
    if name.startswith("qmatvec_"):
        return "draft"
    if name in ("fa_verify_rows_v4_deep", "fa_decode_vec_q_seqs_v4_deep"):
        return "vwalk"
    if name == "fa_decode_combine_seqs":
        return "vcomb"
    if name.startswith("fa_"):
        return "dattn"
    if name.startswith("gdn_") or name.startswith("ssm_conv"):
        return "gdn"
    if name == "copy_batch_uniform_f32":
        return "rbk"
    return "glue"


ms = {}
busy = 0.0
pe = win[0][0]
for s, e, n in win:
    ms[part(n)] = ms.get(part(n), 0.0) + (e - s) / 1e6
    if e > pe:
        busy += (e - max(s, pe)) / 1e6
        pe = e
rounds = sum(1 for s, e, n in win if n.startswith("qmatvec_q5_K_mmvq_b4_r2"))
window = (t_end - win[0][0]) / 1e6
idle = window - busy
big = 64e6
cp = con.execute("SELECT end - start, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ?", (win[0][0], t_end)).fetchall()
mset = con.execute("SELECT end - start, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET WHERE start >= ? AND end <= ?", (win[0][0], t_end)).fetchall()
cp_small = sum(d for d, b in cp if b < big) / 1e6
cp_big = sum(d for d, b in cp if b >= big) / 1e6
ms_big = sum(d for d, b in mset if b >= big) / 1e6
n_big = sum(1 for d, b in cp if b >= big)
# the publication's whole kernel gap (allocation and host time included), when there is one
event = 0.0
big_ops = con.execute("SELECT min(start), max(end) FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start >= ? AND end <= ? AND bytes >= ?",
                      (win[0][0], t_end, big)).fetchone()
if big_ops[0] is not None:
    before = max(e for s, e, n in win if e <= big_ops[0])
    after = min(s for s, e, n in win if s >= big_ops[1])
    event = (after - before) / 1e6
kv_gb = ptok * 1856 / 1e9
bw = 226.0
floors = {"trunk": 13.70 / bw * 1e3, "head": 0.874 / bw * 1e3, "draft": 1.07 / bw * 1e3,
          "vattn": kv_gb * 16 / bw * 1e3, "dattn": kv_gb * 3 / bw * 1e3, "gdn": 0.30 / bw * 1e3}
R = rounds
out = [
    ("trunk matvecs, verify at T=4 (NVFP4)", ms.get("trunk", 0) / R, floors["trunk"]),
    ("output head, verify (Q5_K)", ms.get("head", 0) / R, floors["head"]),
    ("draft: 3 steps of the MTP block and its 32k head", ms.get("draft", 0) / R, floors["draft"]),
    ("verify attention, 16 layers: the walk", ms.get("vwalk", 0) / R, floors["vattn"]),
    ("verify attention: the combine", ms.get("vcomb", 0) / R, 0.0),
    ("draft attention, the MTP layer, 3 steps", ms.get("dattn", 0) / R, floors["dattn"]),
    ("GDN scan and conv", ms.get("gdn", 0) / R, floors["gdn"]),
    ("rollback copies", (ms.get("rbk", 0) + cp_small) / R, 0.0),
    ("norms, quantize, glue, argmax", ms.get("glue", 0) / R, 0.0),
    ("launch gaps and host", (idle - cp_small - event) / R, 0.0),
]
print(f"prompt {ptok}: decode window {window:.1f} ms, {R} rounds; kernels {busy:.1f} ms, idle {idle:.1f} ms; "
      f"prefix-cache publication: one gap of {event:.1f} ms ({n_big} plane copies, "
      f"{sum(b for d, b in cp if b >= big) / 1e6:.0f} MB, copies {cp_big:.1f} ms, zeroing {ms_big:.1f} ms); "
      f"{(window - event) / R:.2f} ms per round without it")
tot = 0.0
for name, v, f in out:
    tot += v
    print(f"  {name:72s} {v:8.2f}  floor {f:6.2f}")
print(f"  {'round (sum of the rows)':72s} {tot:8.2f}  floor {sum(f for _, _, f in out):6.2f}")
