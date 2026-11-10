#!/usr/bin/env python3
"""split.py prof.sqlite reqs.jsonl OUTDIR: the sampled-round split (memra#1186).

Reads one Nsight Systems export of a server boot that served the six requests of req_driver.py and
writes, per request, where the decode window's wall time went.

The device timeline of a request is one stream, so every kernel, device copy and memset is one
interval and the idle between two intervals is host time. A state machine walks the intervals in
start order and gives each one, and each idle gap before it, to a phase:

  solo-init    a T=1 trunk pass that feeds a burst's boundary token (the init feed)
  after-init   from that pass's output head to the first draft step: the 1 MB logits readback, the
               host argmax, the 1 MB upload of the retained row, the burst's setup copies
  draft-fwd    a draft step from its embed to its output head (the MTP block and the head)
  draft-draw   from the draft head to the next draft step or the verify pass: the draw (argmax, or
               the counter bump and Gumbel perturb, or filter stats and filtered perturb, or the
               eager penalised chain's kernels and readbacks), the 4-byte token readback
  verify       the T=K+1 trunk pass, embed to output head
  verify-draw  from the verify head to the accept readback: the row copies, filter stats, Gumbel
               perturb and argmax per row (greedy: the argmax per row), the K+1-token readback
  commit       from the accept readback to the next draft step or solo pass: the partial-accept
               restore, the draft KV fill, the next round's snapshot, and, between two bursts, the
               burst end and the next burst's entry
  solo-tail    a T=1 trunk pass after a round: the sampled tail's commit of the pending bonus (and,
               for any class, the flush of the pending token when the request ends)
  after-tail   from that pass's output head to the next init feed: the 1 MB logits readback, the
               boundary token draw, the draft KV fill, the burst end and the next burst's entry

Markers: an embed launch (`embed_gather_u32`, `embed_gather_u32_t`) opens a draft step when the
next marker is `qmatvec_q5_K_mmvq_il` (the full-vocabulary head of a draft step or of a T=1 pass),
the verify pass when it is `qmatvec_q5_K_mmvq_b4_r2` (the verify head), and a draft KV fill
otherwise (it stays in the phase it falls in); a T=1 pass is the run that holds `gdn_scan_s128_b`,
opened by the host upload of its embedding row (one HtoD of n_embd floats) and closed by its head.
The decode window opens at the cold init feed (the first token is out before it) and closes at the
request's last interval. A round is one verify head launch.

Outputs in OUTDIR: summary.txt (every table), phases.tsv, delta.tsv, kernels.tsv, boundary.tsv, hostcalls.tsv.
"""
import collections
import json
import pathlib
import sqlite3
import sys

db, reqs_path, outdir = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
outdir.mkdir(parents=True, exist_ok=True)
EMBD_BYTES = 5120 * 4  # one embedding row of this model
VH = "qmatvec_q5_K_mmvq_b4_r2"
IL = "qmatvec_q5_K_mmvq_il"
EMBED = ("embed_gather_u32", "embed_gather_u32_t")
PHASES = ["solo-init", "after-init", "draft-fwd", "draft-draw", "verify", "verify-draw", "commit",
          "solo-tail", "after-tail"]
KIND = {1: "HtoD", 2: "DtoH", 8: "DtoD"}

con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
K = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s "
                "ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
K = [(s, e, n.split("(")[0].split("<")[0].replace("void ", "")) for s, e, n in K]
C = con.execute("SELECT start, end, bytes, copyKind FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
M = con.execute("SELECT start, end, bytes FROM CUPTI_ACTIVITY_KIND_MEMSET ORDER BY start").fetchall()
RT = con.execute("SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s "
                 "ON s.id = r.nameId WHERE s.value LIKE 'cuMemcpyDtoH%' OR s.value LIKE 'cuStreamSynchronize%' "
                 "OR s.value LIKE 'cuGraphLaunch%' ORDER BY r.start").fetchall()

# requests: device work separated by more than 1 s of idle; the last six are the driver's
segs = []
a = 0
prev = K[0][1]
for i in range(1, len(K)):
    if K[i][0] - prev > 1e9:
        segs.append((K[a][0], prev))
        a = i
    prev = max(prev, K[i][1])
segs.append((K[a][0], prev))
reqs = [json.loads(l) for l in open(reqs_path) if l.strip()]
assert len(segs) >= len(reqs), (len(segs), len(reqs))
segs = segs[-len(reqs):]


def op_class(kind, name):
    if kind != "K":
        return name
    if name.startswith("qmatvec_nvfp4"):
        return "matvec NVFP4"
    if name.startswith("qmatvec_q5_K"):
        return "output head Q5_K"
    if name.startswith("fa_"):
        return "attention"
    if name.startswith("gdn_") or name.startswith("ssm_conv"):
        return "GDN scan and conv"
    if name == "copy_batch_uniform_f32":
        return "copy_batch_uniform_f32"
    if name.startswith(("filter_stats", "gumbel_", "memra_sctr", "penalize_", "softmax_", "argmax_", "max_prob", "prob_of")):
        return name
    return "norms, quantize, glue"


SAMPLING = ("filter_stats", "gumbel_", "memra_sctr", "penalize_", "softmax_", "argmax_", "max_prob", "prob_of")
results = []
for req, (t_a, t_b) in zip(reqs, segs):
    ops = [(s, e, "K", n, 0) for s, e, n in K if t_a <= s <= t_b]
    ops += [(s, e, "C", KIND.get(k, str(k)), by) for s, e, by, k in C if t_a <= s <= t_b]
    ops += [(s, e, "M", "memset", by) for s, e, by in M if t_a <= s <= t_b]
    ops.sort()
    # T=1 passes: each run of gdn_scan_s128_b, opened by the nearest earlier embedding-row upload
    solo_start = set()
    in_pass = False
    for i, (s, e, kind, n, by) in enumerate(ops):
        if n == "gdn_scan_s128_b" and not in_pass:
            j = i
            while j > 0 and not (ops[j][2] == "C" and ops[j][3] == "HtoD" and ops[j][4] == EMBD_BYTES):
                j -= 1
                assert i - j < 80, "no embedding upload before a T=1 pass"
            solo_start.add(j)
            in_pass = True
        elif n == IL:
            in_pass = False
    state = "prime"
    busy = collections.defaultdict(float)      # (phase, class) -> ns
    count = collections.Counter()              # (phase, class) -> launches
    gaps = collections.defaultdict(float)      # phase -> ns
    gap_hist = collections.defaultdict(lambda: collections.defaultdict(float))
    kern = collections.defaultdict(lambda: [0, 0.0])   # (phase, kernel name) -> [n, ns]
    units = collections.defaultdict(list)      # phase -> list of [wall ns] per occurrence
    cur_end = None
    t_dec = None
    rounds = 0
    marks = []                                 # (time, new state) transitions, for the API split
    for i, (s, e, kind, n, by) in enumerate(ops):
        old = state
        if i in solo_start:
            state = "solo-init" if old in ("prime", "after-tail", "after-init") else "solo-tail"
            if t_dec is None:
                t_dec = s
                cur_end = s
        elif kind == "K" and n in EMBED and old != "prime" and not old.startswith("solo-"):
            # what this embed opens: the next marker says (a draft KV fill has none of its own)
            j = i + 1
            while j < len(ops) and not (ops[j][2] == "K" and ops[j][3] in (IL, VH) + EMBED) and j not in solo_start:
                j += 1
            nxt = ops[j][3] if j < len(ops) and j not in solo_start else ""
            if nxt == IL:
                state = "draft-fwd"
            elif nxt == VH:
                state = "verify"
        if t_dec is None:
            continue
        gap = max(0, s - cur_end)
        if gap:
            gaps[old] += gap
            b = "under 5 us" if gap < 5e3 else "5 to 100 us" if gap < 1e5 else "100 us to 1 ms" if gap < 1e6 else "1 ms and over"
            gap_hist[old][b] += gap
        if state != old:
            units[state].append(0.0)
            marks.append((s, state))
        if old != state and units[old]:
            pass
        d = max(0, e - max(s, cur_end))
        cls = op_class(kind, n)
        busy[(state, cls)] += d
        count[(state, cls)] += 1
        if kind == "K":
            x = kern[(state, n)]
            x[0] += 1
            x[1] += e - s
        cur_end = max(cur_end, e)
        # transitions that take effect after this interval
        if kind == "K" and n == IL and state in ("solo-init", "solo-tail"):
            state = "after-init" if state == "solo-init" else "after-tail"
        elif kind == "K" and n == IL and state == "draft-fwd":
            state = "draft-draw"
        elif kind == "K" and n == VH and state == "verify":
            state = "verify-draw"
            rounds += 1
        elif kind == "C" and n == "DtoH" and state == "verify-draw":
            state = "commit"
        if state != old and (not marks or marks[-1][1] != state):
            marks.append((e, state))
    t_end = cur_end
    window = t_end - t_dec
    phase_busy = {p: sum(v for (ph, c), v in busy.items() if ph == p) for p in PHASES}
    phase_ms = {p: (phase_busy[p] + gaps.get(p, 0.0)) / 1e6 for p in PHASES}
    assert abs(sum(phase_ms.values()) - window / 1e6) < 1.0, (sum(phase_ms.values()), window / 1e6)
    # passes of each kind, from the transition marks
    n_units = collections.Counter(st for _, st in marks)
    # host API waits by the phase the call starts in
    api = collections.defaultdict(lambda: [0, 0.0])
    mi = 0
    cur = "prime"
    for s, e, name in RT:
        if s < t_dec or s > t_end:
            continue
        while mi < len(marks) and marks[mi][0] <= s:
            cur = marks[mi][1]
            mi += 1
        x = api[(cur, name.split("_v2")[0])]
        x[0] += 1
        x[1] += e - s
    results.append(dict(req=req, rounds=rounds, window_ms=window / 1e6, phase_ms=phase_ms, busy=busy, count=count,
                        gaps=gaps, gap_hist=gap_hist, kern=kern, n_units=n_units, api=api,
                        solo=len(solo_start)))

out = []


def P(line=""):
    out.append(line)


P("The sampled-round split: one request per class, the decode window of each under Nsight Systems")
P()
P("class      prompt  tokens  rounds  drafted  accepted  init feeds  tail passes  window ms  ms/round  client ms/round  text sha")
for r in results:
    q = r["req"]
    dec = (q["total_s"] - q["ttft_s"]) * 1e3
    cut = "" if r["rounds"] == q["rounds"] else f"  (the trace holds {r['rounds']} of its {q['rounds']} rounds: the profiler lost the boot's last events)"
    P(f"{q['class']:9s} {q['prompt_tokens']:7d} {q['completion_tokens']:7d} {r['rounds']:7d} {q['drafted']:8d} {q['accepted']:9d} "
      f"{r['n_units'].get('solo-init', 0):11d} {r['n_units'].get('solo-tail', 0):12d} {r['window_ms']:10.1f} {r['window_ms'] / r['rounds']:9.2f} "
      f"{dec / q['rounds']:16.2f}  {q['sha']}{cut}")
P()
P("init feeds, tail passes: the T=1 trunk passes in the window. A sampled request has one init feed per burst and one tail")
P("pass per burst that ends on a pending token; a greedy request has the cold init feed and one flush when it ends.")
P("window: from the cold init feed to the request's last device work. client ms/round: (total less TTFT) / rounds on the")
P("client's clock, which opens after the cold init feed and, for the greedy requests, closes before the closing flush.")
base = next(r for r in results if r["req"]["class"] == "greedy")
P()
P("== Phases, ms per round (phase wall / rounds); in brackets the change against the greedy request of the same boot")
hdr = f"{'phase':12s}" + "".join(f"{r['req']['class']:>22s}" for r in results if r["req"]["class"] != "warm")
P(hdr)
rows_tsv = ["class\tphase\tms_total\tms_per_round\tbusy_ms_per_round\tgap_ms_per_round\tdelta_vs_greedy_ms_per_round"]
for p in PHASES:
    line = f"{p:12s}"
    for r in results:
        if r["req"]["class"] == "warm":
            continue
        v = r["phase_ms"][p] / r["rounds"]
        d = v - base["phase_ms"][p] / base["rounds"]
        line += f"{v:12.3f} [{d:+7.3f}]"
        bz = sum(x for (ph, c), x in r["busy"].items() if ph == p) / 1e6 / r["rounds"]
        rows_tsv.append(f"{r['req']['class']}\t{p}\t{r['phase_ms'][p]:.3f}\t{v:.4f}\t{bz:.4f}\t{r['gaps'].get(p, 0) / 1e6 / r['rounds']:.4f}\t{d:+.4f}")
    P(line)
line = f"{'round':12s}"
for r in results:
    if r["req"]["class"] == "warm":
        continue
    v = r["window_ms"] / r["rounds"]
    line += f"{v:12.3f} [{v - base['window_ms'] / base['rounds']:+7.3f}]"
P(line)
(outdir / "phases.tsv").write_text("\n".join(rows_tsv) + "\n")
P()
P("== Sampled less greedy, ms per round of wall, by component (the brackets above, grouped)")
GROUPS = [("burst boundary: tail T=1 pass", ["solo-tail"]),
          ("burst boundary: after it (logits readback, boundary draw, fill, burst exit and entry)", ["after-tail"]),
          ("burst boundary: init-feed T=1 pass", ["solo-init"]),
          ("burst boundary: after it (logits readback, host argmax, row upload, burst setup, snapshot)", ["after-init"]),
          ("SUM burst boundary", ["solo-tail", "after-tail", "solo-init", "after-init"]),
          ("in the round: draft forward (MTP block and head)", ["draft-fwd"]),
          ("in the round: draft draw and token readback", ["draft-draw"]),
          ("in the round: verify pass", ["verify"]),
          ("in the round: verify draw and accept readback", ["verify-draw"]),
          ("in the round: commit, fill, snapshot", ["commit"]),
          ("SUM in the round", ["draft-fwd", "draft-draw", "verify", "verify-draw", "commit"])]
others = [r for r in results if r["req"]["class"] not in ("warm", "greedy")]
P(f"{'component':92s}" + "".join(f"{r['req']['class']:>10s}" for r in others))
d_tsv = ["component\t" + "\t".join(r["req"]["class"] for r in others)]
for label, ps in GROUPS:
    vals = [sum(r["phase_ms"][p] for p in ps) / r["rounds"] - sum(base["phase_ms"][p] for p in ps) / base["rounds"] for r in others]
    P(f"{label:92s}" + "".join(f"{v:+10.3f}" for v in vals))
    d_tsv.append(label + "\t" + "\t".join(f"{v:+.4f}" for v in vals))
fb = (base["req"]["rounds"] + base["req"]["accepted"]) / base["req"]["completion_tokens"]
wall = [r["window_ms"] / r["rounds"] - base["window_ms"] / base["rounds"] for r in others]
cred = [-(1 - (r["req"]["rounds"] + r["req"]["accepted"]) / r["req"]["completion_tokens"]) * r["window_ms"] / r["rounds"]
        + (1 - fb) * base["window_ms"] / base["rounds"] for r in others]
P(f"{'round wall, sampled less greedy':92s}" + "".join(f"{v:+10.3f}" for v in wall))
P(f"{'tokens the boundary emits outside the rounds (the issue figure counts round tokens only)':92s}" + "".join(f"{v:+10.3f}" for v in cred))
P(f"{'the issue figure, sampled less greedy':92s}" + "".join(f"{a + b:+10.3f}" for a, b in zip(wall, cred)))
d_tsv.append("round wall\t" + "\t".join(f"{v:+.4f}" for v in wall))
d_tsv.append("boundary tokens outside the rounds\t" + "\t".join(f"{v:+.4f}" for v in cred))
d_tsv.append("issue figure\t" + "\t".join(f"{a + b:+.4f}" for a, b in zip(wall, cred)))
(outdir / "delta.tsv").write_text("\n".join(d_tsv) + "\n")
P()
P("The issue figure is 1000 x tokens per round / decode rate with tokens per round = 1 + K x acceptance, that is wall per")
P("round x (rounds + accepted) / completion tokens. A sampled burst emits one boundary token outside its rounds, so the")
P("figure is the wall per round less that token's share:")
P(f"{'class':9s} {'wall ms/round':>14s} {'round tokens / all tokens':>26s} {'issue figure':>13s}")
for r in results:
    if r["req"]["class"] == "warm":
        continue
    q = r["req"]
    f = (q["rounds"] + q["accepted"]) / q["completion_tokens"]
    P(f"{q['class']:9s} {r['window_ms'] / r['rounds']:14.2f} {f:26.4f} {r['window_ms'] / r['rounds'] * f:13.2f}")
P()
P("== Burst boundary units: count, ms each (wall of the phase / count)")
P(f"{'class':9s} {'unit':12s} {'count':>6s} {'ms each':>9s} {'ms total':>10s}")
b_tsv = ["class\tunit\tcount\tms_each\tms_total"]
for r in results:
    if r["req"]["class"] == "warm":
        continue
    for p in ("solo-init", "after-init", "solo-tail", "after-tail"):
        n = r["n_units"].get(p, 0)
        if n:
            P(f"{r['req']['class']:9s} {p:12s} {n:6d} {r['phase_ms'][p] / n:9.3f} {r['phase_ms'][p]:10.1f}")
            b_tsv.append(f"{r['req']['class']}\t{p}\t{n}\t{r['phase_ms'][p] / n:.3f}\t{r['phase_ms'][p]:.1f}")
(outdir / "boundary.tsv").write_text("\n".join(b_tsv) + "\n")
P()
P("== Inside each phase: device time by class and idle (host), ms per round")
k_tsv = ["class\tphase\top\tcount\tcount_per_round\tus_each\tms_per_round"]
for r in results:
    if r["req"]["class"] == "warm":
        continue
    P(f"-- {r['req']['class']} ({r['rounds']} rounds)")
    for p in PHASES:
        items = sorted(((c, v) for (ph, c), v in r["busy"].items() if ph == p), key=lambda kv: -kv[1])
        if not items and not r["gaps"].get(p):
            continue
        P(f"  {p}: {r['phase_ms'][p] / r['rounds']:.3f} ms per round")
        for c, v in items:
            n = r["count"][(p, c)]
            P(f"      {c:36s} {v / 1e6 / r['rounds']:9.4f} ms/round  {n / r['rounds']:8.2f} per round  {v / 1e3 / n:9.1f} us each")
            k_tsv.append(f"{r['req']['class']}\t{p}\t{c}\t{n}\t{n / r['rounds']:.3f}\t{v / 1e3 / n:.2f}\t{v / 1e6 / r['rounds']:.5f}")
        g = r["gaps"].get(p, 0.0)
        P(f"      {'idle between intervals (host)':36s} {g / 1e6 / r['rounds']:9.4f} ms/round   "
          + ", ".join(f"{b} {v / 1e6 / r['rounds']:.4f}" for b, v in sorted(r["gap_hist"][p].items())))
        k_tsv.append(f"{r['req']['class']}\t{p}\tidle (host)\t\t\t\t{g / 1e6 / r['rounds']:.5f}")
(outdir / "kernels.tsv").write_text("\n".join(k_tsv) + "\n")
P()
P("== Sampling kernels by phase: launches per round, us each, ms per round")
for r in results:
    if r["req"]["class"] == "warm":
        continue
    P(f"-- {r['req']['class']}")
    for (p, n), (c, ns) in sorted(r["kern"].items(), key=lambda kv: (PHASES.index(kv[0][0]), kv[0][1])):
        if n.startswith(SAMPLING):
            P(f"  {p:12s} {n:36s} {c / r['rounds']:7.2f} per round {ns / 1e3 / c:8.1f} us each {ns / 1e6 / r['rounds']:8.4f} ms/round")
P()
P("== Host calls that block, by the phase the call starts in: count per round, ms per round inside the call")
P("   (the host runs ahead of the device, so a readback's wait is device time already counted above, not extra wall)")
g_tsv = ["class\tphase\tcall\tcount\tms_total"]
for r in results:
    if r["req"]["class"] == "warm":
        continue
    P(f"-- {r['req']['class']}")
    for (p, name), (c, ns) in sorted(r["api"].items(), key=lambda kv: (PHASES.index(kv[0][0]) if kv[0][0] in PHASES else 99, kv[0][1])):
        P(f"  {p:12s} {name:28s} {c / r['rounds']:7.2f} per round {ns / 1e6 / r['rounds']:9.3f} ms/round")
        g_tsv.append(f"{r['req']['class']}\t{p}\t{name}\t{c}\t{ns / 1e6:.2f}")
(outdir / "hostcalls.tsv").write_text("\n".join(g_tsv) + "\n")
(outdir / "summary.txt").write_text("\n".join(out) + "\n")
print("\n".join(out))
