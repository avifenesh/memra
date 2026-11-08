#!/usr/bin/env python3
"""gaps_api.py prof.sqlite TTFT_S [--min-ms 1]: GPU idle gaps inside the last served prefill (the
window parts_last.py uses: TTFT_S ending at the last prime attention launch), and what the host's
CUDA runtime/driver calls were doing during each gap of at least MIN_MS.

Per gap: its start relative to the window, length, the kernel before and after, and the API calls
that overlap it (name, overlap). Aggregates: total gap time per (kernel before -> kernel after) pair
and API time inside gaps per call name; the rest of a gap is host time outside any CUDA call."""
import argparse, collections, re, sqlite3

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("ttft", type=float)
ap.add_argument("--min-ms", type=float, default=1.0)
ap.add_argument("--show", type=int, default=12)
a = ap.parse_args()
con = sqlite3.connect(a.db)
rows = con.execute("SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                   "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
t1 = max(e for s, e, n in rows if n.startswith("fa_prefill_qw"))
t0 = t1 - int(a.ttft * 1e9)
win = [(s, e, n) for s, e, n in rows if t0 <= s <= t1]
api = []
for tab in ("CUPTI_ACTIVITY_KIND_RUNTIME", "CUPTI_ACTIVITY_KIND_DRIVER"):
    try:
        api += con.execute(f"SELECT r.start, r.end, s.value FROM {tab} r JOIN StringIds s ON s.id = r.nameId "
                           "WHERE r.end >= ? AND r.start <= ?", (t0, t1)).fetchall()
    except sqlite3.OperationalError:
        pass
api.sort()


def short(n):
    return re.sub(r"\(.*", "", re.sub(r"^void ", "", n))[:48]


gaps = []
last_e, last_n = win[0][1], win[0][2]
for s, e, n in win[1:]:
    if s > last_e:
        gaps.append((last_e, s, last_n, n))
    if e > last_e:
        last_e, last_n = e, n
total = sum(g1 - g0 for g0, g1, _, _ in gaps)
big = [g for g in gaps if g[1] - g[0] >= a.min_ms * 1e6]
print(f"window {(t1 - t0) / 1e9:.3f} s; gaps {len(gaps)} totalling {total / 1e9:.3f} s; "
      f">= {a.min_ms} ms: {len(big)} totalling {sum(g1 - g0 for g0, g1, _, _ in big) / 1e9:.3f} s")
pair = collections.Counter(); pairn = collections.Counter(); apis = collections.Counter()
for g0, g1, nb, na in big:
    k = (short(nb), short(na))
    pair[k] += g1 - g0; pairn[k] += 1
    for s, e, n in api:
        if e < g0 or s > g1:
            continue
        apis[n] += min(e, g1) - max(s, g0)
print("\ngap time by (kernel before -> kernel after)")
for k, v in pair.most_common(15):
    print(f"  {v / 1e9:8.3f} s {pairn[k]:5d} x  {k[0]} -> {k[1]}")
print("\nCUDA API time overlapping those gaps")
for k, v in apis.most_common(15):
    print(f"  {v / 1e9:8.3f} s  {k}")
print(f"\nfirst {a.show} large gaps")
for g0, g1, nb, na in big[:a.show]:
    ov = [(n, (min(e, g1) - max(s, g0)) / 1e6) for s, e, n in api if not (e < g0 or s > g1)]
    ovs = ", ".join(f"{n} {d:.2f}" for n, d in ov[:6])
    print(f"  at {(g0 - t0) / 1e9:8.3f} s {(g1 - g0) / 1e6:8.2f} ms  {short(nb)} -> {short(na)}  [{ovs}]")
