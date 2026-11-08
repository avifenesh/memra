"""Tick-level closed-loop prime-pass simulator (the agentic c8 prime pass: 8 users, 29 turns in order).

A tick primes a set of chunks; its duration is (tokens / rate) + overhead. Admission caps resident
sessions (cap); the rest wait FIFO. Disciplines over the admitted prefilling sessions:
  rr      - memra main: every admitted session advances one chunk (1024) per tick;
  oldest  - only the oldest admitted session advances (chunk 1024) per tick;
  oldest+ - the oldest advances, and so does every other session whose remaining prompt fits in one
            chunk (it finishes this tick), up to a tick token cap.
  python3 tick_sim.py APPROX.json
"""
import json
import sys

lens = [x[1] for x in json.load(open(sys.argv[1]))]
CHUNK = 1024


def sim(rate, cap, disc, users=8, overhead=0.02, tick_cap=4096):
    pending = list(range(len(lens)))
    t = 0.0
    sys_ = []  # [idx, arrival, remaining]
    ttft = {}
    free = users
    while pending or sys_:
        while free and pending:
            i = pending.pop(0)
            sys_.append([i, t, lens[i]])
            free -= 1
        adm = sys_[:cap]
        if disc == 'rr':
            take = [(s, min(CHUNK, s[2])) for s in adm]
        elif disc == 'oldest':
            take = [(adm[0], min(CHUNK, adm[0][2]))]
        else:
            take = [(adm[0], min(CHUNK, adm[0][2]))]
            used = take[0][1]
            for s in adm[1:]:
                if s[2] <= CHUNK and used + s[2] <= tick_cap:
                    take.append((s, s[2]))
                    used += s[2]
        toks = sum(n for _, n in take)
        t += toks / rate + overhead
        for s, n in take:
            s[2] -= n
        for s in [s for s in sys_ if s[2] <= 0]:
            ttft[s[0]] = t - s[1]
            sys_.remove(s)
            free += 1
    v = sorted(ttft.values())
    return v, t


for rate in (800, 1000, 1300):
    for cap in (5, 6, 8):
        for disc in ('rr', 'oldest', 'oldest+'):
            v, wall = sim(rate, cap, disc)
            print(f"rate={rate} cap={cap} {disc:8s} wall={wall:6.1f}s p50={v[len(v)//2]:6.1f}s "
                  f"mean={sum(v)/len(v):6.1f}s max={v[-1]:6.1f}s miss90={sum(1 for x in v if x > 90)}")
