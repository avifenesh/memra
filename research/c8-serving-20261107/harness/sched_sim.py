"""Closed-loop prime-pass simulator: 8 users take the 29 agentic turns in order (ThreadPoolExecutor.map).

Service is pure prefill at an aggregate rate R tok/s. Admission caps the number of resident sessions
(cap); the rest wait FIFO. Two disciplines over the admitted set:
  ps   - processor sharing (each admitted prefill advances at R / n per unit time, memra's 1024-per-tick
         round robin in the limit);
  fifo - the oldest admitted prefill takes the whole rate (vLLM-style FCFS chunked prefill).
Reports TTFT p50 / max and how many requests miss a deadline.
  python3 sched_sim.py APPROX.json
"""
import json
import sys

lens = [x[1] for x in json.load(open(sys.argv[1]))]


def sim(rate, cap, disc, users=8, dt=0.01):
    pending = list(range(len(lens)))  # task queue, in submission order
    t = 0.0
    sys_ = []  # [idx, arrival, remaining]
    ttft = {}
    free_users = users
    while pending or sys_:
        while free_users and pending:
            i = pending.pop(0)
            sys_.append([i, t, float(lens[i])])
            free_users -= 1
        admitted = sys_[:cap]
        if disc == 'ps':
            share = rate * dt / len(admitted)
            for s in admitted:
                s[2] -= share
        else:
            budget = rate * dt
            for s in admitted:
                use = min(budget, s[2])
                s[2] -= use
                budget -= use
                if budget <= 0:
                    break
        t += dt
        done = [s for s in sys_ if s[2] <= 0]
        for s in done:
            ttft[s[0]] = t - s[1]
            sys_.remove(s)
            free_users += 1
    v = sorted(ttft.values())
    return v, t


for rate in (700, 900, 1200, 1800):
    for cap in (5, 8):
        for disc in ('ps', 'fifo'):
            v, wall = sim(rate, cap, disc)
            miss = sum(1 for x in v if x > 90)
            print(f"rate={rate} cap={cap} {disc:4s} wall={wall:6.1f}s ttft_p50={v[len(v)//2]:6.1f}s "
                  f"max={v[-1]:6.1f}s miss90={miss}")
