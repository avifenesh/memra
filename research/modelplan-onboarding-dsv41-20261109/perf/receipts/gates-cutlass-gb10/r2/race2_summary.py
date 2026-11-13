#!/usr/bin/env python3
"""The race campaign's per-arm totals from race2.sh's runs-PART.txt files: runs, compared passes, corrupt passes
(failed out_stream checks of rank 0), and the one-sided 95% upper bound on the per-pass rate (exact binomial;
three over N when none failed).

usage: race2_summary.py RUNS_FILE...
"""
import math
import re
import sys

arms = {}
order = []
for path in sys.argv[1:]:
    for line in open(path):
        m = re.match(r"(\d+-\d+)-(\w+) \(.*?\) rc (\d+) (\d+) checks, (\d+) failed FAIL lines (\d+) passes (\d+) secs (\d+)", line)
        if not m:
            continue
        arm = m.group(2)
        if arm not in arms:
            arms[arm] = {"runs": 0, "passes": 0, "fail_lines": 0, "secs": 0, "bad_runs": 0}
            order.append(arm)
        a = arms[arm]
        a["runs"] += 1
        a["passes"] += int(m.group(7))
        a["fail_lines"] += int(m.group(6))
        a["secs"] += int(m.group(8))
        a["bad_runs"] += int(m.group(6)) > 0


def upper(k, n, conf=0.95):
    """One-sided upper bound of a binomial rate with k events in n trials (Clopper-Pearson)."""
    if n == 0:
        return float("nan")
    if k == 0:
        return 1 - (1 - conf) ** (1 / n)
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        # P(X <= k | p = mid), each term in log space (n is tens of thousands)
        cdf = sum(
            math.exp(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                     + i * math.log(mid) + (n - i) * math.log1p(-mid))
            for i in range(k + 1)
        )
        if cdf > 1 - conf:
            lo = mid
        else:
            hi = mid
    return hi


print("| arm | runs | compared passes | corrupt passes | runs with a corrupt pass | per-pass rate | 95% upper bound |")
print("|---|---|---|---|---|---|---|")
for arm in order:
    a = arms[arm]
    k = a["fail_lines"] // 2  # two failed checks a corrupt pass: one a rank, the ranks always agree
    rate = k / a["passes"] if a["passes"] else float("nan")
    print(f"| {arm} | {a['runs']} | {a['passes']:,} | {k} | {a['bad_runs']} | {rate:.2e} | {upper(k, a['passes']):.2e} |")
