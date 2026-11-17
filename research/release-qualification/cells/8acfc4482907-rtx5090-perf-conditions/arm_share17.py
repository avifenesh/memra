"""Landing 17 heat rule: per-arm share of 1 Hz samples with a thermal or hardware slowdown flag, from arms.tsv
(arm_conditions.py), plus the start temperature. Prints one line and exits 0 when the take counts, 1 when it is void
(an arm over 5%, or a start at 86 C or hotter).
usage: arm_share17.py ARMS_TSV START_TEMP_C"""
import csv
import sys

rows = list(csv.DictReader(open(sys.argv[1]), delimiter="\t"))
start = int(sys.argv[2])
arms = {}
for r in rows:
    arm = "A" if "-A-" in r["run"] else "B"
    s, f = arms.get(arm, (0, 0))
    arms[arm] = (s + int(r["samples"]), f + int(r["thermal_or_hw_slowdown_active"]))
share = {a: (f / s if s else 0.0) for a, (s, f) in arms.items()}
void = start >= 86 or any(v > 0.05 for v in share.values())
print(f"start {start} C; " + "; ".join(f"arm {a}: {arms[a][1]}/{arms[a][0]} = {share[a] * 100:.2f}%" for a in sorted(arms))
      + ("; VOID" if void else "; counts"))
sys.exit(1 if void else 0)
