#!/bin/bash
# Two builds of dsv41-spec on one card, A B B A, one short lease a run: plan abt9 on the 9-layer resident trunk with
# the draft head (batch-time distinct8 with the launch tally, then the 12 draft tapes: the untraced-draft gate and
# the draft's time and launches). Prints each run's batch-time rows and draft time, then both builds side by side.
# usage: ab_spec.sh TAG_A TAG_B CARD          binaries spec/bin/TAG_A/dsv41-spec and spec/bin/TAG_B/dsv41-spec
set -u
A=$1; B=$2; CARD=$3
S=$LANE/spec
grep -h "^batch-time distinct8 " $S/plans/gate9b.plan > $S/plans/abt9.plan
cat $S/plans/dtape.plan >> $S/plans/abt9.plan
i=0
for arm in $A $B $B $A; do
  i=$((i + 1))
  out=abt-$A-$B/r$i-$arm
  BIN=$S/bin/$arm/dsv41-spec bash $S/scripts/lease.sh abt-r$i 300 $CARD bash $S/scripts/spec_run.sh abt9 $out \
    --layers 9 --resident > $S/logs/abt-$A-$B-r$i-$arm.log 2>&1
  echo "$(date +%H:%M:%S) run $i $arm rc $? $(grep -h SPEC-DONE $S/logs/abt-$A-$B-r$i-$arm.log | tail -n 1)"
done
python3 - $S/out/abt-$A-$B <<'PY'
import glob, json, statistics as st, sys
arms = {}
for d in sorted(glob.glob(sys.argv[1] + "/r*")):
    arm = d.rsplit("-", 1)[-1]
    rows = [json.loads(l) for l in open(d + "/stdout.jsonl") if l.startswith("{")]
    bt = [r for r in rows if r.get("job") == "batch-time"][0]
    dt = [r for r in rows if r.get("job") == "draft-tape"]
    a = arms.setdefault(arm, {"argmax": [], "fwd": [], "draft": [], "ok": [], "kern": []})
    a["argmax"].append([bt[f"argmax_{w}_ms"] for w in (1, 2, 4, 8)])
    a["fwd"].append([bt[f"fwd_{w}_ms"] for w in (1, 2, 4, 8)])
    a["draft"].append(st.median(r["timing"]["draft_ms_median"] for r in dt))
    a["ok"].append((sum(r["pass"] for r in dt), sum(r["untraced_drafts_equal"] for r in dt), sum(r["steps"] for r in dt),
                    sum(r["draft_tokens_equal_vendor"] for r in dt)))
    a["kern"].append((bt["launch_ms_8_two_steps"]["per_kernel"], dt[0]["timing"]["draft_launch_ms_two"]))
    print(d.rsplit("/", 1)[-1], "argmax", [round(x, 2) for x in a["argmax"][-1]], "fwd", [round(x, 2) for x in a["fwd"][-1]],
          "draft ms", round(a["draft"][-1], 2), "tapes pass/untraced equal/steps/vendor tokens", a["ok"][-1])
for arm, a in arms.items():
    med = lambda rows: [round(st.median(c), 2) for c in zip(*rows)]
    print(f"{arm}: argmax 1/2/4/8 {med(a['argmax'])} fwd {med(a['fwd'])} draft {round(st.median(a['draft']), 2)} ms")
    k8, kd = a["kern"][-1]
    top = sorted(k8.items(), key=lambda x: -x[1][1])[:6]
    print("   8-seq call, own ms a step:", ", ".join(f"{n} {m / 2:.2f}" for n, (c, m) in top))
    print("   draft, launches and own ms a draft:", kd["total"] // 2, ", ".join(
        f"{n} {m / 2:.2f}" for n, (c, m) in sorted(kd["per_kernel"].items(), key=lambda x: -x[1][1])[:6]),
        "| by part (stages, head, picks):", [round(m / 2, 2) for c, m in kd["per_layer"]])
PY
echo "AB-SPEC-DONE $A $B"
