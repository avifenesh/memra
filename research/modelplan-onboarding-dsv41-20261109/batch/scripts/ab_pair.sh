#!/bin/bash
# Two builds of dsv41-spec on a pair of cards (a rank a card), A B B A on each device transport, one short two-card
# lease a run through the 4-card lane's wrapper: batch-time distinct8 on the 9-layer resident trunk with the launch
# tally and every kernel's own time summed over the two ranks.
# usage: ab_pair.sh TAG_A TAG_B PAIR        e.g. TP4_CPUS=28-31,60-63 ab_pair.sh k16 k19 01
set -u
drops() { journalctl -k --no-pager 2>/dev/null | grep drop_caches | tail -n 2 | awk '{print $3" "$(NF-1)" "$NF}' | tr '\n' ';'; }
echo "$(date +%H:%M:%S) page-cache drops before: $(drops)"
A=$1; B=$2; P=$3
S=$LANE/spec
TL=$LANE/bin/tp4_lease.sh
grep -h "^batch-time distinct8 " $S/plans/gate9b.plan > $S/plans/abp9.plan
for tr in peer-pull device-push; do
  t=${tr%%-*}
  i=0
  for arm in $A $B $B $A; do
    i=$((i + 1))
    out=abp-$A-$B/$t-r$i-$arm
    MEMRA_TP_TRANSPORT=$tr BIN=$S/bin/$arm/dsv41-spec bash $TL spec-abp-$t$i 300 $P bash $S/scripts/spec_run.sh abp9 $out \
      --layers 9 --resident --no-draft --devices 0,1 > $S/logs/abp-$A-$B-$t-r$i-$arm.log 2>&1
    echo "$(date +%H:%M:%S) $tr run $i $arm rc $? $(grep -h SPEC-DONE $S/logs/abp-$A-$B-$t-r$i-$arm.log | tail -n 1)"
  done
done
python3 - $S/out/abp-$A-$B $A $B <<'PY'
import glob, json, statistics as st, sys
root, A, B = sys.argv[1:4]
for t in ("peer", "device"):
    arms = {}
    for d in sorted(glob.glob(f"{root}/{t}-r*")):
        arm = d.rsplit("-", 1)[-1]
        bt = [json.loads(l) for l in open(d + "/stdout.jsonl") if l.startswith("{")][0]
        a = arms.setdefault(arm, {"argmax": [], "fwd": [], "k1": [], "k8": []})
        a["argmax"].append([bt[f"argmax_{w}_ms"] for w in (1, 2, 4, 8)])
        a["fwd"].append([bt[f"fwd_{w}_ms"] for w in (1, 2, 4, 8)])
        a["k1"].append(bt.get("launch_ms_1_two_steps", {}).get("per_kernel", {}))
        a["k8"].append(bt.get("launch_ms_8_two_steps", {}).get("per_kernel", {}))
        print(d.rsplit("/", 1)[-1], "argmax", [round(x, 2) for x in a["argmax"][-1]], "fwd", [round(x, 2) for x in a["fwd"][-1]])
    med = lambda rows: [round(st.median(c), 2) for c in zip(*rows)]
    for arm in (A, B):
        if arm in arms:
            print(f"{t} {arm}: argmax 1/2/4/8 {med(arms[arm]['argmax'])} fwd {med(arms[arm]['fwd'])}")
    if A in arms and B in arms:
        for w, key in ((1, "k1"), (8, "k8")):
            ka, kb = arms[A][key], arms[B][key]
            names = sorted(set().union(*ka, *kb), key=lambda n: -max(st.mean(x.get(n, [0, 0])[1] for x in kb), st.mean(x.get(n, [0, 0])[1] for x in ka)))
            print(f"{t}, {w} sequence(s): kernel, launches a step over both ranks, own ms a step summed over both ranks, {A} -> {B}")
            for n in names[:16]:
                ca = st.mean(x.get(n, [0, 0])[0] for x in ka) / 2; ma = st.mean(x.get(n, [0, 0])[1] for x in ka) / 2
                cb = st.mean(x.get(n, [0, 0])[0] for x in kb) / 2; mb = st.mean(x.get(n, [0, 0])[1] for x in kb) / 2
                print(f"   {n:28s} {ca:6.0f} {ma:6.2f} -> {cb:6.0f} {mb:6.2f}  ({mb - ma:+.2f})")
PY
echo "AB-PAIR-DONE $A $B"
echo "$(date +%H:%M:%S) page-cache drops after: $(drops)"
