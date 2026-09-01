#!/usr/bin/env python3
# Blind pass for the defect-7 cell. Collects every turn-8-arm row (+ t1 baseline rows,
# scored on DQ classes only), extracts judge text per the rubric's thinking-model rule
# (content if non-empty else reasoning), shuffles under neutral ids, writes
# blind/t8/px??.txt and blind/mapping.json. The mapping is written at shuffle time and
# read only after blind/scores.json is complete (same practice as the sq cell).
import json, glob, os, random, sys

LANE = os.path.dirname(os.path.abspath(__file__))
random.seed(20260829)

rows = []
for f in sorted(glob.glob(LANE + "/gen/*-s*.json")):
    r = json.load(open(f))
    rows.append(r)

t8 = [r for r in rows if r["arm"] != "t1"]
random.shuffle(t8)
os.makedirs(LANE + "/blind/t8", exist_ok=True)
mapping = {}
for i, r in enumerate(t8, 1):
    pid = "px%02d" % i
    mapping[pid] = {"arm": r["arm"], "sample": r["sample"]}
    judge = (r.get("content") or "").strip() or (r.get("reasoning") or "").strip()
    meta = "finish=%s content_nonempty=%s\n---\n" % (r.get("finish"), bool((r.get("content") or "").strip()))
    open("%s/blind/t8/%s.txt" % (LANE, pid), "w").write(meta + judge)
json.dump(mapping, open(LANE + "/blind/mapping.json", "w"), indent=1)
print("blinded %d t8-arm rows -> blind/t8/px01..px%02d; mapping written (do not read before scores)" % (len(t8), len(t8)))
