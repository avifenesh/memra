"""Per MoE layer, the largest distance of one position's router selection scores between two tapes'
prefill calls (diagnosis of where a perturbation grows).

usage: python pos_spread.py REFERENCE_DIR CANDIDATE_DIR POSITION
"""

import json, sys
import numpy as np
pos = int(sys.argv[3])
def load(d):
    t = json.load(open(f"{d}/tape.json"))
    return t, np.fromfile(f"{d}/{t['router']['score_file']}", dtype="<f4")
rt, rs = load(sys.argv[1]); ct, cs = load(sys.argv[2])
cr = {(r["call"], r["layer"]): r for r in ct["router"]["records"]}
out = []
for r in rt["router"]["records"]:
    if r["call"] != 0: continue
    c = cr[(0, r["layer"])]; e = r["experts"]
    a = rs[r["score_offset"]+pos*e:r["score_offset"]+(pos+1)*e]; b = cs[c["score_offset"]+pos*e:c["score_offset"]+(pos+1)*e]
    out.append("%d:%.1e" % (r["layer"], np.abs(a.astype(float)-b).max()))
print(" ".join(out))
