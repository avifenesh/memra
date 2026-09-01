# Blinding: gen rows -> shuffled anonymous text files + a sealed mapping.
# The judge reads ONLY blind/t{T}/{anon}.txt and writes scores; mapping-t{T}.json is
# opened only after all scores for that turn are written (enforced by discipline; the
# mapping file is written here and never printed).
import json, os, random, sys

BANK = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(BANK, "raw", "gen")
BLIND = os.path.join(BANK, "blind")

for T in (4, 8):
    rows = []
    for f in sorted(os.listdir(GEN)):
        if not f.endswith(".json") or ("-t%d-" % T) not in f:
            continue
        r = json.load(open(os.path.join(GEN, f)))
        if r.get("turn") != T:
            continue
        text = (r.get("content") or "").strip() or (r.get("reasoning") or "")
        rows.append(dict(src=f, arm=r["arm"], sample=r["sample"],
                         valid=r.get("valid"), invalid_reason=r.get("invalid_reason"),
                         finish=r.get("finish"), text=text))
    random.seed(os.urandom(16))
    random.shuffle(rows)
    os.makedirs(os.path.join(BLIND, "t%d" % T), exist_ok=True)
    mapping = {}
    for i, r in enumerate(rows):
        anon = "%s%02d" % ("mx", i + 1)
        mapping[anon] = dict(src=r["src"], arm=r["arm"], sample=r["sample"],
                             valid=r["valid"], invalid_reason=r["invalid_reason"])
        # repetition ratio: fraction of the text covered by the most repeated 60-char
        # shingle; arm-neutral, helps the LOOP disqualifier call.
        t = r["text"]
        shingles = {}
        for j in range(0, max(0, len(t) - 60), 30):
            s = t[j:j + 60]
            shingles[s] = shingles.get(s, 0) + 1
        rep = max(shingles.values()) if shingles else 0
        hdr = ("ANON=%s TURN=%d CHARS=%d FINISH=%s REPEAT60=%d\n" + "=" * 70 + "\n") % (
            anon, T, len(t), r["finish"], rep)
        open(os.path.join(BLIND, "t%d" % T, anon + ".txt"), "w").write(hdr + t)
    json.dump(mapping, open(os.path.join(BLIND, "mapping-t%d.json" % T), "w"), indent=1)
    print("t%d: %d blind files written" % (T, len(rows)))
print("SEALED. Judge from blind/t{4,8}/*.txt only; open mapping-*.json after scoring.")
