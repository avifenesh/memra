#!/usr/bin/env python3
"""Compare BEFORE vs AFTER battery outputs byte for byte. usage: compare.py BEFORE_DIR AFTER_DIR"""
import hashlib, json, os, re, sys

B, A = sys.argv[1], sys.argv[2]
ok = True


def runspec_digest(path):
    """Deterministic content of a run-spec log: gold tokens, acceptance counts, verdicts, sampled
    streams. Timing lines (tok/s, prime seconds) are dropped."""
    keep = []
    for line in open(path, errors="replace"):
        s = line.rstrip("\n")
        if re.match(r"^\s*tokens: \[", s) or "acceptance:" in s or "sampled tokens:" in s \
                or "SELF-CONSISTENCY" in s or "FIRST DIVERGENCE" in s:
            keep.append(s)
        elif s.startswith("[generate_spec K="):
            keep.append(re.sub(r"\].*$", "]", s))
    return keep


rows = []
for cell in ("greedy", "sampled", "adapt", "replay"):
    db = runspec_digest(os.path.join(B, "runspec", f"{cell}.log"))
    da = runspec_digest(os.path.join(A, "runspec", f"{cell}.log"))
    h = lambda d: hashlib.sha256("\n".join(d).encode()).hexdigest()[:16]
    same = db == da and len(db) > 0
    passes = sum("PASS" in l and "acceptance" in l for l in da)
    rows.append((f"run-spec/{cell}", h(db), h(da), same, f"{passes} K PASS"))
    ok &= same

sb = json.load(open(os.path.join(B, "serve", "summary.json")))
sa = json.load(open(os.path.join(A, "serve", "summary.json")))
for key in sorted(k for k in sb if k != "bin"):
    vb, va = sb[key], sa.get(key)
    if isinstance(vb, dict):
        same = va is not None and vb["sha256"] == va["sha256"] and vb["finish"] == va["finish"] \
            and vb["completion_tokens"] == va["completion_tokens"]
        rows.append((f"serve/{key}", vb["sha256"][:16], (va or {}).get("sha256", "-")[:16], same,
                     f"{vb['completion_tokens']} tok {vb['finish']}"))
    else:
        same = vb == va
        rows.append((f"serve/{key}", str(vb)[:40], str(va)[:40], same, ""))
    ok &= same


def acc_lines(path):
    return [re.sub(r"^.*?\[spec-acc\]", "[spec-acc]", l.rstrip()) for l in open(path) if "[spec-acc]" in l]


for arm in ("main", "demote_a", "demote_b"):
    lb = acc_lines(os.path.join(B, "serve", f"{arm}-server.log"))
    la = acc_lines(os.path.join(A, "serve", f"{arm}-server.log"))
    same = lb == la
    rows.append((f"serve/{arm} [spec-acc] lines", str(len(lb)), str(len(la)), same, ""))
    ok &= same

for r in rows:
    print(f"{'SAME' if r[3] else 'DIFF'}  {r[0]:<40} {r[1]:<18} {r[2]:<18} {r[4]}")
print("VERDICT:", "BYTE-IDENTICAL" if ok else "MISMATCH")
sys.exit(0 if ok else 1)
