"""Compare the v3 GEMM kernels' SASS between two cuobjdump -sass dumps of mmq_nvfp4_w4a8.cu.

Old: mul_mat_f8f4v3<BM,BN,WM,WN,KS,ST,EPI> and mul_mat_f8f4v3t<..,PBM>.
New: one template mul_mat_f8f4v3<..,EPI,PBM>. A shipped kernel maps to PBM == BM, the tail kernel to
the same arguments. Every other function must be identical too.
usage: cmp.py OLD.sass NEW.sass
"""
import re
import sys


def funcs(path):
    out, name, body = {}, None, []
    for line in open(path):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            if name:
                out[name] = body
            name, body = m.group(1), []
            continue
        if name is None:
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s*(.*?)\s*;?\s*(/\*.*\*/)?\s*$", line)
        if m:
            body.append(re.sub(r"\s+", " ", m.group(1)))
    if name:
        out[name] = body
    return out


def key(name):
    m = re.match(r"_ZL?\d+(mul_mat_f8f4v3t?)I((?:Li\d+E)+)E", name)
    if not m:
        return ("other", name)
    args = re.findall(r"Li(\d+)E", m.group(2))
    if m.group(1) == "mul_mat_f8f4v3" and len(args) == 7:
        args = args + [args[0]]
    return ("v3", tuple(args))


old, new = funcs(sys.argv[1]), funcs(sys.argv[2])
ko = {key(n): n for n in old}
kn = {key(n): n for n in new}
bad = 0
for k in sorted(set(ko) | set(kn), key=str):
    if k not in ko or k not in kn:
        print("ONLY-IN", "old" if k in ko else "new", k)
        bad += 1
        continue
    a, b = old[ko[k]], new[kn[k]]
    if a != b:
        n = sum(1 for x, y in zip(a, b) if x != y) + abs(len(a) - len(b))
        print("DIFF", k, len(a), len(b), n)
        bad += 1
    elif k[0] == "v3":
        print("SAME", k, len(a))
print("functions old", len(old), "new", len(new), "mismatches", bad)
