"""cmp.py with the instruction encodings (and so the scheduling control bits) included: every SASS
line of each function with its address comment stripped. usage: cmp_enc.py OLD.sass NEW.sass"""
import re
import sys


def funcs(p):
    out, name = {}, None
    for line in open(p):
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            name = m.group(1)
            out[name] = []
            continue
        if name and "/*" in line:
            out[name].append(re.sub(r"/\*[0-9a-f]{4,}\*/", "", line).strip())
    return out


def key(n):
    m = re.match(r"_ZL?\d+(mul_mat_f8f4v3t?)I((?:Li\d+E)+)E", n)
    if not m:
        return n
    a = re.findall(r"Li(\d+)E", m.group(2))
    if m.group(1) == "mul_mat_f8f4v3" and len(a) == 7:
        a.append(a[0])
    return "v3" + ",".join(a)


o, n = funcs(sys.argv[1]), funcs(sys.argv[2])
ko = {key(k): v for k, v in o.items()}
kn = {key(k): v for k, v in n.items()}
same = sum(1 for k in ko if kn.get(k) == ko[k])
print(f"functions old {len(ko)} new {len(kn)}, identical with encodings {same}")
