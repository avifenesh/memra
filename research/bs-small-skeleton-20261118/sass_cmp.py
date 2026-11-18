#!/usr/bin/env python3
"""Per-kernel SASS comparison of two builds of the same TUs (cuobjdump -sass text and -res-usage).

usage: sass_cmp.py [--digests OUT.tsv] BASE_DIR NEW_DIR [UNIT-ARCH ...]
Each DIR holds UNIT-ARCH.sass and UNIT-ARCH.res (nv_build.sh). Kernels are split at `Function : NAME`, NAME is
demangled with c++filt, and then normalized:
  - the internal-linkage prefixes nvcc derives from the build directory (_INTERNAL_<8 hex>_, _GLOBAL__N__<8 hex>_)
    are masked (the cfix-pro6000 kcmp method);
  - the kernels this change moved into the shared skeleton are renamed back to their route names by RENAME
    (template arguments mapped to the old ones), so a moved kernel is compared with its old self.
The SASS body of a kernel (every line after its `Function :` line up to the next) must match byte for byte, and so
must its resource line (registers, stack, shared, constant banks). Prints one line a kernel and a summary a unit;
--digests writes unit, kernel, base and new symbol, the sha256 of each normalized body and the resource lines.
"""
import hashlib
import re
import subprocess
import sys

MASK = re.compile(r"(_INTERNAL_|_GLOBAL__N__)[0-9a-f]{8}_")

# Names are compared without their parameter lists (the moved kernels' parameter types are the skeleton's records,
# the same layouts) and with "> >" written ">>". New name -> old name:
RENAME = [
    (re.compile(r"^bs::bs_small_moe_kernel<bs::Warp6, bs::MxHw32<1>, bs::EpiBf16, (\d+), 4>$"),
     lambda m: f"v41_moe_small_kernel<{m.group(1)}>"),
    (re.compile(r"^bs::bs_small_moe_kernel<bs::Block2, bs::MxHw32<1>, bs::EpiBf16, (\d+), (\d+)>$"),
     lambda m: f"v41_moe_small_blk_kernel<{m.group(1)}, {m.group(2)}>"),
    (re.compile(r"^bs::bs_small_dense_kernel<bs::MxHw32<(\d)>, bs::EpiBf16, (\d+), (\d+)>$"),
     lambda m: f"v41_dense_small_kernel<{m.group(1)}, {m.group(2)}, {m.group(3)}>"),
    (re.compile(r"^bs::bs_moe_count_kernel<256>$"), lambda m: "v41_moe_count_kernel"),
    (re.compile(r"^bs::bs_moe_scan_kernel<1024>$"), lambda m: "v41_moe_scan_kernel"),
    (re.compile(r"^bs::bs_moe_place_kernel<256>$"), lambda m: "v41_moe_place_kernel"),
    (re.compile(r"^bs::bs_moe_plan_small_kernel<256, 1024>$"), lambda m: "v41_moe_plan_small_kernel"),
    # the configuration gained the activations' type as its first argument
    (re.compile(r"^bs::cut::bs_cutlass_setup_kernel<bs::cut::Cfg<cutlass::mx_float8_t<cutlass::float_e4m3_t>, (.*)$"),
     lambda m: f"v41c::setup_kernel<v41c::Cfg<{m.group(1)}"),
]


def norm_name(dem):
    d = MASK.sub(r"\1xxxxxxxx_", dem)
    if d.startswith("void "):
        d = d[5:]
    d = d.split("(", 1)[0].replace("> >", ">>").strip()
    return d


def demangle(names):
    out = subprocess.run(["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True).stdout
    return out.split("\n")[: len(names)]


def key_of(dem):
    d = norm_name(dem)
    for rx, f in RENAME:
        m = rx.match(d)
        if m:
            return f(m)
    return d


def split_sass(path):
    funcs, cur, body = {}, None, []
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*Function : (\S+)", line)
            if m:
                if cur is not None:
                    funcs[cur] = body
                cur, body = m.group(1), []
                continue
            if cur is not None:
                body.append(line.rstrip("\n"))
    if cur is not None:
        funcs[cur] = body
    return funcs


def split_res(path):
    res, cur = {}, None
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*Function (\S+):", line)
            if m:
                cur = m.group(1)
                continue
            if cur and line.strip().startswith("REG:"):
                res[cur] = line.strip()
                cur = None
    return res


def keyed(funcs):
    names = sorted(funcs)
    dem = demangle(names)
    out = {}
    for n, d in zip(names, dem):
        k = key_of(d)
        # trailing code-section padding (NOP / BRA self loops) is part of the body and is compared too
        assert k not in out, f"two kernels normalize to {k}"
        out[k] = n
    return out


def strip_body(body):
    # the body carries no symbol names; drop blank and section-header lines only
    return [MASK.sub(r"\1xxxxxxxx_", l) for l in body if l.strip() and not l.strip().startswith(".")]


def main():
    args = sys.argv[1:]
    dig = None
    if args[:1] == ["--digests"]:
        dig = open(args[1], "w")
        dig.write("unit\tkernel\tbase_symbol\tnew_symbol\tbase_sha256\tnew_sha256\tequal\tbase_res\tnew_res\n")
        args = args[2:]
    base, new = args[0], args[1]
    units = args[2:] or ["dsv41_gpu-120a", "dsv41_gpu-121a", "dsv41_cutlass-120a", "dsv41_cutlass-121a"]
    total_bad = 0
    for u in units:
        fa, fb = split_sass(f"{base}/{u}.sass"), split_sass(f"{new}/{u}.sass")
        ra, rb = split_res(f"{base}/{u}.res"), split_res(f"{new}/{u}.res")
        ka, kb = keyed(fa), keyed(fb)
        same = moved = bad = 0
        lines = []
        for k in sorted(set(ka) | set(kb)):
            if k not in ka or k not in kb:
                lines.append(f"  ONLY-{'BASE' if k in ka else 'NEW '} {k}")
                bad += 1
                continue
            na, nb = ka[k], kb[k]
            ba, bb = strip_body(fa[na]), strip_body(fb[nb])
            res_eq = ra.get(na) == rb.get(nb)
            if dig:
                ha = hashlib.sha256("\n".join(ba).encode()).hexdigest()
                hb = hashlib.sha256("\n".join(bb).encode()).hexdigest()
                dig.write(f"{u}\t{k}\t{na}\t{nb}\t{ha}\t{hb}\t{int(ba == bb and res_eq)}\t{ra.get(na)}\t{rb.get(nb)}\n")
            if ba == bb and res_eq:
                if na == nb:
                    same += 1
                else:
                    moved += 1
                    lines.append(f"  equal (moved) {k}  [{len(ba)} lines; {rb.get(nb)}]")
            else:
                bad += 1
                nd = sum(1 for x, y in zip(ba, bb) if x != y) + abs(len(ba) - len(bb))
                lines.append(f"  DIFFER {k}: {len(ba)} vs {len(bb)} lines, {nd} differ; res {ra.get(na)} | {rb.get(nb)}")
        print(f"{u}: kernels base {len(ka)} new {len(kb)}; equal {same + moved} ({same} same name, {moved} moved), "
              f"differ or unmatched {bad}")
        for l in lines:
            print(l)
        total_bad += bad
    print("SASS-CMP", "EQUAL" if total_bad == 0 else f"DIFFER {total_bad}")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
