#!/usr/bin/env python3
"""Refuse a chain-launched kernel whose body does not open with MEMRA_PDL_CHAIN_ENTRY().

`memra_chain_launch` (crates/memra-engine/cu/memra_pdl_chain.cuh) may give a kernel the
programmatic stream serialization attribute, which lets its blocks start while the previous kernel
on the stream still runs. That is correct only when the kernel waits for its predecessor before its
first load or store, so the entry must be the body's first statement. The one exception is a
prologue that reads only checkpoint constants: the body opens with MEMRA_PDL_PRE_WAIT(p, ...),
naming const pointer parameters, and until the entry it may touch no other pointer parameter.
Exit 1 lists every kernel that breaks this, and every launched kernel whose definition was not
found.

Usage: tools/check-pdl-chain.py [cu-dir]   (default crates/memra-engine/cu)
"""
import pathlib, re, sys

root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "crates/memra-engine/cu")
files = sorted(list(root.glob("*.cu")) + list(root.glob("*.cuh")))
text = {f: f.read_text() for f in files}
launched = set()
for f, s in text.items():
    if f.name == "memra_pdl_chain.cuh":
        continue
    for m in re.finditer(r"memra_chain_launch\(\s*(?:\\\n|\s)*([A-Za-z_]\w*)", s):
        launched.add(m.group(1))


def bodies(s):
    """(kernel name, text after the body's opening brace) for each __global__ definition."""
    for m in re.finditer(r"__global__", s):
        line = s[s.rfind("\n", 0, m.start()) + 1 : m.start()]
        if line.lstrip().startswith("//"):
            continue
        depth, j, last_open = 0, m.end(), None
        while s[j] not in ";{" or depth:
            if s[j] == "(":
                if depth == 0:
                    last_open = j
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        if s[j] == ";" or last_open is None:
            continue
        name = re.search(r"(\w+)\s*$", s[m.end() : last_open]).group(1)
        yield name, s[last_open + 1 : j], s[j + 1 :]


def pointer_params(params):
    """{name: is_const} for each pointer parameter of a parameter list."""
    out = {}
    for p in params.split(","):
        p = p.strip()
        if "*" not in p:
            continue
        m = re.search(r"(\w+)\s*$", p)
        if m:
            out[m.group(1)] = p.split("*")[0].strip().startswith("const")
    return out


def pre_wait_problem(params, rest):
    """None if a MEMRA_PDL_PRE_WAIT prologue is admissible, else what is wrong with it."""
    head = re.match(r"\s*MEMRA_PDL_PRE_WAIT\(([^)]*)\);", rest)
    names = [n.strip() for n in head.group(1).split(",") if n.strip()]
    ptrs = pointer_params(params)
    for n in names:
        if n not in ptrs:
            return f"MEMRA_PDL_PRE_WAIT names {n}, which is not a pointer parameter"
        if not ptrs[n]:
            return f"MEMRA_PDL_PRE_WAIT names {n}, which is not a const pointer"
    entry = rest.find("MEMRA_PDL_CHAIN_ENTRY();", head.end())
    if entry < 0:
        return "MEMRA_PDL_PRE_WAIT without a MEMRA_PDL_CHAIN_ENTRY() after it"
    prologue = rest[head.end() : entry]
    for n in ptrs:
        # A member access (`threadIdx.x`, `v.y`) is not the parameter.
        if n not in names and re.search(r"(?<![\w.>])%s\b" % re.escape(n), prologue):
            return f"its prologue touches {n} before the entry"
    if re.search(r"\batomic\w*\s*\(", prologue):
        return "its prologue has an atomic before the entry"
    return None


found, bad = set(), []
for f, s in text.items():
    for name, params, rest in bodies(s):
        if name not in launched:
            continue
        found.add(name)
        if rest.lstrip().startswith("MEMRA_PDL_PRE_WAIT("):
            why = pre_wait_problem(params, rest)
            if why:
                bad.append(f"{f.name}: {name}: {why}")
        elif not rest.lstrip().startswith("MEMRA_PDL_CHAIN_ENTRY();"):
            bad.append(f"{f.name}: {name} does not open with MEMRA_PDL_CHAIN_ENTRY()")
missing = sorted(launched - found)
for line in bad + [f"no definition found for chain-launched kernel {n}" for n in missing]:
    print(line)
print(f"pdl-chain: {len(launched)} chain-launched kernels, {len(bad)} without the entry, {len(missing)} undefined")
sys.exit(1 if bad or missing else 0)
