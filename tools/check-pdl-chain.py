#!/usr/bin/env python3
"""Refuse a chain-launched kernel whose body does not open with MEMRA_PDL_CHAIN_ENTRY().

`memra_chain_launch` (crates/memra-engine/cu/memra_pdl_chain.cuh) may give a kernel the
programmatic stream serialization attribute, which lets its blocks start while the previous kernel
on the stream still runs. That is correct only when the kernel waits for its predecessor before its
first load or store, so the entry must be the body's first statement. Exit 1 lists every kernel
that breaks this, and every launched kernel whose definition was not found.

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
        yield name, s[j + 1 :]


found, bad = set(), []
for f, s in text.items():
    for name, rest in bodies(s):
        if name not in launched:
            continue
        found.add(name)
        if not rest.lstrip().startswith("MEMRA_PDL_CHAIN_ENTRY();"):
            bad.append(f"{f.name}: {name} does not open with MEMRA_PDL_CHAIN_ENTRY()")
missing = sorted(launched - found)
for line in bad + [f"no definition found for chain-launched kernel {n}" for n in missing]:
    print(line)
print(f"pdl-chain: {len(launched)} chain-launched kernels, {len(bad)} without the entry, {len(missing)} undefined")
sys.exit(1 if bad or missing else 0)
