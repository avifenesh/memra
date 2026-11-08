#!/usr/bin/env python3
"""CUDA/C++ item scanner (stdlib): function definitions with spans, qualifiers, extern "C",
token hash, identifiers referenced, and kernel launches (<<<>>> / cudaLaunchKernel / FuncSetAttribute).

Usage: cu_items.py FILE... > cu_items.jsonl  (paths relative to the crate dir, e.g. cu/qmatvec.cu)
"""
import hashlib
import json
import re
import sys

ID = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
NUM = re.compile(r"(?:0[xX][0-9a-fA-F']+|[0-9][0-9']*\.?[0-9']*(?:[eE][+-]?[0-9]+)?|\.[0-9]+(?:[eE][+-]?[0-9]+)?)[a-zA-Z0-9_]*")


def lex(src):
    toks = []  # (kind, text, line)
    i, n, line = 0, len(src), 1
    bol = True
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            bol = True
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        if bol and c == "#":
            # preprocessor directive, with continuations
            j = i
            start = line
            while j < n:
                if src[j] == "\n":
                    if src[j - 1] == "\\" or (src[j - 1] == "\r" and src[j - 2] == "\\"):
                        line += 1
                        j += 1
                        continue
                    break
                if src.startswith("/*", j):
                    e = src.find("*/", j + 2)
                    e = n if e < 0 else e
                    line += src.count("\n", j, e)
                    j = e + 2
                    continue
                if src.startswith("//", j):
                    while j < n and src[j] != "\n":
                        j += 1
                    break
                j += 1
            text = src[i:j]
            toks.append(("pp", re.sub(r"\s+", " ", text.replace("\\\n", " ")).strip(), start))
            i = j
            continue
        bol = False
        if src.startswith("//", i):
            while i < n and src[i] != "\n":
                i += 1
            continue
        if src.startswith("/*", i):
            e = src.find("*/", i + 2)
            e = n if e < 0 else e
            line += src.count("\n", i, e)
            i = e + 2
            continue
        if c == "R" and src.startswith('R"', i):
            m = re.match(r'R"([^(]*)\(', src[i:])
            if m:
                close = ")" + m.group(1) + '"'
                e = src.find(close, i)
                e = n if e < 0 else e + len(close)
                toks.append(("str", src[i:e], line))
                line += src.count("\n", i, e)
                i = e
                continue
        if c == '"' or c == "'":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                elif src[j] == "\n":
                    break
                j += 1
            toks.append(("str" if c == '"' else "chr", src[i:j + 1], line))
            i = j + 1
            continue
        m = ID.match(src, i)
        if m:
            toks.append(("id", m.group(0), line))
            i = m.end()
            continue
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            m = NUM.match(src, i)
            toks.append(("num", m.group(0), line))
            i = m.end()
            continue
        if src.startswith("<<<", i) or src.startswith(">>>", i) or src.startswith("::", i):
            toks.append(("p", src[i:i + 3] if src[i] in "<>" else "::", line))
            i += 3 if src[i] in "<>" else 2
            continue
        toks.append(("p", c, line))
        i += 1
    return toks


def match_close(toks, i):
    o = toks[i][1]
    c = {"{": "}", "(": ")", "[": "]"}[o]
    d = 0
    for j in range(i, len(toks)):
        t = toks[j]
        if t[0] == "p":
            if t[1] == o:
                d += 1
            elif t[1] == c:
                d -= 1
                if d == 0:
                    return j
    return len(toks) - 1


QUALS = {"__global__", "__device__", "__host__", "__forceinline__", "inline", "static", "extern",
         "template", "constexpr", "__noinline__", "__launch_bounds__"}
NOT_FN = {"if", "for", "while", "switch", "return", "sizeof", "decltype", "static_assert", "alignas",
          "__launch_bounds__", "__align__", "defined", "catch"}


def scan(path, src):
    toks = lex(src)
    items = []
    i = 0
    n = len(toks)
    ns = []  # (kind, name, close_index)
    seg_start = 0  # start of current declaration
    extern_c_depth = []
    while i < n:
        k, t, ln = toks[i]
        while ns and i > ns[-1][2]:
            ns.pop()
        if k == "pp":
            items.append(dict(kind="pp", name=t.split()[0] if t else "#", line_start=ln, line_end=ln, text=t))
            i += 1
            seg_start = i
            continue
        if k == "p" and t in (";", "}"):
            i += 1
            seg_start = i
            continue
        if k == "id" and t == "namespace":
            j = i + 1
            name = ""
            while j < n and toks[j][1] != "{" and toks[j][1] != ";":
                if toks[j][0] == "id":
                    name += toks[j][1]
                j += 1
            if j < n and toks[j][1] == "{":
                ns.append(("ns", name, match_close(toks, j)))
                i = j + 1
                seg_start = i
                continue
        if k == "id" and t == "extern" and i + 2 < n and toks[i + 1][0] == "str" and toks[i + 2][1] == "{":
            ns.append(("externC", "", match_close(toks, i + 2)))
            i += 3
            seg_start = i
            continue
        if k == "p" and t == "{":
            # what does this brace open? look back over the declaration segment
            seg = toks[seg_start:i]
            words = [x[1] for x in seg if x[0] == "id"]
            close = match_close(toks, i)
            if seg and ("struct" in words or "class" in words or "union" in words or "enum" in words) and not any(
                    x[1] == "(" for x in seg if x[0] == "p"):
                # aggregate: scan its body for methods too
                name = next((words[j + 1] for j, w in enumerate(words[:-1]) if w in ("struct", "class", "union", "enum")), "?")
                items.append(dict(kind="type", name=name, line_start=seg[0][2], line_end=toks[close][2],
                                  toks=toks[seg_start:close + 1], ns=[x[1] for x in ns]))
                ns.append(("type", name, close))
                i += 1
                seg_start = i
                continue
            # function definition: segment has  name ( ... )  [const|noexcept|-> T]  before {
            fname = None
            p = None
            depth = 0
            for j in range(len(seg) - 1, -1, -1):
                x = seg[j]
                if x[1] == ")":
                    depth += 1
                elif x[1] == "(":
                    depth -= 1
                    if depth == 0:
                        # name before this paren (skip template args <...>)
                        q = j - 1
                        if q >= 0 and seg[q][1] == ">":
                            d2 = 0
                            while q >= 0:
                                if seg[q][1] == ">":
                                    d2 += 1
                                elif seg[q][1] == "<":
                                    d2 -= 1
                                    if d2 == 0:
                                        q -= 1
                                        break
                                q -= 1
                        if q >= 0 and seg[q][0] == "id" and seg[q][1] not in NOT_FN:
                            fname = seg[q][1]
                            p = j
                            if q > 0 and seg[q - 1][1] == "::" and q > 1 and seg[q - 2][0] == "id":
                                fname = seg[q - 2][1] + "::" + fname
                            # keep searching outward only if this was an attribute like __launch_bounds__
                        break
            if fname and seg and seg[0][1] not in NOT_FN:
                body = toks[seg_start:close + 1]
                quals = sorted({w for w in words if w in QUALS or w.startswith("__")})
                externc = any(x[0] == "externC" for x in ns) or (
                        "extern" in words and any(x[0] == "str" and x[1] == '"C"' for x in seg))
                launches = set()
                for q in range(i, close):
                    if toks[q][1] == "<<<":
                        # kernel name: identifier before <<< (skip template args)
                        r = q - 1
                        if toks[r][1] == ">":
                            d2 = 0
                            while r > i:
                                if toks[r][1] == ">":
                                    d2 += 1
                                elif toks[r][1] == "<":
                                    d2 -= 1
                                    if d2 == 0:
                                        r -= 1
                                        break
                                r -= 1
                        if toks[r][0] == "id":
                            launches.add(toks[r][1])
                    if toks[q][0] == "id" and toks[q][1] in ("cudaLaunchKernel", "cudaLaunchKernelEx", "cudaFuncSetAttribute",
                                                            "cudaOccupancyMaxActiveBlocksPerMultiprocessor", "cuLaunchKernel"):
                        for r in range(q + 1, min(q + 40, close)):
                            if toks[r][0] == "id" and toks[r][1] not in ("void", "const", "reinterpret_cast", "static_cast", "cudaLaunchKernel", "cudaFuncSetAttribute"):
                                launches.add(toks[r][1])
                                break
                items.append(dict(kind="fn", name=fname, line_start=seg[0][2], line_end=toks[close][2],
                                  quals=quals, extern_c=externc, toks=body, ns=[x[1] for x in ns],
                                  launches=sorted(launches),
                                  template="template" in words))
                i = close + 1
                seg_start = i
                continue
            # other brace (initializer etc.)
            i = close + 1
            continue
        i += 1
    # tokens outside every function and type (file-scope constexpr and __constant__ data, launch
    # bounds split from their function by a directive, operator overloads, anything unparsed) form
    # one synthetic item the map never contains, so a change there is never invisible
    covered = {id(x) for it in items if "toks" in it for x in it["toks"]}
    outside = [x for x in toks if x[0] != "pp" and id(x) not in covered]
    items.append(dict(kind="outside", name="<outside>", line_start=1, line_end=toks[-1][2] if toks else 1,
                      toks=outside, ns=[]))
    out = []
    seen = {}
    for it in items:
        if it["kind"] == "pp":
            rec = dict(id=f"{path}::#{it['line_start']}", file=path, kind="pp", name=it["name"],
                       line_start=it["line_start"], line_end=it["line_end"], text=it["text"][:300])
            m = re.match(r"#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)", it["text"])
            if m:
                # a macro body can launch kernels or call device functions; its invokers inherit those
                rec["macro"] = m.group(1)
                rec["body_idents"] = sorted(set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", it["text"][m.end():])))
            out.append(rec)
            continue
        tk = it.pop("toks")
        q = "<outside>" if it["kind"] == "outside" else "::".join([x for x in it.get("ns", []) if x] + [it["name"]])
        c = seen.get(q, 0) + 1
        seen[q] = c
        qid = q if c == 1 else f"{q}#{c}"
        body = [x for x in tk if x[0] != "pp"]
        h = hashlib.sha256("\x00".join(x[1] for x in body).encode("utf-8", "surrogatepass")).hexdigest()[:20]
        lh = hashlib.sha256("\x00".join(f"{x[2] - it['line_start']}:{x[1]}" for x in body)
                            .encode("utf-8", "surrogatepass")).hexdigest()[:20]
        rec = dict(id=f"{path}::{qid}", file=path, kind=it["kind"], name=it["name"], qual=q,
                   line_start=it["line_start"], line_end=it["line_end"], hash=h, lhash=lh,
                   idents=sorted({x[1] for x in tk if x[0] == "id"}))
        for key in ("quals", "extern_c", "launches", "template"):
            if key in it:
                rec[key] = it[key]
        out.append(rec)
    return out


def main(argv):
    for path in argv:
        src = open(path, encoding="utf-8", errors="surrogateescape").read()
        for r in scan(path, src):
            sys.stdout.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
