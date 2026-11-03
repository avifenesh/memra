#!/usr/bin/env python3
"""Rust item scanner (stdlib only): items with line spans, cfg, test flag and a token hash.

The hash covers the item's token stream with comments, doc comments and whitespace removed,
so a comment-only or rustfmt-only edit keeps the hash; any token change moves it.

Usage: rust_items.py FILE... > items.jsonl     (paths repo-relative)
       rust_items.py --git REF FILE...          (read blobs from a commit)
"""
import hashlib
import json
import re
import subprocess
import sys

IDENT_START = re.compile(r"[A-Za-z_\u0080-\U0010ffff]")
IDENT = re.compile(r"[A-Za-z0-9_\u0080-\U0010ffff]*")
NUM = re.compile(r"0[xob][0-9a-fA-F_]+[a-zA-Z0-9_]*|[0-9][0-9_]*(?:\.[0-9][0-9_]*)?(?:[eE][+-]?[0-9_]+)?[a-zA-Z0-9_]*")


class Tok:
    __slots__ = ("k", "t", "line")

    def __init__(self, k, t, line):
        self.k, self.t, self.line = k, t, line

    def __repr__(self):
        return f"{self.k}:{self.t}@{self.line}"


def lex(src):
    """Tokens: k in {id, lt, str, chr, num, p}. Comments are dropped."""
    out = []
    i, n, line = 0, len(src), 1
    if src.startswith("#!") and not src.startswith("#!["):
        while i < n and src[i] != "\n":
            i += 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v﻿":
            i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "/":
            while i < n and src[i] != "\n":
                i += 1
            continue
        if c == "/" and i + 1 < n and src[i + 1] == "*":
            depth, i = 1, i + 2
            while i < n and depth:
                if src.startswith("/*", i):
                    depth += 1
                    i += 2
                elif src.startswith("*/", i):
                    depth -= 1
                    i += 2
                else:
                    if src[i] == "\n":
                        line += 1
                    i += 1
            continue
        start_line = line
        # raw strings / byte strings / c strings / raw identifiers
        if c in "rbc":
            j = i
            if src.startswith("br", i) or src.startswith("cr", i):
                j = i + 2
            elif c == "r":
                j = i + 1
            if j > i and (src[j:j + 1] == "#" or src[j:j + 1] == '"') and src[i:j] in ("r", "br", "cr"):
                k = j
                while k < n and src[k] == "#":
                    k += 1
                hashes = k - j
                if k < n and src[k] == '"':
                    close = '"' + "#" * hashes
                    end = src.find(close, k + 1)
                    if end < 0:
                        end = n
                    body = src[i:end + len(close)]
                    line += body.count("\n")
                    out.append(Tok("str", body, start_line))
                    i = end + len(close)
                    continue
                if c == "r" and hashes == 1 and k < n and IDENT_START.match(src[k]):
                    m = IDENT.match(src, k + 1)
                    out.append(Tok("id", src[k:m.end()], start_line))
                    i = m.end()
                    continue
            if c in "bc" and src[i + 1:i + 2] == '"':
                i += 1
                c = '"'
                prefix = src[i - 1]
            elif c == "b" and src[i + 1:i + 2] == "'":
                k = i + 2
                if src[k:k + 1] == "\\":
                    k += 2
                else:
                    k += 1
                while k < n and src[k] != "'":
                    k += 1
                out.append(Tok("chr", src[i:k + 1], start_line))
                i = k + 1
                continue
        if c == '"':
            k = i + 1
            while k < n and src[k] != '"':
                if src[k] == "\\":
                    if src[k + 1:k + 2] == "\n":
                        line += 1
                    k += 2
                    continue
                if src[k] == "\n":
                    line += 1
                k += 1
            out.append(Tok("str", src[i:k + 1], start_line))
            i = k + 1
            continue
        if c == "'":
            if src[i + 1:i + 2] == "\\":
                k = i + 2
                if src[k:k + 1] == "u":
                    k = src.find("}", k) + 1
                else:
                    k += 1
                    while k < n and src[k] not in "'\n":
                        k += 1
                out.append(Tok("chr", src[i:k + 1], start_line))
                i = k + 1
                continue
            if i + 2 < n and src[i + 2] == "'":
                out.append(Tok("chr", src[i:i + 3], start_line))
                i += 3
                continue
            m = IDENT.match(src, i + 1)
            out.append(Tok("lt", src[i:m.end()], start_line))
            i = max(m.end(), i + 1)
            continue
        if IDENT_START.match(c):
            m = IDENT.match(src, i + 1)
            out.append(Tok("id", src[i:m.end()], start_line))
            i = m.end()
            continue
        if c.isdigit():
            m = NUM.match(src, i)
            out.append(Tok("num", m.group(0), start_line))
            i = m.end()
            continue
        out.append(Tok("p", c, start_line))
        i += 1
    return out


OPEN = {"{": "}", "(": ")", "[": "]"}
VIS_WORDS = {"pub", "crate", "super", "self", "in", "unsafe", "async", "const", "extern", "default", "safe"}
ITEM_KW = {"fn", "struct", "enum", "union", "trait", "impl", "mod", "type", "const", "static", "use",
           "macro_rules", "extern"}


def match_close(toks, i):
    """toks[i] is an opener; return index of its matching closer."""
    depth = 0
    for j in range(i, len(toks)):
        t = toks[j]
        if t.k == "p":
            if t.t in "{([":
                depth += 1
            elif t.t in "})]":
                depth -= 1
                if depth == 0:
                    return j
    return len(toks) - 1


def module_path(path):
    parts = path.split("/")
    crate = None
    rest = parts
    if len(parts) >= 2 and parts[0] == "crates":
        crate = parts[1].replace("-", "_")
        rest = parts[2:]
    if rest and rest[0] == "src":
        rest = rest[1:]
        if rest and rest[0] == "bin":
            # src/bin/x.rs -> bin x ; src/bin/x/main.rs -> bin x
            name = rest[1][:-3] if rest[1].endswith(".rs") else rest[1]
            return [f"{crate}[bin:{name}]"] + [p[:-3] if p.endswith(".rs") else p for p in rest[2:] if p != "main.rs"]
    mods = [crate or "?"]
    for p in rest:
        if p.endswith(".rs"):
            p = p[:-3]
            if p in ("lib", "main", "mod"):
                continue
        mods.append(p)
    return mods


def impl_name(header):
    """header: tokens after `impl` up to `{`. Returns 'Type' or 'Trait for Type'."""
    toks = [t for t in header]
    # drop leading generics <...>
    i = 0
    if i < len(toks) and toks[i].t == "<":
        depth = 0
        while i < len(toks):
            if toks[i].t == "<":
                depth += 1
            elif toks[i].t == ">":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
    words = toks[i:]
    # cut where clause
    for j, t in enumerate(words):
        if t.k == "id" and t.t == "where":
            words = words[:j]
            break

    def type_ident(seq):
        last = None
        depth = 0
        for t in seq:
            if t.t == "<":
                depth += 1
            elif t.t == ">":
                depth -= 1
            elif depth == 0 and t.k == "id" and t.t not in ("dyn", "mut", "const", "unsafe", "for"):
                last = t.t
        return last or "?"

    fi = None
    depth = 0
    for j, t in enumerate(words):
        if t.t == "<":
            depth += 1
        elif t.t == ">":
            depth -= 1
        elif depth == 0 and t.k == "id" and t.t == "for":
            fi = j
            break
    if fi is not None:
        return f"<{type_ident(words[:fi])} for {type_ident(words[fi + 1:])}>"
    return type_ident(words)


def norm_hash(toks):
    h = hashlib.sha256()
    for t in toks:
        h.update(t.t.encode("utf-8", "surrogatepass"))
        h.update(b"\x00")
    return h.hexdigest()[:20]


def cfg_of(attrs):
    cfgs = []
    test = False
    for a in attrs:
        txt = "".join(t.t if t.k != "id" else " " + t.t + " " for t in a)
        txt = re.sub(r"\s+", " ", txt).strip()
        if re.match(r"# \[ ?(test|tokio :: test|rstest)\b", txt.replace("  ", " ")) or txt.startswith("#[ test") or re.search(r"\[\s*(test|tokio\s*::\s*test)\s*\]", txt):
            test = True
        m = re.search(r"cfg\s*\((.*)\)\s*\]$", txt)
        if m and "cfg_attr" not in txt:
            c = re.sub(r"\s+", "", m.group(1))
            cfgs.append(c)
            if c == "test" or c.startswith("all(test") or c.startswith("any(test"):
                test = True
    return cfgs, test


DOC_ATTR = re.compile(r"^#!?\[doc")


def scan(path, src):
    toks = lex(src)
    items = []
    base_mods = module_path(path)

    def walk(lo, hi, mods, container, in_test, ctx_cfg):
        """Scan tokens[lo:hi] as a sequence of items."""
        i = lo
        while i < hi:
            start = i
            attrs = []
            # attributes
            while i < hi and toks[i].t == "#":
                j = i + 1
                inner = False
                if j < hi and toks[j].t == "!":
                    inner = True
                    j += 1
                if j < hi and toks[j].t == "[":
                    e = match_close(toks, j)
                    if not inner:
                        attrs.append(toks[i:e + 1])
                    i = e + 1
                else:
                    break
            if i >= hi:
                break
            cfgs, is_test = cfg_of(attrs)
            item_test = in_test or is_test
            # find keyword
            j = i
            while j < hi and toks[j].k == "id" and toks[j].t in VIS_WORDS and not (
                    toks[j].t in ("const", "extern", "unsafe") and j + 1 < hi and (
                    toks[j + 1].k == "id" and toks[j + 1].t not in VIS_WORDS | {"fn", "impl", "trait"}
                    or toks[j + 1].t in ("{",) or toks[j + 1].k == "str")):
                if toks[j].t == "pub" and j + 1 < hi and toks[j + 1].t == "(":
                    j = match_close(toks, j + 1) + 1
                    continue
                if toks[j].t == "extern" and j + 1 < hi and toks[j + 1].k == "str":
                    j += 2
                    continue
                j += 1
            kw_i = j
            kw = toks[kw_i].t if kw_i < hi else ""
            if kw == "extern" and kw_i + 2 < hi and toks[kw_i + 1].k == "str" and toks[kw_i + 2].t == "fn":
                kw_i += 2
                kw = "fn"
            first = toks[start]
            # extern "C" { ... } block
            if kw == "extern" and kw_i + 1 < hi and toks[kw_i + 1].k == "str" and kw_i + 2 < hi and toks[kw_i + 2].t == "{":
                o = kw_i + 2
                e = match_close(toks, o)
                walk(o + 1, e, mods, "extern", item_test, cfgs + ctx_cfg)
                i = e + 1
                continue
            if kw == "extern" and kw_i + 1 < hi and toks[kw_i + 1].t == "{":
                o = kw_i + 1
                e = match_close(toks, o)
                walk(o + 1, e, mods, "extern", item_test, cfgs + ctx_cfg)
                i = e + 1
                continue
            if kw == "extern" and kw_i + 1 < hi and toks[kw_i + 1].t == "crate":
                kw = "use"
            if kw in ("mod",):
                name = toks[kw_i + 1].t
                k = kw_i + 2
                if k < hi and toks[k].t == "{":
                    e = match_close(toks, k)
                    sub_test = item_test
                    walk(k + 1, e, mods + [name], "mod", sub_test, cfgs + ctx_cfg)
                    i = e + 1
                else:
                    i = k + 1  # mod x;
                continue
            if kw in ("impl", "trait") or (kw == "unsafe" and kw_i + 1 < hi and toks[kw_i + 1].t in ("impl", "trait")):
                if kw == "unsafe":
                    kw_i += 1
                    kw = toks[kw_i].t
                k = kw_i + 1
                depth = 0
                while k < hi:
                    t = toks[k]
                    if t.t == "<":
                        depth += 1
                    elif t.t == ">" and not (k > 0 and toks[k - 1].t == "-"):
                        depth -= 1
                    elif t.t == "{" and depth <= 0:
                        break
                    elif t.t == ";" and depth <= 0:
                        break
                    elif t.t in "([" :
                        k = match_close(toks, k)
                    k += 1
                if k >= hi or toks[k].t == ";":
                    i = k + 1
                    continue
                header = toks[kw_i + 1:k]
                if kw == "impl":
                    cname = impl_name(header)
                else:
                    cname = header[0].t if header else "?"
                e = match_close(toks, k)
                # record the container header as an item too (trait bounds, assoc consts change semantics)
                items.append(dict(kind=kw, name=cname, mods=mods, container=None, line_start=first.line,
                                  line_end=toks[k].line, cfg=cfgs + ctx_cfg, test=item_test,
                                  toks=toks[start:k + 1]))
                walk(k + 1, e, mods + [cname], kw, item_test, cfgs + ctx_cfg)
                i = e + 1
                continue
            if kw == "fn":
                name = toks[kw_i + 1].t if kw_i + 1 < hi else "?"
                k = kw_i + 2
                while k < hi and toks[k].t not in ("{", ";"):
                    if toks[k].t in "([":
                        k = match_close(toks, k)
                    k += 1
                if k < hi and toks[k].t == "{":
                    e = match_close(toks, k)
                else:
                    e = k
                items.append(dict(kind="fn", name=name, mods=mods, container=container,
                                  line_start=first.line, line_end=toks[min(e, len(toks) - 1)].line,
                                  cfg=cfgs + ctx_cfg, test=item_test, toks=toks[start:e + 1],
                                  body_line=toks[k].line if k < hi else first.line))
                i = e + 1
                continue
            if kw == "macro_rules" and kw_i + 2 < hi and toks[kw_i + 1].t == "!":
                name = toks[kw_i + 2].t
                k = kw_i + 3
                if toks[k].t in "{([":
                    e = match_close(toks, k)
                    if toks[k].t != "{" and e + 1 < hi and toks[e + 1].t == ";":
                        e += 1
                else:
                    e = k
                items.append(dict(kind="macro", name=name, mods=mods, container=container,
                                  line_start=first.line, line_end=toks[e].line, cfg=cfgs + ctx_cfg,
                                  test=item_test, toks=toks[start:e + 1]))
                i = e + 1
                continue
            if kw in ("struct", "enum", "union"):
                name = toks[kw_i + 1].t
                k = kw_i + 2
                while k < hi and toks[k].t not in ("{", ";"):
                    if toks[k].t in "([":
                        k = match_close(toks, k)
                        if k + 1 < hi and toks[k + 1].t == ";":
                            k += 1
                            break
                    k += 1
                e = match_close(toks, k) if k < hi and toks[k].t == "{" else k
                items.append(dict(kind=kw, name=name, mods=mods, container=container, line_start=first.line,
                                  line_end=toks[min(e, len(toks) - 1)].line, cfg=cfgs + ctx_cfg, test=item_test,
                                  toks=toks[start:e + 1]))
                i = e + 1
                continue
            # ;-terminated items: use, const, static, type, extern crate, macro calls, assoc items
            k = kw_i
            if kw in ("const", "static", "type", "use", "let"):
                name = None
                q = kw_i + 1
                while q < hi and toks[q].t in ("mut",):
                    q += 1
                if q < hi and toks[q].k == "id":
                    name = toks[q].t
                while k < hi and toks[k].t != ";":
                    if toks[k].t in "{([":
                        k = match_close(toks, k)
                    k += 1
                e = min(k, hi - 1)
                kind = {"let": "stmt"}.get(kw, kw)
                if kw == "use":
                    name = "use@" + str(first.line)
                items.append(dict(kind=kind, name=name or "_", mods=mods, container=container,
                                  line_start=first.line, line_end=toks[e].line, cfg=cfgs + ctx_cfg,
                                  test=item_test, toks=toks[start:e + 1]))
                i = e + 1
                continue
            # macro invocation at item level: path ! (...) ; or path ! { ... }
            q = kw_i
            while q < hi and (toks[q].k == "id" or toks[q].t == ":"):
                q += 1
            if q < hi and toks[q].t == "!" and q + 1 < hi and toks[q + 1].t in "{([":
                e = match_close(toks, q + 1)
                if toks[q + 1].t != "{" and e + 1 < hi and toks[e + 1].t == ";":
                    e += 1
                mname = toks[q - 1].t if q > kw_i else "?"
                # first identifier inside the invocation helps naming (thread_local!{static X})
                inner_name = None
                for t in toks[q + 2:e]:
                    if t.k == "id" and t.t not in ("static", "pub", "crate", "const", "mut", "ref", "fn"):
                        inner_name = t.t
                        break
                items.append(dict(kind="macro_call", name=f"{mname}!{inner_name or ''}", mods=mods,
                                  container=container, line_start=first.line, line_end=toks[e].line,
                                  cfg=cfgs + ctx_cfg, test=item_test, toks=toks[start:e + 1]))
                i = e + 1
                continue
            # unknown: skip to ; or balanced group
            k = i
            while k < hi and toks[k].t not in (";",):
                if toks[k].t in "{([":
                    k = match_close(toks, k)
                    if toks[k].t == "}":
                        break
                k += 1
            items.append(dict(kind="other", name="?", mods=mods, container=container,
                              line_start=first.line, line_end=toks[min(k, hi - 1)].line,
                              cfg=cfgs + ctx_cfg, test=item_test, toks=toks[start:k + 1]))
            i = k + 1

    walk(0, len(toks), base_mods, "mod", False, [])
    # stable ids
    seen = {}
    out = []
    for it in items:
        qual = "::".join(it["mods"] + [it["name"]])
        n = seen.get(qual, 0) + 1
        seen[qual] = n
        if n > 1:
            qual = f"{qual}#{n}"
        tk = it.pop("toks")
        idents = sorted({t.t for t in tk if t.k == "id"})
        strings = sorted({t.t for t in tk if t.k == "str" and len(t.t) < 200})
        rec = dict(id=f"{path}::{qual}", file=path, qual=qual, kind=it["kind"], name=it["name"],
                   line_start=it["line_start"], line_end=it["line_end"], cfg=it["cfg"], test=it["test"],
                   hash=norm_hash(tk), ntok=len(tk), container=it.get("container"))
        if it.get("body_line"):
            rec["body_line"] = it["body_line"]
        rec["idents"] = idents
        rec["strings"] = strings
        out.append(rec)
    return out


def main(argv):
    ref = None
    if argv and argv[0] == "--git":
        ref = argv[1]
        argv = argv[2:]
    if argv and argv[0] == "--files-from":
        argv = [l.strip() for l in open(argv[1]) if l.strip()]
    w = sys.stdout
    for path in argv:
        if ref:
            src = subprocess.run(["git", "show", f"{ref}:{path}"], capture_output=True).stdout.decode("utf-8", "surrogateescape")
        else:
            with open(path, encoding="utf-8", errors="surrogateescape") as f:
                src = f.read()
        for rec in scan(path, src):
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
