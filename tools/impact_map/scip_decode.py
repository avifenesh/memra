#!/usr/bin/env python3
"""Minimal SCIP protobuf decoder (stdlib). Emits defs, refs and implementation relations.

Output (one JSON object per line, to OUT_DIR):
  defs.jsonl  {s: symbol, f: file, l: line0, el: [start_line0, end_line0] | null, k: kind, n: display}
  refs.tsv    file \t line0 \t symbol      (non-definition occurrences of local-crate symbols)
  impls.tsv   impl_symbol \t trait_symbol  (is_implementation relations)
"""
import json
import os
import sys


def varint(buf, i):
    shift = 0
    result = 0
    while True:
        b = buf[i]
        i += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, i
        shift += 7


def fields(buf, i=0, end=None):
    end = len(buf) if end is None else end
    while i < end:
        key, i = varint(buf, i)
        fno, wt = key >> 3, key & 7
        if wt == 0:
            v, i = varint(buf, i)
            yield fno, wt, v
        elif wt == 2:
            ln, i = varint(buf, i)
            yield fno, wt, (i, i + ln)
            i += ln
        elif wt == 1:
            yield fno, wt, buf[i:i + 8]
            i += 8
        elif wt == 5:
            yield fno, wt, buf[i:i + 4]
            i += 4
        else:
            raise ValueError(f"wire type {wt}")


def packed(buf, span):
    i, end = span
    out = []
    while i < end:
        v, i = varint(buf, i)
        out.append(v)
    return out


def main(path, out_dir):
    buf = open(path, "rb").read()
    os.makedirs(out_dir, exist_ok=True)
    fd = open(os.path.join(out_dir, "defs.jsonl"), "w")
    fr = open(os.path.join(out_dir, "refs.tsv"), "w")
    fi = open(os.path.join(out_dir, "impls.tsv"), "w")
    ndocs = nocc = 0
    for fno, wt, v in fields(buf):
        if fno != 2:
            continue
        ndocs += 1
        rel = None
        occs = []
        syms = []
        for dfno, dwt, dv in fields(buf, *v):
            if dfno == 1:
                rel = buf[dv[0]:dv[1]].decode()
            elif dfno == 2:
                occs.append(dv)
            elif dfno == 3:
                syms.append(dv)
        kinds = {}
        for sv in syms:
            sym = None
            kind = 0
            disp = None
            impls = []
            for sfno, swt, sval in fields(buf, *sv):
                if sfno == 1:
                    sym = buf[sval[0]:sval[1]].decode()
                elif sfno == 5:
                    kind = sval
                elif sfno == 6:
                    disp = buf[sval[0]:sval[1]].decode()
                elif sfno == 4:
                    rsym = None
                    is_impl = False
                    for rfno, rwt, rv in fields(buf, *sval):
                        if rfno == 1:
                            rsym = buf[rv[0]:rv[1]].decode()
                        elif rfno == 3:
                            is_impl = bool(rv)
                    if is_impl and rsym:
                        impls.append(rsym)
            if sym:
                kinds[sym] = (kind, disp)
                for r in impls:
                    fi.write(f"{sym}\t{r}\n")
        for ov in occs:
            rng = None
            sym = None
            roles = 0
            enc = None
            for ofno, owt, oval in fields(buf, *ov):
                if ofno == 1:
                    rng = packed(buf, oval) if owt == 2 else None
                elif ofno == 2:
                    sym = buf[oval[0]:oval[1]].decode()
                elif ofno == 3:
                    roles = oval
                elif ofno == 7:
                    enc = packed(buf, oval) if owt == 2 else None
            if not sym or not rng:
                continue
            nocc += 1
            line = rng[0]
            if roles & 1:
                el = None
                if enc:
                    el = [enc[0], enc[2] if len(enc) == 4 else enc[0]]
                k, disp = kinds.get(sym, (0, None))
                fd.write(json.dumps({"s": sym, "f": rel, "l": line, "el": el, "k": k, "n": disp}) + "\n")
            elif not sym.startswith("local "):
                fr.write(f"{rel}\t{line}\t{sym}\n")
    print(f"docs={ndocs} occurrences={nocc}", file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
