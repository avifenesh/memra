#!/usr/bin/env python3
"""Name the first divergence between two `memra-hidden-trace-v1` traces.

A trace (MEMRA_HYPER_TRACE, written by memra-reference, the CUDA hyper walk and
`hidden_trace_engine`) holds one line per stage per layer, last token row only, as big-endian
f32 bit hex. This tool lines up the stages two runs share, in execution order, and gives one
verdict:

  exact   first stage@layer whose values differ in any bit (two runs of one program,
          e.g. a clean and a fault-injected reference run of the same fixture)
  f32     first stage@layer with rel_l2 = |b - a| / |a| above --rel (default 1e-5)
  quant   step vs drift, for an f32 reference against a quantized engine. A residual-stream
          stage STEPS at layer l when its error e = 1 - cosine exceeds --floor (1e-3) and is
          more than --jump (20) times the same stage's error at the previous layer. Gradual
          growth with depth is drift. The verdict names the first step and the branch stages
          of that layer whose error exceeds the floor.

Order-sensitive stages (default: `route`, the selected (expert, weight) pairs in selection
order) are reported but never decide a verdict: two top-k kernels may emit the same set in a
different order.

Inputs: two trace files, or (--table) a banked per-stage table with columns
`stage L n cos ...` (the layout of research/glm53-flash-bringup-20260827/forward-bisect-receipts).

Exit status: 0 clean (or drift only, in quant), 1 a divergence or step found, 2 incomparable
inputs (different token ids, missing stages, length mismatch, malformed lines).
"""
import argparse
import math
import struct
import sys

RESIDUAL = {"expand", "attn", "layer", "layer_hidden", "collapse"}
ORDER_SENSITIVE = {"route"}
# Trunk-level (layer -1) stages with a fixed place: the stream expansion runs before layer 0,
# the stream collapse and the logits after the last layer. `layer_hidden` lines are appended
# after the run, so file order alone cannot place these on families with no per-layer taps.
TRUNK_START = {"expand"}
TRUNK_END = {"collapse", "logits"}


class Incomparable(Exception):
    pass


def load_trace(path):
    tokens = None
    entries = {}
    order = []
    with open(path) as f:
        for number, line in enumerate(f, 1):
            fields = line.rstrip("\n").split("\t")
            if fields[0] == "format":
                if fields[1:] != ["memra-hidden-trace-v1"]:
                    raise Incomparable(f"{path}:{number}: format {fields[1:]}")
            elif fields[0] == "tokens":
                if tokens is None:
                    tokens = fields[1]
                elif tokens != fields[1]:
                    raise Incomparable(f"{path}:{number}: two different token lines")
            elif fields[0] == "stage":
                if len(fields) != 5:
                    raise Incomparable(f"{path}:{number}: malformed stage line")
                stage, layer, width = fields[1], int(fields[2]), int(fields[3])
                hexes = fields[4].split(",") if fields[4] else []
                if len(hexes) != width:
                    raise Incomparable(f"{path}:{number}: width {width} but {len(hexes)} values")
                key = (stage, layer)
                if key in entries:
                    # A second call at the same stage (e.g. a later chunk): keep the last row,
                    # which is the one the logits come from.
                    order.remove(key)
                entries[key] = hexes
                order.append(key)
    return tokens, entries, order


def execution_order(order):
    """Sort (stage, layer) keys: trunk-level stages written before any layer first, then layers
    in index order, then trunk-level stages written after the layers. Within one layer the
    file order stands: a trace is written in execution order, and lines appended after the
    run (`layer_hidden`) land after that layer's taps."""
    first_layer_pos = next((i for i, (_, l) in enumerate(order) if l >= 0), len(order))
    pos = {key: i for i, key in enumerate(order)}

    def key(k):
        stage, layer = k
        if layer >= 0:
            return (1, layer, pos[k])
        if stage in TRUNK_START:
            return (0, 0, pos[k])
        if stage in TRUNK_END:
            return (2, 0, pos[k])
        return (0 if pos[k] < first_layer_pos else 2, 0, pos[k])

    return sorted(order, key=key)


def metrics(a_hex, b_hex):
    a = [struct.unpack(">f", bytes.fromhex(h))[0] for h in a_hex]
    b = [struct.unpack(">f", bytes.fromhex(h))[0] for h in b_hex]
    bits_equal = a_hex == b_hex
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    nd = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    dot = sum(x * y for x, y in zip(a, b))
    cos = dot / (na * nb) if na > 0 and nb > 0 else (1.0 if na == nb == 0 else float("nan"))
    return {
        "bits_equal": bits_equal,
        "cos": cos,
        "rel": nd / na if na > 0 else (0.0 if nd == 0 else float("inf")),
        "max_abs": max((abs(x - y) for x, y in zip(a, b)), default=0.0),
        "n": len(a),
    }


def rows_from_traces(path_a, path_b):
    ta, ea, oa = load_trace(path_a)
    tb, eb, ob = load_trace(path_b)
    if ta is not None and tb is not None and ta != tb:
        raise Incomparable("the two traces ran different token ids")
    shared = [k for k in oa if k in eb]
    only_a = [k for k in oa if k not in eb]
    if not shared:
        raise Incomparable("the traces share no stage")
    rows = []
    for k in execution_order(shared):
        if len(ea[k]) != len(eb[k]):
            raise Incomparable(f"{k[0]}@{k[1]}: {len(ea[k])} vs {len(eb[k])} values")
        rows.append((k[0], k[1], metrics(ea[k], eb[k])))
    return rows, only_a


def rows_from_table(path):
    rows, order = [], []
    for line in open(path):
        fields = line.split()
        if len(fields) < 4 or fields[0] == "stage":
            continue
        try:
            stage, layer, n, cos = fields[0], int(fields[1]), int(fields[2]), float(fields[3])
        except ValueError:
            continue
        key = (stage, layer)
        if key in order:
            continue
        order.append(key)
        rows.append((stage, layer, {"bits_equal": False, "cos": cos, "rel": float("nan"),
                                    "max_abs": float("nan"), "n": n}))
    by_key = {(s, l): m for s, l, m in rows}
    return [(s, l, by_key[(s, l)]) for s, l in execution_order(order)], []


def verdict(rows, mode, args):
    order_sensitive = set(args.order_sensitive.split(",")) if args.order_sensitive else set()
    deciding = [(s, l, m) for s, l, m in rows if s not in order_sensitive]
    if mode in ("exact", "f32"):
        for s, l, m in deciding:
            bad = (not m["bits_equal"]) if mode == "exact" else not (m["rel"] <= args.rel)
            if bad:
                return 1, f"FIRST DIVERGENCE at {s}@{l} (cos {m['cos']:.6f}, rel {m['rel']:.3g}, max_abs {m['max_abs']:.3g})", (s, l)
        return 0, f"CLEAN: {len(deciding)} shared stages agree ({mode})", None
    # quant: step vs drift on the residual stream
    prev = {}
    for s, l, m in deciding:
        e = 1.0 - m["cos"]
        if s in RESIDUAL and l >= 0:
            p = prev.get(s)
            if e > args.floor and (p is None or e > args.jump * max(p, 1e-6)):
                suspects = [f"{bs}@{bl} (err {1 - bm['cos']:.3g})" for bs, bl, bm in rows
                            if bl == l and bs not in RESIDUAL and 1 - bm["cos"] > args.floor]
                return 1, (f"STEP at {s}@{l}: err {e:.3g} vs {p if p is not None else 0:.3g} at the "
                           f"previous layer; branch stages above the floor in layer {l}: "
                           f"{', '.join(suspects) if suspects else 'none'}"), (s, l)
            prev[s] = e
    drift = [(s, l) for s, l, m in deciding if s in RESIDUAL and l >= 0 and 1 - m["cos"] > args.floor]
    if drift:
        return 0, f"DRIFT, no step: residual error first exceeds {args.floor:g} at {drift[0][0]}@{drift[0][1]}", None
    return 0, "CLEAN: no residual stage exceeds the floor", None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference trace (or --table file)")
    ap.add_argument("b", nargs="?", help="candidate trace")
    ap.add_argument("--table", action="store_true", help="read one banked per-stage table instead of two traces")
    ap.add_argument("--mode", choices=("exact", "f32", "quant"), default="exact")
    ap.add_argument("--rel", type=float, default=1e-5)
    ap.add_argument("--floor", type=float, default=1e-3)
    ap.add_argument("--jump", type=float, default=20.0)
    ap.add_argument("--order-sensitive", default=",".join(sorted(ORDER_SENSITIVE)))
    ap.add_argument("--rows", action="store_true", help="print every compared stage")
    args = ap.parse_args()
    try:
        if args.table:
            if args.mode != "quant":
                raise Incomparable("a banked table carries cosines only: use --mode quant")
            rows, only_a = rows_from_table(args.a)
        else:
            if not args.b:
                raise Incomparable("two traces are required")
            rows, only_a = rows_from_traces(args.a, args.b)
    except (Incomparable, OSError, ValueError) as error:
        print(f"INCOMPARABLE: {error}")
        return 2
    if args.rows:
        for s, l, m in rows:
            print(f"{s:14s} {l:4d} n={m['n']:6d} cos={m['cos']:.6f} rel={m['rel']:.3g} max_abs={m['max_abs']:.3g} bits_equal={m['bits_equal']}")
    if only_a:
        print(f"note: {len(only_a)} stages only in {args.a}, e.g. {only_a[0][0]}@{only_a[0][1]}")
    code, text, _ = verdict(rows, args.mode, args)
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
