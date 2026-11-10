#!/usr/bin/env python3
"""Summarize kvq-rows-probe rows (memra#1177).

usage: summarize.py TOKENIZER_JSON PROMPTS_DIR OUT_MD CELL=ROWS_JSONL [CELL=ROWS_JSONL ...]
CELL is DOC-DEPTH-ARM, for example csv-32768-batch; the prompt is PROMPTS_DIR/DOC-DEPTH.txt.

Per cell and program pair it reports the argmax disagreements (flips), how many of them come
with a top-2 margin above 1 and above 2 logits on each side, which side holds the prompt's next
token, the distribution of the spread at the contending ids and of the largest vocabulary
difference, the top-2 margins, and the rows with a disagreement. Rows whose token ends a line
are record boundaries.

It does not count "margin-explained" flips (the gate's rule: spread above the smaller margin).
For a flip the spread over both top-2 id sets is at least the mean of the two margins, so the
rule holds for every flip by arithmetic (memra#1199).
"""
import json
import statistics
import struct
import sys
from pathlib import Path

from tokenizers import Tokenizer


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def margin(top):
    """Top-2 margin as the probe computes it: an f32 subtraction of two f32 logits."""
    return f32(f32(top[0][1]) - f32(top[1][1]))


def row_class(piece):
    """Where a row sits: `record` (its token ends a line, so the next token opens a record),
    `field` (its token ends in a separator, so the next token opens a field or value),
    `inside` (inside a value, a key or a word)."""
    if "\n" in piece:
        return "record"
    if piece.rstrip().endswith((",", ":", "{", "[", '"')):
        return "field"
    return "inside"


def q(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, round((len(xs) - 1) * p))]


def fmt(x):
    return f"{x:.3f}" if x == x else "n/a"


def main() -> None:
    tok = Tokenizer.from_file(sys.argv[1])
    prompts = Path(sys.argv[2])
    out_md = Path(sys.argv[3])
    cells = [a.split("=", 1) for a in sys.argv[4:]]
    md = []
    summary = []
    flips_md = []
    for name, path in cells:
        doc, depth, arm = name.rsplit("-", 2)
        text = (prompts / f"{doc}-{depth}.txt").read_text(encoding="utf-8")
        ids = tok.encode(text, add_special_tokens=False).ids
        rows = [json.loads(l) for l in open(path) if l.strip()]
        mism = sum(1 for r in rows if ids[r["row"]] != r["tok"])
        pairs = [p for p in ("ab", "ac", "bc") if p in rows[0]]
        for p in pairs:
            x, y = p[0], p[1]
            d = [r[p]["delta"] for r in rows]
            mx = [r[p]["maxdiff"] for r in rows]
            fl = [r for r in rows if r[p]["flip"]]
            exposed = sum(1 for r in rows if r[p]["delta"] > r[p]["min_margin"])
            cls = {r["row"]: row_class(tok.decode([r["tok"]])) for r in rows}
            by_cls = {}
            for c in ("record", "field", "inside"):
                rc = [r for r in rows if cls[r["row"]] == c]
                by_cls[c] = {
                    "rows": len(rc),
                    "flips": sum(1 for r in rc if r[p]["flip"]),
                    "spread_median": statistics.median([r[p]["delta"] for r in rc]) if rc else float("nan"),
                    "spread_max": max([r[p]["delta"] for r in rc]) if rc else float("nan"),
                }
            mar_x = [margin(r[x]["top"]) for r in rows]
            mar_y = [margin(r[y]["top"]) for r in rows]
            acc_x = sum(1 for r in rows if r["next"] is not None and r[x]["top"][0][0] == r["next"])
            acc_y = sum(1 for r in rows if r["next"] is not None and r[y]["top"][0][0] == r["next"])
            n_next = sum(1 for r in rows if r["next"] is not None)
            row = {
                "cell": name,
                "pair": p,
                "rows": len(rows),
                "tok_mismatch_vs_hf": mism,
                "flips": len(fl),
                "exposed_rows": exposed,
                "spread_median": statistics.median(d),
                "spread_p90": q(d, 0.9),
                "rows_spread_over_3": sum(1 for v in d if v > 3),
                "spread_max": max(d),
                "by_class": by_cls,
                f"flips_margin_{x}_over_1": sum(1 for r in fl if margin(r[x]["top"]) > 1.0),
                f"flips_margin_{x}_over_2": sum(1 for r in fl if margin(r[x]["top"]) > 2.0),
                f"flips_margin_{y}_over_1": sum(1 for r in fl if margin(r[y]["top"]) > 1.0),
                f"flips_margin_{y}_over_2": sum(1 for r in fl if margin(r[y]["top"]) > 2.0),
                f"flips_{x}_right": sum(1 for r in fl if r[x]["top"][0][0] == r["next"]),
                f"flips_{y}_right": sum(1 for r in fl if r[y]["top"][0][0] == r["next"]),
                "maxdiff_median": statistics.median(mx),
                "maxdiff_max": max(mx),
                f"margin_{x}_min": min(mar_x),
                f"margin_{x}_p10": q(mar_x, 0.1),
                f"margin_{x}_median": statistics.median(mar_x),
                f"margin_{y}_min": min(mar_y),
                f"margin_{y}_p10": q(mar_y, 0.1),
                f"margin_{y}_median": statistics.median(mar_y),
                f"top1_is_next_{x}": acc_x,
                f"top1_is_next_{y}": acc_y,
                "rows_with_next": n_next,
            }
            summary.append(row)
            for r in fl:
                i = r["row"]
                ctx = tok.decode(ids[max(0, i - 24) : i + 1])
                nxt = tok.decode(ids[i + 1 : i + 4]) if i + 1 < len(ids) else ""
                xt, yt = r[x]["top"], r[y]["top"]
                flips_md.append(
                    f"| {name} | {p} | {i} | `{json.dumps(ctx[-60:])[1:-1]}` | "
                    f"`{json.dumps(nxt)[1:-1]}` | {json.dumps(tok.decode([xt[0][0]]))} "
                    f"{margin(xt):.3f} | {json.dumps(tok.decode([yt[0][0]]))} "
                    f"{margin(yt):.3f} | {r[p]['delta']:.3f} |"
                )
    md.append(
        "| cell | pair | rows | flips | flips, x margin > 1 | > 2 | flips, y margin > 1 | > 2 | "
        "flips x right | y right | rows with spread > smaller margin | spread median | p90 | max | "
        "max vocab diff median | max |"
    )
    md.append("|" + "---|" * 16)
    for s in summary:
        x, y = s["pair"][0], s["pair"][1]
        md.append(
            f"| {s['cell']} | {s['pair']} | {s['rows']} | {s['flips']} | "
            f"{s[f'flips_margin_{x}_over_1']} | {s[f'flips_margin_{x}_over_2']} | "
            f"{s[f'flips_margin_{y}_over_1']} | {s[f'flips_margin_{y}_over_2']} | "
            f"{s[f'flips_{x}_right']} | {s[f'flips_{y}_right']} | {s['exposed_rows']} | "
            f"{fmt(s['spread_median'])} | {fmt(s['spread_p90'])} | {fmt(s['spread_max'])} | "
            f"{fmt(s['maxdiff_median'])} | {fmt(s['maxdiff_max'])} |"
        )
    md.append("")
    md.append(
        "| cell | pair | record rows | flips | spread median | max | field rows | flips | "
        "spread median | max | inside rows | flips | spread median | max |"
    )
    md.append("|" + "---|" * 14)
    for s in summary:
        cells = []
        for c in ("record", "field", "inside"):
            b = s["by_class"][c]
            cells.append(
                f"{b['rows']} | {b['flips']} | {fmt(b['spread_median'])} | {fmt(b['spread_max'])}"
            )
        md.append(f"| {s['cell']} | {s['pair']} | " + " | ".join(cells) + " |")
    md.append("")
    md.append(
        "| cell | pair | margin x min | p10 | median | margin y min | p10 | median | "
        "top1 = next token, x | y | rows |"
    )
    md.append("|" + "---|" * 11)
    for s in summary:
        x, y = s["pair"][0], s["pair"][1]
        md.append(
            f"| {s['cell']} | {s['pair']} | {fmt(s[f'margin_{x}_min'])} | "
            f"{fmt(s[f'margin_{x}_p10'])} | {fmt(s[f'margin_{x}_median'])} | "
            f"{fmt(s[f'margin_{y}_min'])} | {fmt(s[f'margin_{y}_p10'])} | "
            f"{fmt(s[f'margin_{y}_median'])} | {s[f'top1_is_next_{x}']} | "
            f"{s[f'top1_is_next_{y}']} | {s['rows_with_next']} |"
        )
    md.append("")
    md.append(
        "| cell | pair | row | text before (last 60 chars) | true next | x top, margin | "
        "y top, margin | spread |"
    )
    md.append("|" + "---|" * 8)
    md.extend(flips_md)
    out_md.write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(out_md.with_suffix(".jsonl"), "w") as f:
        for s in summary:
            f.write(json.dumps(s) + "\n")
    bad = [s for s in summary if s["tok_mismatch_vs_hf"]]
    if bad:
        print("token mismatch vs the HF tokenizer:", [(s["cell"], s["tok_mismatch_vs_hf"]) for s in bad])
    print(out_md.read_text())


if __name__ == "__main__":
    main()
