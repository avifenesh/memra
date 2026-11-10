#!/usr/bin/env python3
"""Collect every argmax-margin-probe row the repo holds into one TSV.

Reads three kinds of receipt:
  * probe tables (`pos prefill_top1 margin_p decode_top1 margin_d delta@ids agree`) in any
    text file under the given roots: raw gate logs, cell-evidence battery logs, lane notes;
  * the split-rung lane's `am-*.txt` cells (same table, wrapped with the gate's own output);
  * the kv-quant lane's `rows.jsonl.gz` (top-5 logits per program per row, three pairs).

One output row per (source file, table, position). Nothing is filtered or deduplicated here;
`analyze.py` does both and says what it dropped.

usage: collect.py --out rows.tsv [--tables ROOT:LABEL ...] [--kvq DIR:LABEL ...]
  ROOT is searched for files that hold a probe table header. LABEL names the tree (a branch
  and commit) in the `tree` column.
"""
import argparse
import gzip
import json
import os
import re
import sys

HEADER = "prefill_top1  margin_p"
ROW = re.compile(
    r"^(\d+)\s+(\d+)\s+([0-9.eE+-]+)\s+(\d+)\s+([0-9.eE+-]+)\s+([0-9.eE+-]+)\s+(yes|NO)\b"
)
PROBE = re.compile(r"^argmax-margin-probe: T=(\d+) window=(\d+) sm_count=(\d+) model=(\S+)")
GATE = re.compile(
    r"^argmax-margin-gate: model=(\S+) prompt=(\S+) window=(\d+)(?: decode=(\S+))? canary=(\d)"
)
GATE_FILE = re.compile(r"^gate-(.+?)-(board-\d+|p4-16k|deep-\d+|[A-Za-z0-9_]+-\d+k?)(\.batch)?(\.canary)?\.log$")
CELL_DIR = re.compile(r"/cells/([0-9a-f]{12})-([a-z0-9]+)-([a-z-]+)/")
COLUMNS = [
    "tree", "source", "table", "device", "sm_count", "model", "prompt", "T", "arm", "pair",
    "pos", "top_x", "margin_x", "top_y", "margin_y", "spread", "flip", "gap",
]


import hashlib  # noqa: E402

PROMPT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "e2e", "prompts")
PROMPT_OF = {}
for _name in sorted(os.listdir(PROMPT_DIR)) if os.path.isdir(PROMPT_DIR) else []:
    if _name.endswith(".txt"):
        with open(os.path.join(PROMPT_DIR, _name), "rb") as _f:
            PROMPT_OF[hashlib.sha256(_f.read()).hexdigest()] = _name[:-4]


def model_name(path):
    name = os.path.basename(path)
    return name[:-5] if name.endswith(".gguf") else name


def scan_table_file(tree, root, rel, emit):
    path = os.path.join(root, rel)
    try:
        with open(path, encoding="utf-8", errors="replace") as handle:
            lines = handle.read().splitlines()
    except OSError:
        return
    base = os.path.basename(rel)
    from_name = GATE_FILE.match(base)
    # A raw probe log banked with its cell's result.txt (model, arm and prompt sha256 there).
    result = {}
    try:
        with open(os.path.join(os.path.dirname(path), "result.txt"), encoding="utf-8") as handle:
            for line in handle:
                for field in line.split():
                    if "=" in field:
                        k, v = field.split("=", 1)
                        result[k] = v
    except OSError:
        pass
    cell = CELL_DIR.search("/" + rel)
    ctx = {}
    gate = {}
    table = -1
    in_table = False
    for line in lines:
        m = GATE.match(line.strip())
        if m:
            gate = {
                "model": model_name(m.group(1)),
                "prompt": m.group(2).removesuffix(".txt"),
                "arm": m.group(4) or "eager",
                "canary": m.group(5) == "1",
            }
            continue
        m = PROBE.match(line)
        if m:
            ctx = {
                "T": m.group(1),
                "sm_count": m.group(3),
                "model": model_name(m.group(4)),
                "arm": None,
            }
            continue
        if line.startswith("decode program: served batch"):
            ctx["arm"] = "batch"
            continue
        if line.startswith("decode program: served per-session eager"):
            ctx["arm"] = "batch-eager-routed"
            continue
        if HEADER in line and line.lstrip().startswith("pos"):
            table += 1
            in_table = True
            continue
        if not in_table:
            continue
        m = ROW.match(line)
        if not m:
            if line.strip() == "":
                in_table = False
            continue
        pos, p1, mp, d1, md, delta, agree = m.groups()
        arm = ctx.get("arm") or gate.get("arm")
        if arm is None and from_name:
            arm = "batch" if from_name.group(3) else "eager"
        arm = arm or "eager"
        prompt = gate.get("prompt") or (from_name.group(2) if from_name else "")
        if not prompt and "argmax16k" in rel:
            prompt = "p4-16k"
        if not prompt and ("board2048" in base or "board-2048" in base):
            prompt = "board-2048"
        if not prompt and result.get("prompt_sha256") in PROMPT_OF:
            prompt = PROMPT_OF[result["prompt_sha256"]]
        model = ctx.get("model") or gate.get("model") or (from_name.group(1) if from_name else "")
        emit({
            "tree": tree,
            "source": rel,
            "table": table,
            "device": cell.group(2) if cell else "",
            "sm_count": ctx.get("sm_count", ""),
            "model": model,
            "prompt": prompt,
            "T": ctx.get("T", ""),
            "arm": arm,
            "pair": "prefill:" + arm,
            "pos": pos,
            "top_x": p1,
            "margin_x": mp,
            "top_y": d1,
            "margin_y": md,
            "spread": delta,
            "flip": "1" if agree == "NO" else "0",
            "gap": "",
        })


def scan_tables(tree, root, emit):
    for sub in ("research", "docs"):
        top = os.path.join(root, sub)
        for dirpath, _dirs, files in os.walk(top):
            for name in sorted(files):
                if name.endswith((".injected", ".gz", ".png", ".bin", ".safetensors", ".gguf")):
                    continue
                path = os.path.join(dirpath, name)
                try:
                    if os.path.getsize(path) > 64 << 20:
                        continue
                    with open(path, "rb") as handle:
                        if HEADER.encode() not in handle.read():
                            continue
                except OSError:
                    continue
                scan_table_file(tree, root, os.path.relpath(path, root), emit)


def contender_gap(x_top, y_top):
    """The contender gap shift from two top-5 lists, or "" when a needed logit is not in them.

    Flip (x's top-1 i, y's top-1 j): (x_i - x_j) + (y_j - y_i), how far the two programs
    disagree about i against j. Agreement (shared top-1 i): the largest
    |(x_i - x_k) - (y_i - y_k)| over the two runner-ups k.
    """
    xl, yl = dict(x_top), dict(y_top)
    i, j = x_top[0][0], y_top[0][0]
    if i != j:
        if j not in xl or i not in yl:
            return ""
        return repr((xl[i] - xl[j]) + (yl[j] - yl[i]))
    worst = 0.0
    for k in {x_top[1][0], y_top[1][0]}:
        if k not in xl or k not in yl:
            return ""
        worst = max(worst, abs((xl[i] - xl[k]) - (yl[i] - yl[k])))
    return repr(worst)


def scan_kvq(tree, lane_dir, emit):
    """kv-quant rows: a = f32-KV prefill, b = prime over the quantized cache, c = decode over
    the quantized cache (the run's --decode arm). Pair ac is the gate's own pair."""
    runs = os.path.join(lane_dir, "runs")
    names = {"a": "prefill", "b": "prime", "c": None}
    for run in sorted(os.listdir(runs)):
        path = os.path.join(runs, run, "rows.jsonl.gz")
        if not os.path.exists(path):
            continue
        parts = run.split("-")
        arm = parts[-1]
        prompt = "-".join(parts[:-1])
        rel = os.path.join(os.path.basename(lane_dir), "runs", run, "rows.jsonl.gz")
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for pair in ("ab", "ac", "bc"):
                    if pair not in row:
                        continue
                    x, y = row[pair[0]], row[pair[1]]
                    side = lambda k: names[k] or arm
                    emit({
                        "tree": tree,
                        "source": rel,
                        "table": 0,
                        "device": "gb10",
                        "sm_count": "48",
                        "model": "Qwen3.8-27B-NVFP4-Q5K-mtp",
                        "prompt": prompt,
                        "T": parts[-2],
                        "arm": arm,
                        "pair": side(pair[0]) + ":" + side(pair[1]),
                        "pos": row["row"],
                        "top_x": x["top"][0][0],
                        "margin_x": repr(x["top"][0][1] - x["top"][1][1]),
                        "top_y": y["top"][0][0],
                        "margin_y": repr(y["top"][0][1] - y["top"][1][1]),
                        "spread": repr(row[pair]["delta"]),
                        "flip": "1" if row[pair]["flip"] else "0",
                        "gap": contender_gap(x["top"], y["top"]),
                    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--tables", action="append", default=[])
    ap.add_argument("--kvq", action="append", default=[])
    args = ap.parse_args()
    count = 0
    with open(args.out, "w", encoding="utf-8") as out:
        out.write("\t".join(COLUMNS) + "\n")

        def emit(rec):
            nonlocal count
            out.write("\t".join(str(rec[c]) for c in COLUMNS) + "\n")
            count += 1

        for spec in args.tables:
            root, label = spec.rsplit(":", 1)
            scan_tables(label, root, emit)
        for spec in args.kvq:
            lane_dir, label = spec.rsplit(":", 1)
            scan_kvq(label, lane_dir, emit)
    print(f"{count} rows -> {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
