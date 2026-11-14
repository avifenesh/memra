#!/usr/bin/env python3
"""The fields dsv41-tpbench records for each row-block call of a `perf_cfix_four.sh` A/B, by arm.

usage: call_fields.py RECEIPT_DIR [RANKS]   (reads RECEIPT_DIR/runs/run*-{A,B}/cells.jsonl or .gz; markdown on stdout)

tpbench has no per-launch timing. What a call's row carries:
  ttft_s                    the call to its first token read back
  forward_s                 the host's time to issue the forward (it returns before the cards finish)
  readback_wait_s_per_rank  host time blocked on read-backs, a rank
  comm                      the gathers of the call: collectives, copies, bytes (a region counted once a peer)
Round 0 of every cell is left out (warm-up). Medians over the remaining calls of an arm's runs.

With RANKS, the last table gives what one rank stores in the split MoE's hidden gather (its region times its
peers) against the push limit of `dsv41_gpu/tp.rs` (224 KiB): 7 rows of the hidden width a call row (6 routed
experts and the shared expert), a rank's 1 / RANKS of 2,304 columns, as bf16 (2 bytes) or as FP8 codes (1 byte).
"""
import glob
import gzip
import json
import os
import statistics as st
import sys

PUSH_LIMIT = 224 << 10
INTER, PAIRS_A_ROW = 2304, 7


def lines(path):
    if os.path.exists(path):
        return open(path)
    return gzip.open(path + ".gz", "rt")


def main():
    root = sys.argv[1].rstrip("/")
    ranks = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    acc = {}
    for d in sorted(glob.glob(os.path.join(root, "runs", "run*-[AB]"))):
        arm = d[-1]
        for line in lines(os.path.join(d, "cells.jsonl")):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["kind"] != "prefill" or r["round"] == 0:
                continue
            a = acc.setdefault((r["prompt_len"], arm), {"ttft": [], "fwd": [], "rb": [], "comm": set()})
            a["ttft"].append(r["ttft_s"] * 1e3)
            a["fwd"].append(r["forward_s"] * 1e3)
            a["rb"].append(r["readback_wait_s_per_rank"] * 1e3)
            a["comm"].add((r["comm"]["collectives"], r["comm"]["copies"], r["comm"]["bytes"]))
    widths = sorted({m for m, _ in acc})
    print(f"## {os.path.basename(root)}: a call's recorded fields by arm (medians, ms; round 0 left out)\n")
    print("| rows | arm | calls | first token | host issue (forward_s) | read-back wait a rank | collectives | copies | bytes |")
    print("|---|---|---|---|---|---|---|---|---|")
    for m in widths:
        for arm in "AB":
            a = acc[(m, arm)]
            comm = sorted(a["comm"])
            c = " / ".join(str(x[0]) for x in comm)
            p = " / ".join(str(x[1]) for x in comm)
            b = " / ".join(f"{x[2]:,}" for x in comm)
            print(f"| {m} | {arm} | {len(a['ttft'])} | {st.median(a['ttft']):.3f} | {st.median(a['fwd']):.3f} | "
                  f"{st.median(a['rb']):.3f} | {c} | {p} | {b} |")
    print("\nB minus A, ms:\n")
    print("| rows | first token | host issue | read-back wait a rank | bytes B / A |")
    print("|---|---|---|---|---|")
    for m in widths:
        A, B = acc[(m, "A")], acc[(m, "B")]
        ba = max(x[2] for x in B["comm"]) / max(x[2] for x in A["comm"])
        print(f"| {m} | {st.median(B['ttft']) - st.median(A['ttft']):+.3f} | "
              f"{st.median(B['fwd']) - st.median(A['fwd']):+.3f} | {st.median(B['rb']) - st.median(A['rb']):+.3f} | {ba:.3f} |")
    if ranks > 1:
        col = INTER // ranks
        print(f"\nThe split MoE's hidden gather at {ranks} ranks: what a rank stores (region x {ranks - 1} peers), "
              f"KiB, against the push limit of {PUSH_LIMIT >> 10} KiB:\n")
        print("| rows | bf16 rows | inside kernels | FP8 codes | inside kernels |")
        print("|---|---|---|---|---|")
        for m in sorted(set(widths) | {10, 14, 15}):
            bf = PAIRS_A_ROW * m * col * 2 * (ranks - 1)
            cd = PAIRS_A_ROW * m * col * (ranks - 1)
            print(f"| {m} | {bf / 1024:.1f} | {'yes' if bf <= PUSH_LIMIT else 'no'} | {cd / 1024:.1f} | "
                  f"{'yes' if cd <= PUSH_LIMIT else 'no'} |")


main()
