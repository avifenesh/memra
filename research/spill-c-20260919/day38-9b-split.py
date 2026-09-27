#!/usr/bin/env python3
"""Day 38 item 7: the 9B plain entry's byte split, from banked server logs only (no card, no artifact read).

Reads every `*server.log` under the given receipt dirs, counts the distinct shapes of the lines that carry an entry's
bytes (seq, digests and times masked), and does the arithmetic of DAY38.md section 3 on the printed figures. Every
MB figure the engine prints is `bytes / 1e6` at one decimal, so a printed `x.yMB` is the half-open interval
[x.y - 0.05, x.y + 0.05) MB; the bounds below carry that rounding and nothing else.

    day38-9b-split.py <receipt dir> [<receipt dir> ...]
"""
import os
import re
import sys
from collections import Counter

SHAPES = (
    ("demote_submitted", re.compile(r"\[prefix-host\] demote submitted off the tick: 64 tokens, ([0-9.]+)MB, ticket seq=\d+, (\d+) items")),
    ("d2h_receipt", re.compile(r"\[prefix-host\] contracts door D2H receipt: .* items=(\d+) \((\d+) KV planes(, draft)?\) complete=\d+ require=ok")),
    ("copy_complete", re.compile(r"\[prefix-host\] demote copy complete off the tick: .*; (\d+) heap payloads \(([0-9.]+)MB\) handed")),
    ("digests_landed", re.compile(r"\[prefix-host\] demote digests landed off the tick: ticket seq=\d+, (\d+) payloads \(([0-9.]+)MB\)")),
    ("capture_submitted", re.compile(r"\[prefix-cache\] capture submitted off the tick \((seed|spec-boundary)\): 64 tokens, (\d+) planes \(([0-9.]+)MB\)")),
    ("capture_draft_note", re.compile(r"\[prefix-cache\] capture submitted off the tick \((spec-boundary)\): .*draft plane (\d+) rows \(([0-9.]+)KB\) in the batch")),
    ("d2d_receipt", re.compile(r"\[prefix-cache\] contracts door D2D capture receipt: .* items=(\d+) bytes=(\d+) ")),
    ("insert", re.compile(r"\[prefix-cache\] insert \((seed|spec-boundary|host-promote)\): 64 tokens, ([0-9.]+)MB")),
    ("evict_preflight", re.compile(r"\[prefix-cache\] evict \(snapshot preflight, LRU\): 64 tokens, ([0-9.]+)MB")),
    ("demote_done", re.compile(r"\[prefix-host\] demote: 64 tokens, ([0-9.]+)MB in ")),
    ("admission_coeff", re.compile(r"\[admission\] \"gate\": plain (\d+) B/token, spec (\d+) B/token")),
    ("loaded", re.compile(r"\[worker\]   loaded \"gate\": (\d+) layers, eos=(\d+)")),
    ("numeric_identity", re.compile(r"\[prefix-host\] contracts door: model gate program identity .* numeric=(\S+) ")),
)


def iv(mb):
    """The byte interval a printed one-decimal MB figure stands for."""
    x = float(mb)
    return (x - 0.05) * 1e6, (x + 0.05) * 1e6


def main(dirs):
    per_dir = {}
    for d in dirs:
        logs = sorted(os.path.join(r, f) for r, _, fs in os.walk(d) for f in fs if f.endswith("server.log"))
        c = Counter()
        census_refusals = 0
        tok_bytes_lines = 0
        for p in logs:
            for line in open(p, errors="replace"):
                if "host byte census does not match snapshot accounting" in line:   # worker.rs:12120-12129
                    census_refusals += 1
                if re.search(r"tok_bytes|k_tok|v_tok", line, re.I):                  # a per-token K or V size
                    tok_bytes_lines += 1
                for name, rx in SHAPES:
                    m = rx.search(line)
                    if m:
                        c[(name,) + m.groups()] += 1
        per_dir[d] = (len(logs), c)
        print(f"DAY38 9B LOGS dir={d} server_logs={len(logs)} census_refusals={census_refusals} tok_bytes_lines={tok_bytes_lines}")
        for k in sorted(c, key=lambda k: (k[0], str(k[1:]))):
            print(f"  DAY38 9B SHAPE dir={d} {k[0]} groups={list(k[1:])} lines={c[k]}")

    # The arithmetic, on the figures the shapes print (plain 64-token class)
    kv = 950272                                   # d2d_receipt items=16 bytes=950272 (seed capture, 8 KV planes)
    kv_spec = 1069056                             # d2d_receipt items=18 bytes=1069056 (spec-boundary, 8 planes + draft)
    plain_bpt, spec_bpt = 14848, 16704            # admission_coeff
    toks = 64
    print(f"DAY38 9B KV plain: d2d bytes={kv} = {toks} tokens x {kv // toks} B/token (exact: {kv % toks == 0}); "
          f"admission plain coefficient {plain_bpt} B/token x {toks} = {plain_bpt * toks} (equal: {plain_bpt * toks == kv}); "
          f"mean per KV plane (K+V) {kv // 8} B = {toks} x {kv // 8 // toks} B/token over 8 planes (per-plane sizes not printed)")
    print(f"DAY38 9B KV draft: spec d2d bytes {kv_spec} - plain {kv} = {kv_spec - kv} = {toks} x {(kv_spec - kv) // toks} B/token; "
          f"admission spec - plain = {spec_bpt - plain_bpt} B/token (equal: {(kv_spec - kv) // toks == spec_bpt - plain_bpt})")
    e_lo, e_hi = iv("54.6")                        # demote_submitted host_bytes
    d_lo, d_hi = iv("53.6")                        # evict_preflight / insert (seed) dead.bytes
    h_lo, h_hi = iv("53.7")                        # copy_complete / digests_landed heap payload bytes
    tok_b = 4 * toks
    # host_bytes = dead.bytes + 4*toks + 4*logits            (worker.rs:8298)
    # dead.bytes = KV + conv + ssm + 4*last_h                (worker.rs:12100-12129, census = snapshot accounting)
    # heap      = conv + ssm + 4*logits + 4*last_h          (worker.rs:9894-9924, 12895-12896; every f32 payload is
    #             Heap when the arena is off: HostF32::down :8267, from_slice :8275, logits :12272, last_h :12275)
    # => host_bytes = KV + heap + 4*toks, identically
    h2_lo, h2_hi = e_lo - kv - tok_b, e_hi - kv - tok_b
    print(f"DAY38 9B HEAP: printed 53.7MB -> [{h_lo:.0f}, {h_hi:.0f}); from host_bytes - KV - 4*toks: [{h2_lo:.0f}, {h2_hi:.0f}); "
          f"both hold: [{max(h_lo, h2_lo):.0f}, {min(h_hi, h2_hi):.0f})")
    lg_lo, lg_hi = e_lo - d_hi - tok_b, e_hi - d_lo - tok_b
    print(f"DAY38 9B LOGITS: 4 x len(last_logits) = host_bytes - dead.bytes - 4*toks in ({lg_lo:.0f}, {lg_hi:.0f}) B "
          f"(printed 54.6MB and 53.6MB; one-decimal rounding only)")
    # All three printed figures at once: heap = host_bytes - KV - 4*toks >= 53.65e6 lifts host_bytes' floor, and
    # 4*logits = host_bytes - dead.bytes - 4*toks inherits it.
    e3_lo = max(e_lo, h_lo + kv + tok_b)
    e3_hi = min(e_hi, h_hi + kv + tok_b)
    print(f"DAY38 9B LOGITS, all three printed figures: host_bytes in [{e3_lo:.0f}, {e3_hi:.0f}) B; "
          f"4 x len(last_logits) in ({e3_lo - d_hi - tok_b:.0f}, {e3_hi - d_lo - tok_b:.0f}) B")
    rh_lo, rh_hi = d_lo - kv, d_hi - kv
    print(f"DAY38 9B RECURRENT+HIDDEN: conv + ssm + 4 x len(last_h) = dead.bytes - KV in [{rh_lo:.0f}, {rh_hi:.0f}) B; "
          f"not separable from these lines")
    # Shares of the host image, worst case each way over the intervals above (E from all three figures)
    print(f"DAY38 9B SHARES of host_bytes: KV {kv / e3_hi * 100:.3f}% to {kv / e3_lo * 100:.3f}%; "
          f"conv + ssm + hidden {rh_lo / e3_hi * 100:.2f}% to {rh_hi / e3_lo * 100:.2f}%; "
          f"logits {(e3_lo - d_hi - tok_b) / e3_hi * 100:.2f}% to {(e3_hi - d_lo - tok_b) / e3_lo * 100:.2f}%")
    print(f"DAY38 9B DRAFT NOTE: (kb + vb) / 1e3 (worker.rs:14801) = {(kv_spec - kv) / 1e3:.3f} KB, printed 118.8KB")
    print(f"DAY38 9B PAYLOADS: 50 heap payloads = Logits (1) + Hidden (1) + Conv/Ssm slots (48); the 48 are not split by class "
          f"on any banked line")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    sys.exit(main(sys.argv[1:]))
