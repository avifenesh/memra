#!/usr/bin/env python3
"""parts.py: per-part GPU kernel time of one served prefill from an nsys sqlite export.

  parts.py prof.sqlite [--gap 2.0] [--json out.json]

The prefill window is the longest GPU-busy segment (kernels separated by idle gaps > GAP s);
boot, calibration and decode fall outside it. Kernel names are classified by substring into
the parts of the Qwen3.8-27B hybrid program."""
import argparse, collections, json, re, sqlite3

PARTS = [
    ("gemm: NVFP4 F8F4 MMQ", r"mul_mat_q_nvfp4_f8f4"),
    ("gemm: NVFP4 F8F4 v2 tile", r"mul_mat_f8f4v2"),
    ("gemm: F8F4 v2 weight image", r"f8f4v2_image"),
    ("gemm: activation e4m3 quantize", r"quantize_mmq_e4m3"),
    ("gemm: other MMQ / matvec", r"mul_mat_q|mmvq|matvec|gemv|qmatvec|cutlass|gemm"),
    ("attention: FA prime kernel", r"fa_prefill|fa_mla"),
    ("attention: KV dequant workspace", r"fa_dequant_kv"),
    ("attention: KV append/quantize", r"append_quantize_kv|append_kv"),
    ("attention: rope/qk-norm/gate glue", r"rope|q_gate_split|sig_mul|rms_norm_qkv"),
    ("gdn: conv/l2/gates", r"ssm_conv|l2_norm|gdn_glog|sigmoid_v|gdn_pad"),
    ("gdn: chunk scan kernels", r"gdn|ssm_scan|delta|wy_|kkt|chunk_state|solve"),
    ("glue: norms/adds/act", r"rms_norm|add|silu|mul|scale|copy|cast|f32_to|bf16_to|f16|convert|fill|memset|embed|get_rows"),
]


def classify(name):
    for part, pat in PARTS:
        if re.search(pat, name):
            return part
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("db")
    ap.add_argument("--gap", type=float, default=2.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    con = sqlite3.connect(a.db)
    rows = con.execute(
        "SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON s.id = COALESCE(k.demangledName, k.shortName) ORDER BY k.start").fetchall()
    try:
        mem = con.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    except sqlite3.OperationalError:
        mem = []
    segs = []
    cur = [rows[0]]
    for r in rows[1:]:
        if (r[0] - cur[-1][1]) / 1e9 > a.gap:
            segs.append(cur); cur = []
        cur.append(r)
    segs.append(cur)
    seg = max(segs, key=lambda s: sum(r[1] - r[0] for r in s))
    t0, t1 = seg[0][0], max(r[1] for r in seg)
    wall = (t1 - t0) / 1e9
    busy = 0; last = t0
    for s, e, _ in seg:
        if e > last:
            busy += e - max(s, last); last = e
    per = collections.defaultdict(lambda: [0, 0])
    part = collections.defaultdict(float)
    for s, e, n in seg:
        short = re.sub(r"\(.*", "", n)
        short = re.sub(r"^void ", "", short)
        per[short][0] += e - s; per[short][1] += 1
        part[classify(short)] += (e - s) / 1e9
    tot = sum(part.values())
    print(f"segments {len(segs)}; window {wall:.3f} s wall, GPU busy {busy/1e9:.3f} s ({100*busy/1e9/wall:.2f}%), "
          f"kernels {len(seg)}, kernel-sum {tot:.3f} s")
    print(f"{'part':40s} {'s':>9s} {'share':>7s}")
    for k, v in sorted(part.items(), key=lambda x: -x[1]):
        print(f"{k:40s} {v:9.3f} {100*v/tot:6.2f}%")
    print("\ntop kernels")
    top = sorted(per.items(), key=lambda x: -x[1][0])[:40]
    for n, (d, c) in top:
        print(f"{d/1e9:9.3f} s {c:7d} x {d/c/1e6:9.3f} ms  [{classify(n)}] {n[:110]}")
    if a.json:
        json.dump({"window_s": wall, "busy_s": busy / 1e9, "kernel_sum_s": tot, "parts_s": part,
                   "top": [{"name": n, "s": d / 1e9, "count": c} for n, (d, c) in top]},
                  open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
