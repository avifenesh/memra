"""The decode tables of README.md from decode_gate.py's receipts.

usage: tables_decode.py RECEIPT.json [RECEIPT.json ...] > decode-tables.md"""
import json
import sys


def e(x):
    return "-" if x is None else f"{x:.2e}"


def pct(x):
    return f"{100 * x:.1f}%"


rs = [json.load(open(p)) for p in sys.argv[1:]]
print("Teacher-forced, pooled per group (KL lower bound from the shared top-32 plus a rest bucket):\n")
print("| group | subset | steps | KL memra | KL floor | ratio | top-1 memra | top-1 floor | band | pass |")
print("|---|---|---|---|---|---|---|---|---|---|")
for r in rs:
    for name, s in r["pooled"].items():
        if s["steps"] == 0:
            continue
        band = f"KL <= {e(2 * s['kl_floor_mean'])}, top-1 >= {pct(s['top1_floor'] - s['top1_slack'])}"
        print(f"| {r['name']} | {name} | {s['steps']} | {e(s['kl_native_mean'])} | {e(s['kl_floor_mean'])} | "
              f"{s['kl_ratio']:.2f} | {pct(s['top1_native'])} | {pct(s['top1_floor'])} | {band} | "
              f"{'PASS' if s['pass'] else 'FAIL'} |")
print("\nFull-logit steps (exact KL against the tape's full logits):\n")
print("| group | steps | KL memra mean / max | KL floor mean / max |")
print("|---|---|---|---|")
for r in rs:
    f = r["full_logit_steps"]
    print(f"| {r['name']} | {f['steps']} | {e(f['kl_native_mean'])} / {e(f['kl_native_max'])} | "
          f"{e(f['kl_floor_mean'])} / {e(f['kl_floor_max'])} |")
print("\nPer case (teacher-forced):\n")
print("| case | prompt | steps | ref = tape | KL memra mean / max | KL floor mean / max | top-1 memra | top-1 floor | "
      "first miss memra / floor |")
print("|---|---|---|---|---|---|---|---|---|")
for r in rs:
    for p in r["per_case"]:
        print(f"| {p['case']} | {p['prompt']:,} | {p['steps']} | {'yes' if p['ref_reproduces_tape'] else 'NO'} | "
              f"{e(p['kl_native_mean'])} / {e(p['kl_native_max'])} | {e(p['kl_floor_mean'])} / {e(p['kl_floor_max'])} | "
              f"{p['top1_native']}/{p['steps']} | {p['top1_floor']}/{p['steps']} | "
              f"{p['first_native_miss'] if p['first_native_miss'] is not None else '-'} / "
              f"{p['first_floor_miss'] if p['first_floor_miss'] is not None else '-'} |")
gs = [g for r in rs for g in r["greedy"]]
if gs:
    print("\nGreedy generation against the vendor's tokens:\n")
    print("| case | vendor steps | memra steps | exact prefix | parts at | vendor margin (logit) | vendor p1 / p2 | "
          "memra's token, vendor rank | floor picks | CPU picks | near-tie | prefix steps = forced run |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rs:
        for g in r["greedy"]:
            if "diverge_step" in g:
                fl = "memra's" if g["floor_picks_native"] else ("vendor's" if g["floor_matches_vendor"] else "another")
                cp = "-" if g.get("cpu_token") is None else ("memra's" if g["cpu_picks_native"] else (
                    "vendor's" if g["cpu_token"] == g["vendor_token"] else "another"))
                row = (f"{g['diverge_step']} | {g['vendor_margin_logit']:.3f} | {g['vendor_p1']:.3f} / {g['vendor_p2']:.3f} | "
                       f"{g['native_token']}, #{g['native_token_vendor_rank']} | {fl} | {cp} | "
                       f"{'yes' if g['near_tie'] else 'no'}")
            else:
                row = "- | - | - | - | - | - | -"
            same = (f"{g['forced_vs_greedy_identical_steps']} steps identical" if g["forced_vs_greedy_identical"]
                    else "DIFFER")
            print(f"| {g['case']} | {g['vendor_steps']} | {g['native_steps']} | {g['exact_prefix']} | {row} | {same} |")
        m = r.get("vendor_margin_move")
        if m:
            print(f"\n{r['name']}: the vendor's top-2 margin moves between its two realizations by {m['p50']:.3f} logits at "
                  f"the median step, {m['p99']:.3f} at p99 and {m['max']:.3f} at most ({m['steps']} steps).\n")
