#!/usr/bin/env python3
"""Write docs/impact-map/{findings.md,findings.json,subsystems/*.md} from the annotation results,
the scout findings and the skeptic verdicts. Findings whose quote does not match the tree are
dropped; refuted defects are dropped; unverified ones are kept and marked.

Usage: report.py RESULTS.json DEFECT_VERDICTS.json OUT_DIR
"""
import collections
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
import ingest  # noqa: E402

SEV = {"high": 0, "medium": 1, "low": 2}
CAT_TITLE = {"error": "Errors (real bugs)", "broken_gate": "Broken gates", "dead_code": "Dead code",
             "unreachable": "Unreachable paths", "stale_flag": "Stale flags and doors", "stale_doc": "Stale docs",
             "decoupling": "Decoupling needed", "mergeable": "Mergeable", "reusable": "Reusable"}
ORDER = ["error", "broken_gate", "dead_code", "unreachable", "stale_flag", "stale_doc", "decoupling", "mergeable", "reusable"]
DEFECTS = {"error", "broken_gate", "dead_code", "unreachable", "stale_flag", "stale_doc"}
SLICE_TITLES = {}


def norm_cat(c):
    c = (c or "").lower().replace("-", "_")
    return {"correctness": "error", "bug": "error", "policy": "error", "build": "error", "dead": "dead_code",
            "stale": "stale_doc", "doc": "stale_doc", "flag": "stale_flag"}.get(c, c if c in CAT_TITLE else "error")


def main():
    res = json.load(open(sys.argv[1]))
    verdicts = {}
    if os.path.exists(sys.argv[2]):
        for v in json.load(open(sys.argv[2])):
            verdicts[v["id"]] = v
    out_dir = sys.argv[3]
    os.makedirs(os.path.join(out_dir, "subsystems"), exist_ok=True)
    findings = []
    for a in res["annotated"]:
        if not a or a.get("failed"):
            continue
        for i, f in enumerate(a.get("findings", [])):
            findings.append(dict(f, id=f"{a['slice']}:F{i}", slice=a["slice"]))
    for key in ("packs", "routes", "devices"):
        for i, f in enumerate((res["scouts"].get(key) or {}).get("findings", [])):
            findings.append(dict(f, id=f"scout-{key}:F{i}", slice=f"scout:{key}", detail=f.get("detail", ""), fix=f.get("fix", "")))
    own = os.environ.get("MEMRA_IMPACT_OWN_FINDINGS")
    prechecked = json.load(open(own)) if own and os.path.exists(own) else []
    kept, dropped = list(prechecked), collections.Counter()
    for f in findings:
        f["category"] = norm_cat(f.get("category"))
        ev = ingest.pin({"file": f.get("file"), "line": f.get("line"), "quote": f.get("quote")})
        if not ev:
            dropped["bad_quote"] += 1
            continue
        f["file"], f["line"] = ev["file"], ev["line"]
        v = verdicts.get(f["id"])
        if v:
            if v["verdict"] == "refuted":
                dropped["refuted"] += 1
                continue
            f["verification"] = v["verdict"]
            f["severity"] = v.get("severity") or f["severity"]
            if v.get("fix"):
                f["fix"] = v["fix"]
            f["verify_reason"] = v.get("reason", "")
        else:
            f["verification"] = "not verified" if f["category"] in DEFECTS else "evidence-checked"
        kept.append(f)
    # dedup same file:line + category
    seen, uniq = set(), []
    for f in sorted(kept, key=lambda f: (ORDER.index(f["category"]), SEV.get(f["severity"], 3), f["file"], f["line"])):
        k = (f["category"], f["file"], f["line"])
        if k in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(k)
        uniq.append(f)
    json.dump(uniq, open(os.path.join(out_dir, "findings.json"), "w"), indent=1)
    by = collections.defaultdict(list)
    for f in uniq:
        by[f["category"]].append(f)
    lines = ["# Findings", "",
             "Marked while mapping the repo (owner's ask: mergeable, reusable, decoupling needed, errors). Nothing here is fixed by this change.",
             "Defects (errors, broken gates, dead code, unreachable paths, stale flags and docs) were checked by an independent skeptic agent told to refute them; refuted ones are not listed. `verification` says what happened to each.",
             "Every row's quote was checked against the tree at the map commit.", "",
             "| category | count | high | medium | low |", "|---|---|---|---|---|"]
    for c in ORDER:
        fs = by.get(c, [])
        if fs:
            n = collections.Counter(f["severity"] for f in fs)
            lines.append(f"| {CAT_TITLE[c]} | {len(fs)} | {n.get('high', 0)} | {n.get('medium', 0)} | {n.get('low', 0)} |")
    lines.append("")
    for c in ORDER:
        fs = by.get(c, [])
        if not fs:
            continue
        lines += [f"## {CAT_TITLE[c]}", ""]
        for f in fs:
            title = re.sub(r"\s+", " ", f.get("title", "")).strip()
            lines.append(f"### {f['severity']}: {title}")
            lines.append("")
            lines.append(f"- where: `{f['file']}:{f['line']}` ({f['slice']}, {f['verification']})")
            if f.get("detail"):
                lines.append(f"- detail: {re.sub(r'\s+', ' ', f['detail']).strip()}")
            if f.get("impact_cut"):
                lines.append(f"- cut: {re.sub(r'\s+', ' ', f['impact_cut']).strip()}")
            if f.get("verify_reason") and f["verification"] in ("confirmed", "unsure"):
                lines.append(f"- check: {re.sub(r'\s+', ' ', f['verify_reason']).strip()}")
            lines.append(f"- fix: {re.sub(r'\s+', ' ', f.get('fix', '')).strip()}")
            lines.append("")
    # mechanical candidates (stage 0) and the annotators' verdicts on them
    cv = collections.Counter()
    confirmed = collections.defaultdict(list)
    for a in res["annotated"]:
        if not a or a.get("failed"):
            continue
        for v in a.get("candidate_verdicts", []):
            if v.get("kind") == "coverage":
                continue
            cv[(v.get("kind"), v.get("verdict"))] += 1
            if v.get("verdict") == "confirmed" and v.get("kind") in ("dead_candidate", "exact_duplicate", "near_duplicate", "stale_flag", "big_fn"):
                confirmed[v["kind"]].append(v)
    lines += ["## Mechanical candidates", "",
              "Stage 0 listed candidates by script (items with no caller in the graph, identical or near-identical bodies, functions of 400+ lines, flags past their decide-by date, files with many device checks). The slice annotators judged them; these verdicts were not re-checked by a skeptic.", "",
              "| candidate kind | confirmed | rejected | unsure |", "|---|---|---|---|"]
    for kind in sorted({k for k, _ in cv}):
        lines.append(f"| {kind} | {cv[(kind, 'confirmed')]} | {cv[(kind, 'rejected')]} | {cv[(kind, 'unsure')]} |")
    lines.append("")
    titles = {"dead_candidate": "No caller (confirmed dead)", "exact_duplicate": "Identical bodies", "near_duplicate": "Near-identical bodies",
              "stale_flag": "Doors past their decide-by date", "big_fn": "Functions of 400+ lines (decoupling candidates)"}
    for kind, vs in confirmed.items():
        lines += [f"### {titles[kind]} ({len(vs)})", ""]
        for v in sorted(vs, key=lambda v: (v["file"], v["line"])):
            lines.append(f"- `{v['file']}:{v['line']}`: {re.sub(r'\s+', ' ', v.get('note', '')).strip()[:300]}")
        lines.append("")
    open(os.path.join(out_dir, "findings.md"), "w").write("\n".join(lines).replace("\u2014", "-") + "\n")
    for a in res["annotated"]:
        if not a or a.get("failed") or not a.get("writeup_md"):
            continue
        text = a["writeup_md"].replace("\u2014", "-").strip()
        open(os.path.join(out_dir, "subsystems", f"{a['slice']}.md"), "w").write(text + "\n")
    print(json.dumps({"findings": len(uniq), "by_category": {c: len(v) for c, v in by.items()}, "dropped": dropped}))


if __name__ == "__main__":
    main()
