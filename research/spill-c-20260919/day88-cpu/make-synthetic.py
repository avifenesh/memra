#!/usr/bin/env python3
"""DAY88: a synthetic receipts root (promo/, promo-res/, promo-spec/) for the reader's mechanics dry check, from
DAY85's BOX41 `i22` receipts: the door runs (`i22`) stand for the four door arms and REF's runs (`ref`) for the two
legacy arms, each copy given the lines its arm prints under the promoted binary (the qualification, the pool kind,
the prefetch's default and effective state, the rollback's `off` line); the resident cell reuses REF's runs with the
resident `off` line; the spec cell is two stand-in logs. The reading means nothing.
usage: make-synthetic.py <day85-i22-cell-dir> <out-root>
"""
import re
import shutil
import sys
from pathlib import Path

src, out = Path(sys.argv[1]) / "ev", Path(sys.argv[2])
POOL = re.compile(r"(host pinned pool bytes=.* fill_reserve=\d+)$")
LINES = {
    "q": "[experts-via-tier] qualified sha256=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf",
    "pf": "[moe-prefetch] default=on (a qualified artifact) effective=on",
    "pf0": "[moe-prefetch] default=on (a qualified artifact) effective=off (MEMRA_MOE_PREFETCH=0)",
    "off": "[experts-via-tier] off: MEMRA_EXPERTS_VIA_TIER=0 (the legacy slot cache)",
    "res": "[experts-via-tier] off: the experts are resident",
}
ARMS = {  # arm: (source arm, lines added, pool suffix)
    "naked": ("i22", ["q", "pf"], " registered"),
    "q22": ("i22", [], ""),
    "alloc": ("i22", ["q", "pf"], " allocated"),
    "nopf": ("i22", ["q", "pf0"], " registered"),
    "legacy": ("ref", ["off", "pf"], None),
    "legnopf": ("ref", ["off", "pf0"], None),
}


def copy(cell, arm, source, lines, suffix):
    ev = out / cell / "ev"
    ev.mkdir(parents=True, exist_ok=True)
    for log in sorted(src.glob(f"o[12]-{source}-r[1-5].log")):
        order, _, rep = log.stem.split("-")
        text = log.read_text(errors="replace").splitlines()
        stamp = text[0].partition("\t")[0]
        body = [f"{stamp}\t{LINES[k]}" for k in lines]
        for line in text:
            ts, _, rest = line.partition("\t")
            if suffix is not None and POOL.search(rest):
                rest = POOL.sub(lambda m: m.group(1) + suffix, rest)
            body.append(f"{ts}\t{rest}")
        label = f"{order}-{arm}-{rep}"
        (ev / f"{label}.log").write_text("\n".join(body) + "\n")
        shutil.copy(src / f"{log.stem}.exit", ev / f"{label}.exit")
    marks = [line.replace(f"-{source}-", f"-{arm}-") for line in (src / "marks.tsv").read_text().splitlines()
             if f"-{source}-" in line]
    with open(ev / "marks.tsv", "a") as fh:
        fh.write("\n".join(marks) + "\n")


for arm, (source, lines, suffix) in ARMS.items():
    copy("promo", arm, source, lines, suffix)
copy("promo-res", "naked", "ref", ["q", "pf", "res"], None)
copy("promo-res", "legacy", "ref", ["off", "pf"], None)
ev = out / "promo-spec" / "ev"
ev.mkdir(parents=True, exist_ok=True)
installed = "[experts-via-tier] installed artifact_sha256=df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf"
(ev / "spec-naked.log").write_text(f"00:00:00.000\t{installed}\n00:00:01.000\t=== SELF-CONSISTENCY PASS ===\n")
(ev / "spec-legacy.log").write_text(f"00:00:00.000\t{LINES['off']}\n00:00:01.000\t=== SELF-CONSISTENCY PASS ===\n")
(ev / "spec-naked.exit").write_text("0\n")
(ev / "spec-legacy.exit").write_text("0\n")
print(f"synthetic root {out}")
