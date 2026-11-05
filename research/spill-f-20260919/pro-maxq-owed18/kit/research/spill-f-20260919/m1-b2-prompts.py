#!/usr/bin/env python3
"""Deterministic B2 prompt set (M1-PREREG.md B2 amendment): 128 distinct prompts, 6,500 words each.

Plain words only (no digits, which some tokenizers split per character), so each prompt stays
well under MEMRA_CTX=8192 tokens. Seeded with random.Random(str), whose string seeding is stable
across CPython 3 releases. Usage: m1-b2-prompts.py --out FILE [--check MANIFEST]
"""
import argparse
import hashlib
import json
import random
import sys

WORDS = """able about above across action active actual adjust admit advance advice affect afford
after again agent agree ahead allow alone along alter amount angle annual answer anyone appear apply
area argue arrive aspect assess assist assume attach attempt attend author avoid award aware balance
basic basis battle beach bear beat become before begin behind belief belong below bench benefit beyond
bitter blade blank blind block board boost border bottom bounce branch brave bread break brief bright
bring broad brush budget build burden button cabin cable camera canal carbon career careful carry
castle casual catch cause ceiling center chain chair chance change channel chapter charge chart check
choice circle civil claim class clean clear client climb clock close cloud coast coffee collect column
combine comfort common company compare complex concern conduct confirm connect consider contact
contain content context control convert copper corner cotton council count couple course cover craft
create credit crowd cruise culture curious current custom cycle damage danger debate decade decide
declare decline define degree deliver demand depend deposit design detail detect device dialog differ
direct discuss display distance divide doctor domain double draft drawer dream drive during eager early
earth easily economy edge editor effect effort eight either elder element eleven emerge employ enable
energy engage engine enjoy enough ensure entire entry equal error escape estate event every exact
example expand expect expert explain export expose extend extra fabric factor fairly family famous
farmer fashion father feature figure filter final finger finish fiscal flavor flight floor follow
forest formal forward frame freeze friend front fruit future garden gather gentle giant glass global
golden govern grade grain grand gravel great green ground group growth guard guess guide habit handle
harbor harvest health heavy height hidden highway history holder honest horizon hunger ignore image
impact import include income indeed index infant inform inland input insect inside insist install
intend invest island itself jacket journal judge junior kernel kitchen ladder landing language laptop
launch lawyer layer leader league legacy lemon lesson letter level license light limit linear liquid
listen little lively locate lodge logic lonely lumber magnet manage manner marble margin market master
matter meadow measure medium member memory mental method middle minute mirror mobile modern modest
moment motion mountain muscle mutual narrow native nature nearby needle neither network neutral normal
notice novel number object obtain office often online option orange orbit order origin output owner
oxygen packet paddle palace parcel parent partner patrol pattern pencil people pepper period permit
person phrase pilot planet plastic player pocket poetry policy polite portal powder praise prefer
prepare present prison private process produce profit prompt proper protect prove public purple
puzzle quarter quick quiet rabbit radar random rapid rather reason recall record reduce reform region
relax remain remote repair report rescue resist resort result retail return reveal review reward
rhythm ribbon ridge river robust rocket roster rotate rough route rubber runway saddle safety salmon
sample saving scale scheme school screen season second secret select senior sensor series session
settle shadow shallow shelter shield signal silent silver simple single sister sketch slender smooth
socket solid source spirit spring square stable status steady stream street strong studio submit
summer supply surface survey switch symbol system table talent target teacher temple tender theory
thread timber tissue toggle topic tower track trade travel treaty tribute tunnel update upper useful
valley vector velvet vendor verify vessel victory village violet vision visual volume voyage wagon
wallet wander warmth weather weekly welcome window winter wisdom wonder worker yellow yield zone""".split()


def generate():
    lines = []
    for i in range(128):
        rng = random.Random(f"m1-b2-prompt-{i}")
        head = f"Review section {WORDS[i % len(WORDS)]} {WORDS[(7 * i + 3) % len(WORDS)]} {i // 26 * ' again'}".strip()
        body = " ".join(rng.choice(WORDS) for _ in range(6500))
        lines.append(json.dumps({"id": i, "prompt": f"{head}. {body}."}) + "\n")
    return "".join(lines).encode()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--check")
    a = ap.parse_args()
    data = generate()
    with open(a.out, "xb") as f:
        f.write(data)
    digest = hashlib.sha256(data).hexdigest()
    print(json.dumps({"prompts": 128, "words_per_prompt": 6500, "bytes": len(data), "sha256": digest}))
    if a.check:
        want = json.load(open(a.check))["sha256"]
        if want != digest:
            print(f"B2 PROMPTS MISMATCH: {digest} != {want}", file=sys.stderr)
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
