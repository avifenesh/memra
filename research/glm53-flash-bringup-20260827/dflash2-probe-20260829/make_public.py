#!/usr/bin/env python3
"""Produce the public (bankable) twins of the run manifests: transcript-derived text is
reduced to sha16 + a 60-char head (the lane's banked-rows precedent), token ids are
dropped, live build fingerprints are redacted (sev1 boundary pattern)."""
import hashlib
import json
import re
import shutil
import sys

DST = sys.argv[1] if len(sys.argv) > 1 else "."


def redact(s):
    return re.sub(r"memra-[0-9a-f]{12,}", "memra-<redacted-build-fingerprint>", s)


prompts = json.load(open("/root/dfp2/scoring_prompts.json"))
pub = [{k: p[k] for k in ["name", "base", "frac", "cut_chars", "n_ids", "render_sha16"]}
       for p in prompts]
json.dump(pub, open(f"{DST}/scoring-prompts.public.json", "w"), indent=1)

rollouts = json.load(open("/root/dfp2/rollouts.json"))
pub = []
for r in rollouts:
    q = {k: r[k] for k in [
        "name", "prompt_tokens", "n_prompt_ids_sent", "completion_tokens",
        "n_cont_ids_retok", "retok_idempotent", "finish_reason", "elapsed_s",
        "engine_elapsed_s"]}
    q["system_fingerprint"] = redact(r.get("system_fingerprint") or "")
    q["text_sha16"] = hashlib.sha256(r["text"].encode()).hexdigest()[:16]
    q["text_head"] = r["text"][:60]
    pub.append(q)
json.dump(pub, open(f"{DST}/rollouts.public.json", "w"), indent=1)

with open(f"{DST}/captures-manifest.txt", "w") as f:
    f.write("feature captures: run-safetensors teacher-forced over prompt+continuation,\n"
            "MEMRA_TRACE_LAYER_ROWS layers 5,14,24,33,42, stream-mean contracted, f32 LE.\n"
            "argmax line = last-position greedy pick of the capture forward; on finish=stop\n"
            "rollouts it must be an EOS-class id (the token the server stopped on).\n\n")
    import os
    for name in sorted(os.listdir("/root/dfp2/cap")):
        d = f"/root/dfp2/cap/{name}"
        sizes = {x: os.path.getsize(f"{d}/{x}") for x in sorted(os.listdir(d)) if x.endswith(".f32")}
        rows = {x: s // (4096 * 4) for x, s in sizes.items()}
        am = ""
        for line in open(f"{d}/run.log"):
            if "argmax token" in line:
                am = line.strip()
        f.write(f"{name}: rows={set(rows.values())} files={len(sizes)} {redact(am)}\n")

for src in ["summary.json", "decode_rate.json", "loop_check.json",
            "cycles_dflash2.json", "cycles_ngram.json"]:
    shutil.copy(f"/root/dfp2/{src}", f"{DST}/{src}")

print("public twins written to", DST)
