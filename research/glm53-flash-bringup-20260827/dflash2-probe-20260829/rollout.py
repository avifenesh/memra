#!/usr/bin/env python3
"""Target greedy rollouts on the SERVING binary via the raw /v1/completions surface.

Greedy is the instrument (byte-determinism makes the DFlash2 acceptance rule an exactness
check), never the product. prompt_ids in, text out; the continuation is retokenized with
the artifact tokenizer to get the canonical target-path ids (the engine has no token-id
echo on this surface). A retokenization idempotence check rides along; positions are
scored in this canonical token space and any residual segmentation noise is stated in the
receipt, not hidden.

Env: MEMRA_PREFIX_CACHE_MB=0 pinned at server boot (defence in depth for the glm5
restore defect and to keep every rollout a cold prefill).
"""
import json
import time
import urllib.request

from tokenizers import Tokenizer

EP = "http://127.0.0.1:18402/v1/completions"
MODEL = "zai/glm-5.3-flash"
MAX_TOKENS = 256
ART = "/root/models/glm53-nvfp4"

tok = Tokenizer.from_file(f"{ART}/tokenizer.json")
prompts = json.load(open("/root/dfp2/scoring_prompts.json"))
rows = []

for p in prompts:
    body = {
        "model": MODEL,
        "prompt_ids": p["ids"],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.0,
    }
    t0 = time.time()
    req = urllib.request.Request(
        EP, data=json.dumps(body).encode(), headers={"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        j = json.loads(r.read())
    el = time.time() - t0
    ch = j["choices"][0]
    text = ch["text"]
    cont_ids = tok.encode(text, add_special_tokens=False).ids
    # Idempotence: decode(encode(text)) must reproduce text or the canonical space is lossy.
    rt = tok.decode(cont_ids, skip_special_tokens=False)
    row = {
        "name": p["name"],
        "prompt_tokens": j["usage"]["prompt_tokens"],
        "n_prompt_ids_sent": p["n_ids"],
        "completion_tokens": j["usage"]["completion_tokens"],
        "n_cont_ids_retok": len(cont_ids),
        "retok_idempotent": rt == text,
        "finish_reason": ch["finish_reason"],
        "elapsed_s": round(el, 2),
        "engine_elapsed_s": j["usage"].get("elapsed_s"),
        "system_fingerprint": j.get("system_fingerprint"),
        "text": text,
        "cont_ids": cont_ids,
    }
    rows.append(row)
    print(
        f'{row["name"]}: pt={row["prompt_tokens"]} ct={row["completion_tokens"]} '
        f'retok={row["n_cont_ids_retok"]} idem={row["retok_idempotent"]} '
        f'finish={row["finish_reason"]} {row["elapsed_s"]}s'
    )

json.dump(rows, open("/root/dfp2/rollouts.json", "w"))
mism = [r["name"] for r in rows if r["prompt_tokens"] != r["n_prompt_ids_sent"]]
print(f"prompt_tokens mismatches: {mism or 'none'}")
noidem = [r["name"] for r in rows if not r["retok_idempotent"]]
print(f"retok non-idempotent: {noidem or 'none'}")
drift = [(r["name"], r["completion_tokens"], r["n_cont_ids_retok"]) for r in rows
         if r["completion_tokens"] != r["n_cont_ids_retok"]]
print(f"generated-vs-retokenized length drift: {drift or 'none'}")
