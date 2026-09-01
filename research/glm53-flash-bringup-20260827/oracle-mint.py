#!/usr/bin/env python3
# Pinned-oracle mint for GLM-5.3-Flash (zai-org/GLM-5.3-Flash @ 04c4e9e95c5da8862dced7e5056455116f83a7e0).
# External implementation (transformers) used OFF-SERVING to create pinned oracle
# evidence only, per the darklanes engine-consumption law. Greedy (the instrument),
# short real prompts, banked per-step argmax ids + top-8 logits + final bytes.
# Runs on 2x96GB + host offload (FP8 weights, 328 GB -> device_map auto + offload).
import json, sys, time, pathlib, hashlib
import torch
from transformers import AutoTokenizer, Glm5NextForConditionalGeneration

REV = "04c4e9e95c5da8862dced7e5056455116f83a7e0"
MODEL_DIR = pathlib.Path.home() / "models/glm53-flash"
OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "oracle-bank.jsonl")
MAX_NEW = int(sys.argv[2]) if len(sys.argv) > 2 else 64

# Real prompts (LAW: real prompts, never synthetic). Short, fixed, diverse shapes.
PROMPTS = [
    ("greet",   [{"role": "user", "content": "Hello! What model are you?"}]),
    ("code",    [{"role": "user", "content": "Write a Python function that reverses a linked list, with a docstring."}]),
    ("reason",  [{"role": "user", "content": "A train leaves at 3pm going 60 km/h; another at 4pm going 90 km/h on the same track from the same station. When does the second catch the first?"}]),
    ("tooluse", [{"role": "user", "content": "List the files in the current directory and explain what you would do next."}]),
]
EFFORTS = ["max", "low"]  # thinking default is max; low exercises the effort switch

tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), revision=None, trust_remote_code=False)
model = Glm5NextForConditionalGeneration.from_pretrained(
    str(MODEL_DIR), dtype="auto", device_map="auto",
    max_memory={0: "82GiB", 1: "82GiB", "cpu": "300GiB"},
    offload_folder=str(pathlib.Path.home()/"offload"),
)
model.eval()

rows = 0
with open(OUT, "a") as f:
    for effort in EFFORTS:
        for name, messages in PROMPTS:
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                reasoning_effort=effort,
            )
            ids = tok(text, return_tensors="pt").input_ids
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(
                    ids.to(model.device), max_new_tokens=MAX_NEW, do_sample=False,
                    output_scores=True, return_dict_in_generate=True,
                )
            gen = out.sequences[0, ids.shape[1]:]
            steps = []
            for i, sc in enumerate(out.scores):
                top = torch.topk(sc[0].float(), 8)
                steps.append({"argmax": int(sc[0].argmax()),
                              "top8_ids": top.indices.tolist(),
                              "top8_logits": [round(v, 4) for v in top.values.tolist()]})
            completion = tok.decode(gen, skip_special_tokens=False)
            row = {
                "rev": REV, "prompt": name, "effort": effort,
                "prompt_sha16": hashlib.sha256(text.encode()).hexdigest()[:16],
                "prompt_token_ids": ids[0].tolist(),
                "gen_token_ids": gen.tolist(),
                "completion_bytes_sha": hashlib.sha256(completion.encode()).hexdigest(),
                "completion_head": completion[:120],
                "steps": steps, "elapsed_s": round(time.time() - t0, 1),
                "torch": torch.__version__,
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            rows += 1
            print(f"[oracle] {name}/{effort}: {len(gen)} tok in {row['elapsed_s']}s", flush=True)
print(f"ORACLE-MINT-DONE rows={rows}", flush=True)
