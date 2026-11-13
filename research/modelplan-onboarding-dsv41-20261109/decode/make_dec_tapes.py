"""Tapes for the decode byte gate's capture: gsm (42-token prompt, decode at 42..47) and straddle126 (the story
prompt plus the vendor's first 104 story tokens as a 126-token prompt, decode at 126..131: below, at and past the
window). Only prompt_ids and gen_ids, which capture_layers.py reads."""
import os
import sys

import torch

tapes, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
g = torch.load(os.path.join(tapes, "gsm.pt"), weights_only=False, map_location="cpu")
torch.save({"prompt_ids": list(g["prompt_ids"]), "gen_ids": [int(x) for x in g["gen_ids"]]}, os.path.join(out, "gsm.pt"))
s = torch.load(os.path.join(tapes, "story.pt"), weights_only=False, map_location="cpu")
gen = [int(x) for x in s["gen_ids"]]
torch.save({"prompt_ids": list(s["prompt_ids"]) + gen[:104], "gen_ids": gen[104:]}, os.path.join(out, "straddle126.pt"))
print("gsm", len(g["prompt_ids"]), "straddle126", len(s["prompt_ids"]) + 104)
