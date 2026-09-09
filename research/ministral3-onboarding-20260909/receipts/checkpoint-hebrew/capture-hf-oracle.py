#!/usr/bin/env python3
import argparse
import struct
import torch
import transformers
from transformers import AutoModelForCausalLM

MODEL = "/workspace/ministral/text-fp8"
REVISION = None
TOKENS = [1, 2, 3, 4]

parser = argparse.ArgumentParser(description="Offline HF correctness oracle for Memra onboarding")
parser.add_argument("--out", default="hf-oracle.tsv")
args = parser.parse_args()

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    revision=REVISION,
    dtype=torch.float32,
    trust_remote_code=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()
with torch.no_grad():
    logits = model(input_ids=torch.tensor([TOKENS], device=device)).logits[0, -1].float().cpu()

with open(args.out, "w", encoding="utf-8") as f:
    f.write("format\tmemra-checkpoint-oracle-v1\n")
    f.write("engine\thf-transformers-fp32\n")
    f.write("numeric_class\tsource-weights-float32-accumulation\n")
    f.write(f"transformers_version\t{transformers.__version__}\n")
    f.write(f"torch_version\t{torch.__version__}\n")
    f.write("tokens\t" + ",".join(map(str, TOKENS)) + "\n")
    f.write(f"vocab\t{logits.numel()}\n")
    for index, value in enumerate(logits.tolist()):
        bits = struct.unpack("<I", struct.pack("<f", value))[0]
        f.write(f"logit\t{index}\t{bits:08x}\n")
