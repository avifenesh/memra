"""Convert a DeepSeek-V4.1 vendor-oracle case (`<id>.pt` from the CPU oracle run) into the tape
`memra model verify checkpoint --oracle` reads.

The tape (`memra-checkpoint-oracle-v1`, parsed by `parse_checkpoint_oracle` in crates/memra-cli/src/lib.rs) is one
case: tab-separated `key\\tvalue` lines `format`, `engine`, `numeric_class`, `tokens` (comma-separated prompt ids),
`vocab`, then one `logit\\t<index>\\t<f32 bits as 8 hex digits>` line for every vocabulary index, the full logits row
after the prompt's last position. Other lines are ignored. The gate compares it against the native runner's tape
(`<runner> <checkpoint_dir> 1 2 3 4` with `MEMRA_ORACLE_OUT=<tsv>` and `MEMRA_FULL_PREC=1`): `numeric_class` must
be equal, `tokens` and `vocab` equal, every logit within the pack's `atol + rtol * |reference|`, and the argmax equal
when the pack requires it. `verify checkpoint` always runs the native side on tokens 1,2,3,4, so only a case whose
`prompt_ids` are exactly [1, 2, 3, 4] can pass it; any other case converts for offline comparison and is flagged.

From the case record this uses `prompt_ids` and `full_logits[0]` (the logits after the prefill, step 0). The greedy
`gen_ids`, the per-step top-32 logits, the later full-logits steps and the prefill hidden states have no consumer in
the tape format.

usage: uv run --with torch python3 -I oracle_to_checkpoint_tape.py CASE.pt OUT.tsv
           [--numeric-class deepseek-v41-bf16-correctly-rounded] [--engine NAME]
The numeric class names the program both sides ran: the bf16-activation checkpoint program with correctly rounded
primitives (exact.py on the oracle side, memra_gguf::dsv41_forward on the native side). An oracle run without
exact.py and the CUDA top-k rule is a different numeric class; say so with --numeric-class.
"""
import argparse
import hashlib
import struct
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument("case")
ap.add_argument("out")
ap.add_argument("--numeric-class", default="deepseek-v41-bf16-correctly-rounded")
ap.add_argument("--engine", default="deepseek-v41-vendor-model.py-cpu")
a = ap.parse_args()

rec = torch.load(a.case, map_location="cpu", weights_only=True)
prompt = [int(t) for t in rec["prompt_ids"]]
full = rec.get("full_logits")
if full is None or full.dim() != 2 or full.size(0) < 1:
    sys.exit(f"{a.case}: no full logits for step 0; rerun the oracle with --full-logits >= 1")
logits = full[0].float()
if not bool(torch.isfinite(logits).all()):
    sys.exit(f"{a.case}: non-finite logits")
with open(a.out, "w") as out:
    out.write("format\tmemra-checkpoint-oracle-v1\n")
    out.write(f"engine\t{a.engine}\n")
    out.write(f"numeric_class\t{a.numeric_class}\n")
    out.write(f"oracle_case\t{rec['id']}\n")
    out.write(f"oracle_case_sha256\t{hashlib.sha256(open(a.case, 'rb').read()).hexdigest()}\n")
    out.write("tokens\t" + ",".join(map(str, prompt)) + "\n")
    out.write(f"vocab\t{logits.numel()}\n")
    for index, value in enumerate(logits.tolist()):
        out.write(f"logit\t{index}\t{struct.unpack('<I', struct.pack('<f', value))[0]:08x}\n")
if prompt != [1, 2, 3, 4]:
    print(f"note: prompt_ids {prompt[:8]}... are not [1, 2, 3, 4]; memra model verify checkpoint runs its native side "
          "on tokens 1,2,3,4 and will refuse this tape as an identity mismatch", file=sys.stderr)
print(f"wrote {a.out}: {logits.numel()} logits, argmax {int(logits.argmax())}")
