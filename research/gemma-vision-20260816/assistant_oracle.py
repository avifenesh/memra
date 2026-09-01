#!/usr/bin/env python3
"""Reference oracle for the Gemma-4-31B official MTP assistant draft head.

Purpose: produce the ground-truth draft logits/tokens that a future memra
Gemma4Assistant arm must match — the same build-the-reference-first discipline the
vision tower lane used (numpy/HF reference gated the CUDA to 1.0 cosine).

This is NOT the standalone-runnable part of this fork's deliverable: the assistant's
forward REQUIRES `shared_kv_states` from a paired backbone forward (google/gemma-4-31b-it),
so the true oracle needs BOTH checkpoints resident — run on the Japan box (62GB backbone +
939MB assistant), not in a fork. What this file pins is the EXACT call shape and the two
things a memra kernel must reproduce bit-for-bit: the shared-KV attention inputs and the
centroid-masked decode.

Usage (on a box with transformers >= the gemma4_assistant release + both checkpoints):
    python assistant_oracle.py --backbone google/gemma-4-31b-it \
        --assistant google/gemma-4-31B-it-assistant \
        --prompt "Explain binary search." --k 3 --out oracle.json
"""
import argparse, json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--assistant", required=True)
    ap.add_argument("--prompt", default="Explain binary search, including edge cases.")
    ap.add_argument("--k", type=int, default=3, help="draft depth (num_speculative_tokens)")
    ap.add_argument("--out", default="assistant-oracle.json")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, Gemma4ForCausalLM, Gemma4AssistantForCausalLM

    tok = AutoTokenizer.from_pretrained(args.backbone)
    backbone = Gemma4ForCausalLM.from_pretrained(args.backbone, dtype=torch.bfloat16, device_map="cuda:0")
    assistant = Gemma4AssistantForCausalLM.from_pretrained(args.assistant, dtype=torch.bfloat16, device_map="cuda:0")
    backbone.eval(); assistant.eval()

    ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        # 1) Backbone forward with cache — this is the KV the assistant shares.
        bout = backbone(ids, use_cache=True, output_hidden_states=True)
        # shared_kv_states: per layer_type, the LAST layer's (K, V) of that type.
        # The exact extraction dict is what HF's assisted path passes to the assistant;
        # this harness pins its shape — see ASSISTANT-ARM-SPEC.md primitive #1.
        # 2) The pre_projection input concat (2 x backbone_hidden). Its exact construction
        #    is the ONE unpinned glue detail (candidate_generator.py) — capture what HF
        #    actually feeds by hooking assistant.pre_projection's input here.
        captured = {}

        def hook(_m, inp, _out):
            captured["pre_proj_in"] = inp[0].detach().float().cpu()

        h = assistant.pre_projection.register_forward_hook(hook)
        # NOTE: the real call goes through backbone.generate(assistant_model=assistant, ...);
        # this scaffold documents the seam. Fill the generate() call per the installed
        # transformers version, then dump:
        h.remove()

    # Emit the oracle a memra kernel must match: per draft step, the full-vocab logit row
    # (post centroid-mask), the argmax draft token, and the pre_projection input tensor
    # (so the concat construction is captured, not guessed).
    out = {
        "prompt": args.prompt,
        "k": args.k,
        "note": "fill generate() for the installed transformers version; see spec primitive #2",
        "captured_keys": list(captured.keys()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
