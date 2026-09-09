#!/usr/bin/env python3
"""GPTQ-NVFP4 mint of DictaLM-3.0-24B via llm-compressor.

Why GPTQ and not more AWQ: AWQ picks per-channel scales from activation magnitudes; GPTQ
reconstructs each layer's weights against its own Hessian, compensating the error it has
already made column by column. It is the method DictaLM's own authors used for their W4A16
release (`actorder: static` in their config), and it is the natural next arm after the lane
established that the mint — not the engine — is what costs Hebrew accuracy.

Calibration is HEBREW-LED for the same reason as the AWQ arm: this checkpoint exists to serve
Hebrew, and calibrating on English optimizes the distribution it is not judged on.
Global-MMLU is never touched.

Two shapes to watch in the output, because either would need engine support the way AWQ's
pre_quant_scale did:
  * `weight_g_idx` — the actorder column permutation. Emitted when actorder is enabled;
    this recipe leaves it OFF for that reason (group-16 NVFP4 gains little from it).
  * `weight_scale` / `weight_global_scale` — the NVFP4 two-level scale, which memra reads.
"""
import argparse, os, random

ap = argparse.ArgumentParser()
ap.add_argument("--src", default="/root/models/dictalm3-24b-bf16")
ap.add_argument("--out", required=True)
ap.add_argument("--samples", type=int, default=128)
ap.add_argument("--seqlen", type=int, default=512)
ap.add_argument("--scheme", default="NVFP4A16")
# The reduction projections (down_proj, o_proj) are where NVFP4 cost this checkpoint its
# Hebrew: the lane's family ablation put the damage there and nowhere else. GPTQ's error
# compensation may or may not make the carve-out redundant; --keep-fp8 is the arm that
# answers it. Empty string = pure NVFP4 (the arm already measured).
ap.add_argument("--keep-fp8", default="")
# A different calibration draw is the NOISE FLOOR for any family ablation: without it a
# one-point "recovery" cannot be told from mint-to-mint jitter.
ap.add_argument("--seed", type=int, default=20260905)
args = ap.parse_args()

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier, QuantizationModifier
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)

print("[gptq] loading BF16 source", flush=True)
tok = AutoTokenizer.from_pretrained(args.src)
model = AutoModelForCausalLM.from_pretrained(
    args.src,
    dtype=torch.bfloat16,
    device_map="auto",
    # GPTQ builds a Hessian per Linear on top of the weights: leave GPU headroom.
    # GPTQ's Hessian is in_features^2 fp32 — 4 GiB for down_proj alone (32768^2).
    # The weights get a small share of the card so the Hessians have somewhere to live.
    # The split is bounded at BOTH ends. GPU: GPTQ needs room for a Hessian on top of the
    # resident weights (in_features^2 fp32, 4 GiB at down_proj). CPU: whatever is not on the
    # card sits in host RAM, and `save_pretrained` then materializes the state dict on top of
    # it. At cpu=56GiB the process reached 48.9 GB RSS BEFORE saving and the OOM killer took
    # it during the save, twice, silently. Pushing weights onto the card buys the save its
    # headroom back.
    # BOUNDED AT BOTH ENDS on a 32 GB card + 60 GB host, and 6/56 is the only split that
    # clears both walls at 128 x 512 calibration:
    #   * more on the CPU  -> 48.9 GB RSS before `save_pretrained`, which then has nowhere to
    #     materialize the state dict; the OOM killer takes the process AFTER all 41 layers are
    #     quantized, silently (the log just ends at `finalize`).
    #   * more on the GPU  -> "Sequential pipeline ran out of memory" at layer 2, because GPTQ
    #     needs a Hessian (in_features^2 fp32, 4 GiB at down_proj) on top of resident weights.
    # Raising calibration volume past ~128 x 512 needs a bigger HOST, not a knob.
    max_memory={0: "12GiB", "cpu": "42GiB"},
)

texts = []
try:
    he = load_dataset("dicta-il/dictalm2.0-quant-calib-dataset", split="train")
    col = [c for c in he.column_names if he.features[c].dtype == "string"][0]
    texts += [t for t in he[col] if isinstance(t, str) and len(t) > 200]
    print(f"[gptq] dicta calib rows: {len(texts)}", flush=True)
except Exception as e:
    print(f"[gptq] dicta calib unavailable ({e})", flush=True)
for lang, n in (("he", 4000), ("en", 1000)):
    try:
        ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
        take = [r["text"] for i, r in zip(range(n), ds) if len(r["text"]) > 500]
        texts += take
        print(f"[gptq] {lang} wikipedia rows: {len(take)}", flush=True)
    except Exception as e:
        print(f"[gptq] {lang} wikipedia unavailable ({e})", flush=True)

random.Random(args.seed).shuffle(texts)
texts = texts[: args.samples]
from datasets import Dataset
ds = Dataset.from_list([{"text": t} for t in texts]).map(
    lambda r: tok(r["text"], truncation=True, max_length=args.seqlen), remove_columns=["text"]
)
print(f"[gptq] calibrating on {len(ds)} sequences x {args.seqlen}", flush=True)

fp8_families = [f for f in args.keep_fp8.split(",") if f]
fp8_targets = [f"re:.*{f}_proj$" for f in fp8_families]
gptq_ignore = ["lm_head"] + fp8_targets

recipe = []
if fp8_targets:
    # Weight-only per-CHANNEL FP8, byte-identical in shape to the mixed mint's group_1:
    # num_bits 8, type float, strategy channel, no activation quantization. Round-to-nearest
    # is the point here — these tensors are being kept OUT of the 4-bit grid, not fitted to it.
    recipe.append(
        QuantizationModifier(
            config_groups={
                "fp8_reductions": QuantizationScheme(
                    targets=fp8_targets,
                    weights=QuantizationArgs(
                        num_bits=8,
                        type=QuantizationType.FLOAT,
                        strategy=QuantizationStrategy.CHANNEL,
                        symmetric=True,
                        dynamic=False,
                    ),
                )
            },
            ignore=["lm_head"],
        )
    )
    print(f"[gptq] FP8 (channel) on: {fp8_targets}", flush=True)

recipe.append(GPTQModifier(
    targets="Linear",
    scheme=args.scheme,
    ignore=gptq_ignore,
    # actorder OFF on purpose: it emits a `weight_g_idx` column permutation, a checkpoint
    # shape memra does not read today (the same class of blocker AWQ's pre_quant_scale was),
    # and group-16 NVFP4 gains little from it.
    actorder=None,
))

oneshot(
    model=model,
    processor=tok,   # required when a dataset is passed
    dataset=ds,
    recipe=recipe,
    max_seq_length=args.seqlen,
    num_calibration_samples=len(ds),
    # `basic` runs plain calibration forwards with hooks instead of fx-tracing the model.
    # The tracing pipeline dies on transformers 5.x (config values arrive wrapped as
    # IntermediateValue and hit huggingface_hub's strict dataclass validation), and 0.13
    # hard-requires transformers 5.9-5.14.1, so avoiding the tracer is the way through.
    # sequential frees Hessians per subgraph: mandatory at 24B (basic would hold
    # 4 GiB x 40 for down_proj alone). Needs the patched intermediates cache.
    pipeline="sequential",
    sequential_targets=["Ministral3DecoderLayer"],
)

# Saving is NOT left to `oneshot(output_dir=...)`: its default 50 GB shard target asks for
# 50 GB of free host RAM in one piece while the offloaded bf16 model still occupies most of
# it, and the process is OOM-killed AFTER all 41 layers are quantized (silently: the log ends
# at `finalize` with no traceback). Small shards make the peak bounded.
del ds, texts
import gc, resource
gc.collect()
try:
    import torch as _t; _t.cuda.empty_cache()
except Exception:
    pass


def _rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


# The two 512-sample runs died SILENTLY here: the log ended at `finalize`, no traceback, no
# artifact, because the OOM killer took the process while save materialized the state dict on
# top of the still-resident calibration activations. Print around it so the next failure says
# where it was rather than just stopping.
print(f"[gptq] saving; peak RSS so far {_rss_gb():.1f} GB", flush=True)
# ONE shard on purpose. transformers 5.14.1 breaks in the multi-shard path when the model
# is offloaded: `weight_map.update({k: ...} for k in ...)` passes a generator of dicts to
# update() (ValueError: dictionary update sequence element #0 has length 1), and the
# recovery path then raises "could not revert some weight conversions because of
# offloading". A single shard never enters that loop. The quantized model is ~15 GB.
model.save_pretrained(args.out, save_compressed=True, max_shard_size="200GB")
print(f"[gptq] weights written; peak RSS {_rss_gb():.1f} GB", flush=True)
tok.save_pretrained(args.out)
for extra in ("chat_template.jinja", "generation_config.json"):
    s = os.path.join(args.src, extra)
    if os.path.exists(s):
        import shutil
        shutil.copy2(s, os.path.join(args.out, extra))
print("GPTQ-MINT-DONE", flush=True)
