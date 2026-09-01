#!/usr/bin/env python3
"""Upload the GLM-5.3-Flash NVFP4 mint. Token comes from HF_TOKEN in the environment.
No credential is written by this script and none is embedded in it."""
import os, sys, pathlib
from huggingface_hub import HfApi

REPO = "Avifenesh/GLM-5.3-Flash-NVFP4"
SRC = pathlib.Path.home() / "models" / "glm53-nvfp4"
CARD = pathlib.Path.home() / "glm53-hf-lane" / "CARD.md"
# Never published:
#  - README.md in that dir is the VENDOR's card (zai's benchmark tables + citation).
#  - config.json.pre-keeplist-fix is an internal backup of the pre-fix config.
#  - .memra-repack/ is a 160 GiB regenerable expert-slab cache the engine builds on
#    load. It reappeared on the box after another lane booted a server against this
#    artifact, so excluding it explicitly is not theoretical.
EXCLUDE = [
    "README.md",
    "config.json.pre-keeplist-fix",
    ".memra-repack/*",
    ".memra-repack/**",
    "**/.memra-repack/**",
]

tok = os.environ.get("HF_TOKEN")
if not tok:
    sys.exit("HF_TOKEN missing from environment")
api = HfApi(token=tok)

stage = sys.argv[1] if len(sys.argv) > 1 else "all"

if stage in ("repo", "small", "all"):
    api.create_repo(REPO, repo_type="model", private=False, exist_ok=True)
    print("repo ready:", REPO, flush=True)

# Allowlist, not a blocklist. The upload set is 28 known files; naming them is the
# only form that cannot be widened by something new appearing in the directory.
SMALL = [
    "config.json", "generation_config.json", "hf_quant_config.json",
    "model.safetensors.index.json", "processor_config.json",
    "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
]
SHARDS = [f"model-{i:05d}-of-00020.safetensors" for i in range(1, 21)]

if stage in ("small", "all"):
    api.upload_file(
        path_or_fileobj=str(CARD), path_in_repo="README.md",
        repo_id=REPO, repo_type="model",
        commit_message="Model card: NVFP4 mint of GLM-5.3-Flash, gated against its BF16 source",
    )
    print("card up", flush=True)
    missing = [f for f in SMALL if not (SRC / f).is_file()]
    if missing:
        sys.exit(f"missing expected files: {missing}")
    api.upload_folder(
        folder_path=str(SRC), repo_id=REPO, repo_type="model",
        allow_patterns=SMALL, ignore_patterns=EXCLUDE,
        commit_message="Config, tokenizer, chat template, quant config",
    )
    print("small files up", flush=True)

if stage in ("weights", "all"):
    missing = [f for f in SHARDS if not (SRC / f).is_file()]
    if missing:
        sys.exit(f"missing shards: {missing}")
    api.upload_folder(
        folder_path=str(SRC), repo_id=REPO, repo_type="model",
        allow_patterns=SHARDS, ignore_patterns=EXCLUDE,
        commit_message="NVFP4 weights, 20 shards",
    )
    print("weights up", flush=True)

print("DONE-STAGE", stage, flush=True)
