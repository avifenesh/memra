#!/usr/bin/env bash
# Phase 2: build train files + 3 concurrent training arms (2 cards each), cards 6-7 stay for recon.
# Waits for regen + MT addendum. Logs: ~/arms.log
set -ux
exec >> /home/ubuntu/arms.log 2>&1
PYE=/scratch/venvs/eval/bin/python
SF=/scratch/repos/SpecForge
OUT=/scratch/corpus
until grep -q "MT ADDENDUM DONE" /home/ubuntu/regen-mt.log 2>/dev/null; do sleep 120; done

# 1. train files
$PYE - <<'EOF'
import json, pathlib
out = pathlib.Path("/scratch/corpus/train")
out.mkdir(exist_ok=True)
def rows(p):
    p = pathlib.Path(p)
    if not p.is_file(): return []
    return [json.loads(l) for l in p.open() if l.strip()]
own = []
for f in ["own-think-exploded.jsonl","own-nothink.jsonl","own-mt-think-exploded.jsonl","own-mt-nothink.jsonl",
          "own-think.jsonl","own-mt-think.jsonl"]:
    # prefer exploded variants; fall back to raw think files only if exploded missing
    if "exploded" in f or "nothink" in f:
        own += rows(f"/scratch/corpus/regen/{f}")
pb = rows("/scratch/corpus/regen/pb-think-exploded.jsonl") or rows("/scratch/corpus/regen/pb-think.jsonl")
pb += rows("/scratch/corpus/regen/pb-nothink.jsonl")
print("own rows", len(own), "pb rows", len(pb))
# Arm A: own x4 oversample + pb
with (out/"arm-a-own-mix.jsonl").open("w") as f:
    for i in range(4):
        for r in own:
            f.write(json.dumps({**r, "id": f'{r.get("id","x")}-dup{i}'})+"\n")
    for r in pb: f.write(json.dumps(r)+"\n")
# Arm B: pb only
with (out/"arm-b-pb.jsonl").open("w") as f:
    for r in pb: f.write(json.dumps(r)+"\n")
# Arm C: own x4 only + pb (same as A; geometry differs)
import shutil; shutil.copy(out/"arm-a-own-mix.jsonl", out/"arm-c-own-mix.jsonl")
print("train files written")
EOF

# 2. arm C draft config: z-lab 3.6 geometry against the 3.8 target
$PYE - <<'EOF'
import json
z = json.load(open("/scratch/models/zlab-q36-dflash/config.json"))
z.pop("auto_map", None); z["architectures"] = ["DSparkDraftModel"]
z["mask_token_id"] = 248077
z.setdefault("dflash_config", {})
z["dflash_config"].setdefault("target_layer_ids", z.get("target_layer_ids", [1,16,31,46,61]))
z["dflash_config"]["mask_token_id"] = 248077
json.dump(z, open("/scratch/repos/SpecForge/configs/qwen3.8-27b-dflash16-warm36.json","w"), indent=4)
print("arm C config written; block", z.get("block_size"), "taps", z["dflash_config"]["target_layer_ids"])
EOF

# 3. stop regen replica servers, free all cards
for g in 0 1 2 3 4 5 6 7; do tmux kill-session -t srv$g 2>/dev/null || true; done
sleep 10

# 4. launch arms
cd $SF
launch_arm() { # name cfgjson data capgpu traingpu port extra
  local NAME=$1 CFG=$2 DATA=$3 CAP=$4 TRN=$5 PORT=$6 EXTRA=${7:-}
  tmux new-session -d -s $NAME "cd $SF && VIRTUAL_ENV=/scratch/venvs/train /scratch/venvs/train/bin/specforge train \
    -c examples/configs/qwen3.6-27b-dspark-disaggregated.yaml \
    model.target_model_path=/scratch/models/qwen38-27b-fp8 \
    model.draft_model_config=$CFG \
    model.mask_token_id=248077 \
    data.train_data_path=$DATA \
    data.chat_template=qwen3.5 data.max_length=4096 \
    training.save_interval=125 \
    run_id=$NAME output_dir=/scratch/ckpt/$NAME \
    deployment.disaggregated.control_dir=/scratch/ckpt/$NAME/control \
    'deployment.disaggregated.managed_local.trainer_cuda_visible_devices=[\"$TRN\"]' \
    'deployment.disaggregated.managed_local.capture_servers=[{\"port\":$PORT,\"cuda_visible_devices\":[\"$CAP\"],\"tp_size\":1,\"mem_fraction_static\":0.7}]' \
    $EXTRA 2>&1 | tee /home/ubuntu/$NAME.log"
}
launch_arm arm-a-own-cold configs/qwen3.8-27b-dspark.json /scratch/corpus/train/arm-a-own-mix.jsonl 0 1 31000
launch_arm arm-b-pb-control configs/qwen3.8-27b-dspark.json /scratch/corpus/train/arm-b-pb.jsonl 2 3 32000
launch_arm arm-c-warm36 configs/qwen3.8-27b-dflash16-warm36.json /scratch/corpus/train/arm-c-own-mix.jsonl 4 5 33000 \
  model.draft_checkpoint_path=/scratch/models/zlab-q36-dflash
echo "ARMS LAUNCHED $(date -u +%FT%TZ)"
