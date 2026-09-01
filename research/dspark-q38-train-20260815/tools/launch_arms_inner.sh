#!/usr/bin/env bash
# Launch the 3 training arms in tmux, mooncake on PATH inside each session.
set -ux
SF=/scratch/repos/SpecForge
OUT=/scratch/corpus

launch_arm() {
  local NAME=$1 CFG=$2 DATA=$3 CAP=$4 TRN=$5 PORT=$6 IDX=$7 EXTRA=${8:-}
  tmux new-session -d -s "$NAME" \
    "export PATH=/scratch/venvs/train/bin:\$PATH; cd $SF && specforge train \
     -c examples/configs/qwen3.6-27b-dspark-disaggregated.yaml \
     model.target_model_path=/scratch/models/qwen38-27b-fp8 \
     model.draft_model_config=$CFG \
     model.mask_token_id=248077 \
     model.sglang_attention_backend=triton \
     runtime.in_flight_high_watermark=1024 \
     runtime.in_flight_low_watermark=512 \
     data.train_data_path=$DATA \
     data.chat_template=qwen3.5 data.max_length=4096 \
     training.save_interval=125 \
     run_id=$NAME output_dir=/scratch/ckpt/$NAME \
     deployment.disaggregated.control_dir=/scratch/ckpt/$NAME/control \
     deployment.trainer.master_port=$((29500+IDX*10)) \
     deployment.disaggregated.managed_local.mooncake.rpc_port=$((35551+IDX*10)) \
     deployment.disaggregated.managed_local.mooncake.metadata_port=$((35880+IDX*10)) \
     deployment.disaggregated.managed_local.mooncake.metrics_port=$((35903+IDX*10)) \
     deployment.disaggregated.managed_local.trainer_cuda_visible_devices='[\"$TRN\"]' \
     deployment.disaggregated.managed_local.capture_servers='[{\"port\":$PORT,\"cuda_visible_devices\":[\"$CAP\"],\"tp_size\":1,\"mem_fraction_static\":0.7}]' \
     $EXTRA 2>&1 | tee /home/ubuntu/$NAME.log"
}

launch_arm arm-a-own-cold   configs/qwen3.8-27b-dspark.json          $OUT/train/arm-a-own-mix.jsonl 0 1 31000 0
launch_arm arm-b-pb-control configs/qwen3.8-27b-dspark.json          $OUT/train/arm-b-pb.jsonl      2 3 32000 1
launch_arm arm-c-warm36     configs/qwen3.8-27b-dflash16-warm36.json $OUT/train/arm-c-own-mix.jsonl 4 5 33000 2 model.draft_checkpoint_path=/scratch/models/zlab-q36-dflash
echo "ARMS LAUNCHED"
