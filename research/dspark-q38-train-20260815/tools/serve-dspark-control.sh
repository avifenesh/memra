#!/usr/bin/env bash
# GPU0 DSPARK control server (RadixArk drafter, FP8 target). flashinfer: fa3 refuses sm_120.
exec env CUDA_VISIBLE_DEVICES=${GPU:-0} /scratch/venvs/eval/bin/python -m sglang.launch_server \
  --trust-remote-code --model-path /scratch/models/qwen38-27b-fp8 --tp-size 1 \
  --speculative-algorithm DSPARK --speculative-draft-model-path /scratch/models/radixark-q38-dspark \
  --speculative-dspark-block-size 7 --speculative-draft-model-quantization unquant \
  --mamba-scheduler-strategy extra_buffer --attention-backend flashinfer \
  --mem-fraction-static 0.85 --port ${PORT:-30000}
