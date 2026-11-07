#!/usr/bin/env bash
# Runs ON a GB10 host under the lease: ncu on the GDN / quantize / glue kernels of one run-gen
# prime (MEMRA_PP_PRIME=1, ~16k record prompt, 4,096-row chunks), the first launches of each.
# usage: ncu_prime.sh BIN OUTNAME
set -u
D=$HOME/.cache/memra-gb10-next
BIN=$1; NAME=$2
G8=$HOME/models/memra-q38/Qwen3.8-27B-NVFP4-MTP-GGUF/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
mkdir -p "$D/ncu"
python3 - "$D/ncu/prompt-600.txt" <<'EOF'
import sys
WORDS = "alpha river copper lantern orbit meadow cipher harbor violet granite ember falcon".split()
lines = [f"Record {i}: " + " ".join(WORDS[(i * 7 + j * 3) % 12] for j in range(14)) + f" {i % 97}." for i in range(600)]
open(sys.argv[1], "w").write("\n".join(lines))
EOF
RE='regex:^(gdn_chunk_state_mma|gdn_chunk_solve32_f32|gdn_chunk_output_mma|gdn_chunk_attn_f32|ssm_conv1d_gdn_state_f32|l2_norm_pp_v2_f32|add_f32|rms_norm_f32_v2|gated_rmsnorm_f32|qmatvec_nvfp4_dp4a_rp|quantize_q8_1|sigmoid_f32|mul_f32)$|quantize_mmq_e4m3'
# GB10 exposes no DRAM counters (tacit ncu-gb10-no-dram-counters-use-l2-read): L2 sectors bound the
# DRAM bytes for these streaming kernels. skip the first (embedding-adjacent) launches; 80 launches covers every kernel kind of a few layers
sudo -n -E env PATH="$PATH" MEMRA_PP_PRIME=1 MEMRA_PROMPT_FILE=$D/ncu/prompt-600.txt MEMRA_PP_ONLY=1 MEMRA_PP_REPS=1 MEMRA_PP_WARMUP=0 \
  /usr/local/cuda/bin/ncu ${NCU_EXTRA:-} --kernel-name-base function -k "$RE" -s 200 -c 80 \
  --metrics gpu__time_duration.sum,lts__t_sectors_op_read.sum,lts__t_sectors_op_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,lts__throughput.avg.pct_of_peak_sustained_elapsed,launch__grid_size,launch__block_size,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active \
  -f -o "$D/ncu/$NAME" "$BIN/run-gen" "$G8" > "$D/ncu/$NAME.log" 2>&1
echo "ncu rc=$?"
sudo -n chown "$USER" "$D/ncu/$NAME.ncu-rep" 2>/dev/null
/usr/local/cuda/bin/ncu -i "$D/ncu/$NAME.ncu-rep" --page raw --csv \
  > "$D/ncu/$NAME.csv" 2>&1
echo NCU-PRIME-DONE
