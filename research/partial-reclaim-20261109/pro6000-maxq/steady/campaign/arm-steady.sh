#!/usr/bin/env bash
# One steady-traffic arm on one RTX PRO 6000 Max-Q (run under tools/memra-gpu-run, which sets CUDA_VISIBLE_DEVICES to
# the leased card): boot memra-server on Qwen3.8-27B NVFP4 + Q5_K with its MTP head at MEMRA_CTX=32768, run
# harness/steady.py with $STEADY_ARGS, stop it. No power raise: the box's own cap holds. A watchdog applies the Max-Q
# thermal rule (>= 95 C, or >= 92 C with the fan >= 95 % on 3 samples in a row, 10 s apart) and stops the server.
#   arm-steady.sh LABEL BIN OUTDIR [VAR=VALUE ...]
set -u
LABEL=$1; BIN=$2; OUT=$3; shift 3
K=<kit>
MODEL=<models>/Qwen3.8-27B-NVFP4-Q5K-mtp.gguf
H=$K/reclaim/harness
GPU=${CUDA_VISIBLE_DEVICES:?run under memra-gpu-run}
PORT=${ARM_PORT:-18310}
D=$OUT/$LABEL; mkdir -p "$D"
{
  echo "arm $LABEL workload=steady start=$(TZ=Asia/Jerusalem date -Is) gpu=$GPU lease=${MEMRA_GPU_LEASE_FILE:-}"
  sha256sum "$BIN"
  echo "env: $* STEADY_ARGS=${STEADY_ARGS:-}"
  nvidia-smi -i "$GPU" --query-gpu=index,name,temperature.gpu,fan.speed,clocks.sm,power.limit,memory.used,memory.free --format=csv,noheader
} > "$D/arm.txt"
nvidia-smi -i "$GPU" --query-gpu=timestamp,memory.used,memory.free,temperature.gpu,fan.speed,clocks.sm,power.draw,power.limit --format=csv,noheader -lms 250 > "$D/gpu-tel.csv" 2>&1 &
TEL=$!
env CUDA_DEVICE_ORDER=PCI_BUS_ID MEMRA_COMPAT=openai "MEMRA_MODELS=gate=$MODEL" "MEMRA_ADDR=127.0.0.1:$PORT" \
  MEMRA_CTX=32768 "$@" "$BIN" > "$D/serve.log" 2>&1 &
SP=$!
python3 - "$GPU" "$SP" "$D/thermal.txt" <<'EOF' &
import subprocess, sys, time, os, signal
gpu, sp, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
hot = 0; tmax = fmax = 0
with open(out, "w") as f:
    while True:
        try:
            os.kill(sp, 0)
        except OSError:
            break
        r = subprocess.run(["nvidia-smi", "-i", gpu, "--query-gpu=temperature.gpu,fan.speed", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip()
        try:
            t, fan = (int(float(x)) for x in r.split(","))
        except ValueError:
            time.sleep(10); continue
        tmax, fmax = max(tmax, t), max(fmax, fan)
        hot = hot + 1 if (t >= 95 or (t >= 92 and fan >= 95)) else 0
        if hot >= 3:
            f.write(f"thermal stop at {time.strftime('%T')}: {t} C, fan {fan} %\n"); f.flush()
            os.kill(sp, signal.SIGTERM)
            break
        time.sleep(10)
    f.write(f"max {tmax} C, max fan {fmax} %\n")
EOF
WD=$!
rc=0
if ! waitfor --http "http://127.0.0.1:$PORT/v1/models" --timeout 8m --every 3s --fail-after 160 --quiet; then
  echo "boot FAILED" >> "$D/arm.txt"; tail -n 20 "$D/serve.log" >> "$D/arm.txt"; rc=3
else
  # shellcheck disable=SC2086
  timeout --foreground 4800 python3 "$H/steady.py" "$PORT" "$D/rows.json" ${STEADY_ARGS:-} > "$D/result.json" 2> "$D/client.err" || rc=$?
fi
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader > "$D/compute-apps-end.txt" 2>&1
kill -TERM $SP 2>/dev/null; waitfor --gone $SP --timeout 90s --quiet || kill -KILL $SP 2>/dev/null
wait $SP 2>/dev/null
kill $TEL 2>/dev/null; wait $TEL $WD 2>/dev/null
grep -a -m1 '\[prefix-cache\] on:' "$D/serve.log" | cut -c1-1200 >> "$D/arm.txt"
grep -a -ci 'out of memory\|CUDA_ERROR_OUT_OF_MEMORY' "$D/serve.log" | sed 's/^/oom_lines=/' >> "$D/arm.txt"
grep -a -c 'reclaim-on-defer' "$D/serve.log" | sed 's/^/reclaim_on_defer_lines=/' >> "$D/arm.txt"
cat "$D/thermal.txt" >> "$D/arm.txt" 2>/dev/null
cat "$D/result.json" >> "$D/arm.txt" 2>/dev/null
echo "rc=$rc end=$(TZ=Asia/Jerusalem date -Is)" >> "$D/arm.txt"
exit $rc
