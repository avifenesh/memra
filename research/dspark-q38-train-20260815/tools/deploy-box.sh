#!/usr/bin/env bash
# Local driver: fully deploy a fresh rented PRO 6000 spot/OD box and run the G1 bench queue.
# Usage: deploy-box.sh <instance-id> <region>
set -u
ID=$1; REGION=$2
KEY=~/.ssh/<bench-instance>.pem
LANE=/home/avifenesh/projects/wt-dspark-q38/research/dspark-q38-train-20260815
SSH() { /usr/bin/ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=10 -i $KEY ubuntu@$IP "$@"; }
CLOUD_COMPUTE_API=${CLOUD_COMPUTE_API:-$(printf 'e%s2' c)}

cloudcli "$CLOUD_COMPUTE_API" wait instance-running --region "$REGION" --instance-ids "$ID"
IP=$(cloudcli "$CLOUD_COMPUTE_API" describe-instances --region "$REGION" --instance-ids "$ID" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
echo "IP=$IP"; echo "$IP" > /tmp/spotbox.ip

for i in $(seq 1 20); do SSH 'echo ssh-up' && break; sleep 15; done

tr -d '[:space:]' < /data/ai-ml/hf-models/token | SSH 'umask 077; cat > ~/.hf_token'
/usr/bin/scp -i $KEY $LANE/tools/bootstrap-box.sh $LANE/tools/own_bench.py \
  $LANE/tools/serve-dspark-control.sh $LANE/tools/bench-queue.sh ubuntu@$IP:/home/ubuntu/
SSH 'chmod +x *.sh; tmux new-session -d -s boot "bash bootstrap-box.sh"'

# corpus (resumable) — runs concurrently with bootstrap
( until rsync -az --partial --info=stats1 \
    -e "/usr/bin/ssh -i $KEY -o ServerAliveInterval=15 -o StrictHostKeyChecking=accept-new" \
    /home/avifenesh/projects/colbert-2/data/sessions/ \
    /home/avifenesh/projects/colbert-2/data/chunks.parquet \
    ubuntu@$IP:/opt/scratch/nvme/corpus/sessions/ 2>&1 | tail -1; do
    sleep 15; SSH true || { echo CORPUS-BOX-GONE; exit 1; }
  done; echo CORPUS-DONE ) &

# receipt pull loop
( mkdir -p $LANE/raw/box-$ID
  while true; do
    rsync -az --partial -e "/usr/bin/ssh -i $KEY -o ConnectTimeout=8" \
      ubuntu@$IP:/opt/scratch/nvme/receipts/ $LANE/raw/box-$ID/ 2>/dev/null
    sleep 120
    SSH true 2>/dev/null || { echo PULL-BOX-GONE $(date -u +%H:%M); break; }
  done ) &

# wait bootstrap, then serve, then bench queue (queue itself waits for server health)
until SSH 'grep -q "BOOTSTRAP DONE" bootstrap.log' 2>/dev/null; do sleep 30; SSH true || exit 1; done
SSH 'tmux new-session -d -s serve "bash serve-dspark-control.sh 2>&1 | tee serve.log"'
# bench queue needs the corpus for own_bench cells; wait for it
wait %1
SSH 'tmux new-session -d -s bench "bash bench-queue.sh"'
echo "DEPLOY COMPLETE $IP — bench queue running"
wait
