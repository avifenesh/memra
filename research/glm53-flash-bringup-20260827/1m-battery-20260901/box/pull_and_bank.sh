#!/usr/bin/env bash
# Runs on the RIG. Pulls the window's receipts off the box, scrubs box identity, stages them
# into the memra worktree. The box only ever produces receipts; git happens here.
set -uo pipefail
WT=/tmp/wt-1m-battery/research/glm53-flash-bringup-20260827/1m-battery-20260901
SSHC=(/usr/bin/ssh -o IdentitiesOnly=yes -o ConnectTimeout=25 -i "$HOME/.ssh/id_ed25519" -p <box-b-ssh-port> root@<ip>)
mkdir -p "$WT/receipts" "$WT/logs"
echo "=== pulling receipts ==="
"${SSHC[@]}" 'cd /root/out-1m && tar cf - receipts logs --exclude="*.serverlog" 2>/dev/null' | tar xf - -C "$WT" 2>&1 | tail -3
echo "=== pulling the serverlogs that carry engagement/acceptance evidence (bounded) ==="
"${SSHC[@]}" 'cd /root/out-1m && for f in $(find receipts -name "*.serverlog" | head -60); do echo "##### $f"; grep -aE "\[glm5-acc\]|\[glm5-phase|\[mla-tc-prefill\]|\[moe-grouped-prefill\]|\[admission\]|\[admit-oom\]|engaged|verify walk|PMIN=|resident-experts" "$f" | head -40; done' > "$WT/receipts/serverlog-evidence.txt" 2>/dev/null
wc -l "$WT/receipts/serverlog-evidence.txt"
echo "=== boot logs are large; keep only the evidence lines + head/tail per boot ==="
"${SSHC[@]}" 'cd /root/out-1m && for f in logs/boot-*.log; do echo "##### $f"; grep -aE "\[glm5-spec\]|\[spec-gate\]|\[spec-k\]|\[mla-tc-prefill\]|\[moe-grouped-prefill\]|\[admission\]|\[admit-oom\]|resident-experts|listening on|cross-device transport|template caps|engaged|verify walk|PMIN=|panic|CUDA_ERROR" "$f" | head -50; done' > "$WT/logs/boot-evidence.txt" 2>/dev/null
rm -f "$WT"/logs/boot-*.log 2>/dev/null
wc -l "$WT/logs/boot-evidence.txt"
echo "=== scrub box identity ==="
SCRUB_HOST=<ip> SCRUB_PORT=<box-b-ssh-port> python3 /tmp/1m-box/scrub.py "$WT"
echo "=== residual secret sweep (must print nothing) ==="
grep -rnE '65\.7\.5\.146|<box-b-ssh-port>|gui-apikey=[0-9a-f]|ssh2\.vast\.ai' "$WT" 2>/dev/null | head -10 || true
echo "=== sizes ==="
du -sh "$WT"; find "$WT" -type f | wc -l
