#!/usr/bin/env bash
# Queue the candidate build only after the #1089 landing's second (base) cell build holds the rig
# build lock, so the unfair flock queue cannot put this build between its two builds.
#   build-after-landing.sh CELL_EVIDENCE_PID FIRST_FLOCK_PID LABEL
set -u
CE=$1; FIRST=$2; LABEL=$3
kill -0 "$FIRST" 2>/dev/null && waitfor --gone "$FIRST" --timeout 60m --quiet
# The base build: a new flock child of the cell tool with a running cargo under it.
cat > <harness>/.second-build-probe.sh <<EOF
for f in \$(ps -o pid= --ppid $CE); do
  ps -o args= --ppid "\$f" | grep -q "cargo build" && exit 0
done
kill -0 $CE 2>/dev/null || exit 0
exit 1
EOF
waitfor --until "bash <harness>/.second-build-probe.sh" --timeout 90m --every 5s --quiet
echo "$(TZ=Asia/Jerusalem date +%H:%M:%S) queueing build $LABEL"
bash <lane>/build.sh "$LABEL"
