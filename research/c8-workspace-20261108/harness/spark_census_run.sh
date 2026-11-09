#!/usr/bin/env bash
# Rig side: copy the census check to a Spark unit and run it there. Called under sparklock run.
#   spark_census_run.sh HOST COMMIT LOG
set -uo pipefail
HOST=$1; C=$2; LOG=$3
H=$HOME/.cache/memra-c8r2
/usr/bin/ssh -n "$HOST" 'mkdir -p ~/.cache/memra-c8r2-gb10' || exit 2
scp -q "$H/harness/census_check.py" "$H/harness/spark_census.sh" "$H/census/tools-main.tgz" \
  "$H/census/tools-branch.tgz" "$HOST:.cache/memra-c8r2-gb10/" || exit 3
/usr/bin/ssh -n "$HOST" "cd ~/.cache/memra-c8r2-gb10 && timeout 40m bash spark_census.sh $C" > "$LOG" 2>&1
echo "rc=$?" >> "$LOG"
