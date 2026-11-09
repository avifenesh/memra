#!/usr/bin/env bash
# Several phases' arms in one gpulock lease, each arm a fresh boot, in the given order.
#   A=BIN_A B=BIN_B group2.sh "PHASE|N0|ORDER|WORKLOAD|VAR=VALUE ..." ...
set -u
H=<harness>
R=<runs>
S=$R/.group2-$(date +%s).sh
{
  echo '#!/usr/bin/env bash'
  for spec in "$@"; do
    IFS='|' read -r phase n0 order wl envs <<< "$spec"
    echo "mkdir -p '$R/$phase'"
    echo "n=$n0"
    echo "for arm in $order; do"
    echo "  n=\$((n + 1))"
    echo "  [ -e <lane>/YIELD ] && { echo 'yield'; exit 0; }"
    echo "  case \$arm in A) bin='$A' ;; B) bin='$B' ;; esac"
    echo "  label=\$(printf %02d-%s \"\$n\" \"\$arm\")"
    echo "  echo \"\$(TZ=Asia/Jerusalem date -Is) $phase \$label start\""
    echo "  bash '$H/arm2.sh' \"\$label\" \"\$bin\" '$R/$phase' '$wl' $envs"
    echo "  echo \"\$(TZ=Asia/Jerusalem date -Is) $phase \$label rc=\$?\""
    echo "done"
  done
} > "$S"
echo "$(TZ=Asia/Jerusalem date -Is) group2 waiting for the 5090: $*"
gpulock run --ttl 80m --wait 6h --for "memra c8r2 arms: $(printf '%s ' "$@" | cut -c1-120)" -- bash "$S"
echo "GROUP2_DONE $*"
