#!/usr/bin/env bash
# Several arms in one gpulock lease (fresh boot each, in the given order), so a phase of 8-minute
# arms is not stretched by a neighbour's 37-minute arms between every one of them.
#   A=BIN_A B=BIN_B group.sh PHASE N0 "B A A B" WORKLOAD [VAR=VALUE ...]
# N0: the arms of this phase already run (labels continue from N0+1).
set -u
PHASE=$1; N0=$2; ORDER=$3; WL=$4; shift 4
H=<harness>
O=<runs>/$PHASE
mkdir -p "$O"
cat > "$O/.group-$N0.sh" <<EOF
#!/usr/bin/env bash
n=$N0
for arm in $ORDER; do
  n=\$((n + 1))
  [ -e <lane>/YIELD ] && { echo "yield before \$n-\$arm"; break; }
  case \$arm in A) bin="$A" ;; B) bin="$B" ;; esac
  label=\$(printf %02d-%s "\$n" "\$arm")
  echo "\$(TZ=Asia/Jerusalem date -Is) $PHASE \$label start"
  bash "$H/arm2.sh" "\$label" "\$bin" "$O" "$WL" $*
  echo "\$(TZ=Asia/Jerusalem date -Is) $PHASE \$label rc=\$?"
done
EOF
echo "$(TZ=Asia/Jerusalem date -Is) $PHASE arms $((N0 + 1))..: $ORDER waiting for the 5090"
gpulock run --ttl 75m --wait 6h --for "memra c8r2 $PHASE arms $ORDER (fresh boot each)" -- bash "$O/.group-$N0.sh"
echo "GROUP_DONE $PHASE $N0"
