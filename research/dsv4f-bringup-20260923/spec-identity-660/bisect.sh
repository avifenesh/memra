#!/usr/bin/env bash
# bisect.sh <tag> <cfg>...: each cfg is "name:VAR=v,VAR=v:flags"; runs the DSpark gate bit cells
# (MEMRA_DSV4_GATE_BITONLY) per cfg, serially, waiting for the box GPU lock between runs.
set -uo pipefail
tag=$1; shift
bin=/root/lane/target/release/dsv4-gpu-dspark-gate
for cfg in "$@"; do
    name=${cfg%%:*}; rest=${cfg#*:}; vars=${rest%%:*}; flags=${rest#*:}
    out=/root/rcpt/$tag-$name
    while ! flock -n /tmp/memra-gpu.lock true; do sleep 10; done
    envs=(MEMRA_DSV4_DRAFTER=dspark MEMRA_DSV4_DECODE_PATH=device MEMRA_DSV4_GATE_BITONLY=1)
    [[ -n "$vars" ]] && IFS=, read -ra extra <<< "$vars" && envs+=("${extra[@]}")
    # shellcheck disable=SC2086
    env "${envs[@]}" /root/box/gate.sh "$out" $bin /data/dsv4f/nvfp4 /root/box/dspark-fx-tape416.json "$out/out" 1 0,1 $flags
    echo "$name $(grep -E 'verdict \(c\)|bit-gate:' "$out/gate.log" | tr '\n' ' ')" >> /root/rcpt/$tag.summary
done
echo BISECT_DONE >> /root/rcpt/$tag.summary
