#!/usr/bin/env bash
# red_arm.sh <built argmax-margin-probe> [<base commit>]
# The red arm of memra#1199, on CPU: the same synthetic tables through the gate script at the
# base commit and through this tree's. The measurement is a stub that prints a table; the
# verdict is each script's own (awk at the base, the built probe's --judge here).
#
#   wide flip   prefill holds token 1 by 14.4, decode picks token 2 by 2.5 (spread 16.7)
#   near tie    margins 0.02 and 0.03
#   wide row    both pick token 1, at margins 14.4 and 7.2
#
# The base gate passes all three. This tree's gate fails the wide flip and the wide row and
# passes the near tie, with a floor of 0.4.
set -uo pipefail
judge=$(realpath "$1"); base=${2:-origin/main}
repo=$(cd "$(dirname "$0")/../../.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/memra-red-arm.XXXXXX")
prime='PRIME-GATE: class=MATCH bp_argmax=7 tw_argmax=7 tw_margin=1.5e0 bp_margin=1.25e0 maxdiff=3.0e-2'
declare -A TABLE=(
    [wide-flip]='10 1 1.44e1 2 2.5e0 1.67e1 NO
11 2 1.0 2 1.0 0.1 yes'
    [near-tie]='10 1 2.0e-2 2 3.0e-2 4.0e-2 NO
11 2 1.0 2 1.0 0.1 yes'
    [wide-row]='10 1 1.44e1 1 7.2e0 7.2e0 yes
11 2 1.0 2 1.0 0.1 yes'
)
for tree in base head; do
    root=$work/$tree
    mkdir -p "$root/tools" "$root/target/release"
    if [ $tree = base ]; then
        git -C "$repo" show "$base:tools/argmax-margin-gate.sh" > "$root/tools/argmax-margin-gate.sh"
    else
        cp "$repo/tools/argmax-margin-gate.sh" "$root/tools/"
    fi
    printf 'A prompt fixture.\n' > "$root/prompt.txt"
    : > "$root/model.gguf"
    sha=$(sha256sum "$root/prompt.txt" | cut -d' ' -f1)
    printf 'model\tprompt\t%s\teager\tCPU fixture device\t4.000000000e-1\t2040\t2047\t1.000000000e-1\t2.000000000e-1\t3.000000000e-1\t-\tresearch/fixture/gate.log\n' "$sha" \
        > "$root/tools/argmax-margin-floors.tsv"
    for name in wide-flip near-tie wide-row; do
        printf '#!/bin/sh\nif [ "$1" = --judge ]; then exec %s "$@"; fi\nprintf "%%s\\n" "%s" "%s"\n' \
            "$judge" "${TABLE[$name]}" "$prime" > "$root/target/release/argmax-margin-probe"
        chmod 700 "$root/target/release/argmax-margin-probe"
        extra=()
        [ $tree = head ] && extra=(--device "CPU fixture device")
        out=$(env -u CARGO_TARGET_DIR bash "$root/tools/argmax-margin-gate.sh" "$root/model.gguf" \
            --prompt "$root/prompt.txt" --window 2 --logdir "$root/logs" "${extra[@]}" 2>&1)
        rc=$?
        echo "== $tree gate, $name: exit $rc"
        echo "$out" | grep -vE '^  raw: ' | sed 's/^/    /'
    done
done
rm -r "$work"
