#!/usr/bin/env bash
# Check literal runtime MEMRA_* environment reads against docs/FLAGS.md.
# EVERY uncovered name fails. There is no grandfather list — it was deleted 2026-08-23 once
# measurement showed all 75 of its entries were already documented, i.e. every exemption was
# dead while still being able to absorb a future one. See the retired-grandfather note below.
set -euo pipefail
shopt -s extglob

# Pin collation: comm requires its inputs sorted under the SAME locale it compares with;
# en_US-style collation folds underscores and made comm reject sort's own output.
export LC_ALL=C

cd -- "$(dirname -- "$0")/.."

# --list: print the CENSUS (one runtime MEMRA_* name per line, nothing else) and exit 0.
#
# It exists because its absence made an assertion impossible to write. tools/test_check_flags.sh
# could only ask "does this name appear in docs/FLAGS.md" — and a name in a doc says nothing
# about whether the census can SEE the read. That is exactly how
# MEMRA_ALLOW_UNKNOWN_PRETOKENIZER passed: documented, and invisible to the gate. With --list
# the fixture asserts the census itself, on the live tree.
#
# The census is computed by the same code below either way; --list only changes what is
# printed. A separate lister would be a second implementation and could drift green.
LIST_ONLY=0
if [[ "${1:-}" == "--list" ]]; then
    LIST_ONLY=1
    shift
fi

flag_doc=docs/FLAGS.md

# ---- the grandfather list is GONE, and its absence is asserted (2026-08-23) ----
# This gate used to carry a 75-name baseline at research/docsync3-20260811/flags-drift.txt whose
# entries were exempt from the census: `comm -23 uncovered baseline` dropped them, so an
# undocumented name in that list exited 0.
#
# It was deleted rather than taught to expire, because MEASUREMENT SAID THERE WAS NOTHING TO
# EXPIRE: all 75 entries had since been documented in docs/FLAGS.md anyway, so every exemption
# was already dead. An expiry checker for a file whose every entry is inert is machinery
# maintaining an empty room. Deleting the file is the smaller change AND the stronger one.
#
# The hazard it left behind is the reason this is not merely tidying. Probed in a throwaway tree
# before the cut: delete MEMRA_SPEC's docs/FLAGS.md row and the census EXITED 0, because the
# baseline still covered the name; the same tree with the name absent from the baseline exited 1.
# So for 75 live flags — MEMRA_SPEC, MEMRA_VERIFY_GATE, MEMRA_DEBUG among them — DELETING
# DOCUMENTATION kept this gate green. Note the exact shape: not invisible, NON-BLOCKING. The
# names were still printed under "uncovered runtime names (known and new)", which is why it
# escaped notice for weeks — a printed non-fatal line is the same shape as the local-ci.sh
# WARNING that let three commits red main on 2026-08-23.
#
# House law banked from three instances in one night (this baseline, the public-boundary
# allowlist before rule-scoping, and verify-allowlist with no caller at all): AN EXCEPTIONS LIST
# NEEDS ITS OWN EXPIRY CHECK, OR IT SILENTLY ABSORBS THE REGRESSION IT WAS NEVER GRANTED FOR —
# and when every entry is dead, DELETE THE FILE rather than maintain a checker for nothing.
#
# Absence is asserted rather than assumed, so re-granting is a deliberate act with a stated
# reason instead of a file quietly reappearing. Both doors are refused: the retired path, and
# the env that used to relocate it. The env is refused rather than IGNORED on purpose — a
# no-op environment variable is how a caller believes it is grandfathering while it is not.
retired_baseline=research/docsync3-20260811/flags-drift.txt
if [[ -e "$retired_baseline" ]]; then
    echo "check-flags: REFUSED — the retired grandfather list is back: $retired_baseline" >&2
    echo "    Every one of its 75 entries was already documented when it was deleted, and while" >&2
    echo "    it existed, deleting a docs/FLAGS.md row for any of them kept this gate GREEN." >&2
    echo "    If a grandfather list is genuinely wanted again, it needs an expiry check and an" >&2
    echo "    owner ruling — not a file. Delete it, or document the flags instead." >&2
    exit 2
fi
if [[ -n "${MEMRA_FLAGS_DRIFT_BASELINE:-}" ]]; then
    echo "check-flags: REFUSED — MEMRA_FLAGS_DRIFT_BASELINE is set but baselines are retired." >&2
    echo "    It is refused rather than ignored: a no-op env is how a caller believes it is" >&2
    echo "    grandfathering a flag when the gate has already stopped honouring it." >&2
    echo "    unset MEMRA_FLAGS_DRIFT_BASELINE" >&2
    exit 2
fi

# EVERY crate's src, discovered, not a hand-maintained subset of three.
#
# The v0.94.0 train is why. The engine-hardening lane added
# MEMRA_ALLOW_UNKNOWN_PRETOKENIZER — the flag that decides whether a model with an
# unrecognized GGUF pre-tokenizer loads AT ALL — in crates/memra-tokenizer/src, which this
# list did not name, so the gate whose entire job is catching an undocumented operator flag
# was blind to the most consequential kind of flag there is. It also undercut the flags-docs
# lane's "546 census, 0 stale" claim, which was measured through the same hole.
# memra-gguf/src (6 reads) and memra-probe/src (1) sat outside the census for the same
# reason. A hand-list has this hole again the next time a crate is added; discovery does not.
# We also include `src/bin` (memra#120): harness/gate/probe binaries declare operator knobs
# and benchmark flags that must be documented in FLAGS.md rather than escaping census.
runtime_dirs=()
while IFS= read -r runtime_dir; do
    runtime_dirs+=("$runtime_dir")
done < <(find crates -mindepth 2 -maxdepth 3 -type d \( -path '*/src' -o -path '*/src/bin' \) 2>/dev/null | sort)

command -v rg >/dev/null || { echo "check-flags: rg is required" >&2; exit 2; }
[[ -f "$flag_doc" ]] || { echo "check-flags: missing $flag_doc" >&2; exit 2; }
if (( ${#runtime_dirs[@]} == 0 )); then
    echo "check-flags: no crates/*/src or crates/*/src/bin found — the census would be vacuously green" >&2
    exit 2
fi

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
reads_file="$temp_dir/reads"
literal_file="$temp_dir/literal"
const_map="$temp_dir/constmap"
patterns_file="$temp_dir/patterns"
uncovered_file="$temp_dir/uncovered"

# Pass A — literal reads. Keep this deliberately grep-shaped: it catches literal reads in
# source while avoiding MEMRA_* names that appear only in comments or in set_var calls inside
# test/bench harnesses.
rg -N 'env::var(_os)?[[:space:]]*\(|(option_)?env![[:space:]]*\(' \
    "${runtime_dirs[@]}" --glob '*.rs' 2>/dev/null \
    | rg -o 'MEMRA_[A-Z0-9_]+' | sort -u > "$literal_file" || true

# Pass B — CONST-INDIRECTED reads. Pass A matches a MEMRA_ name on the same line as the env
# call, so `std::env::var(ALLOW_UNKNOWN_PRETOKENIZER_ENV)` — with the name three lines up in a
# `pub const` — is invisible to it. That is the second half of the v0.94.0 blind spot, and it
# is not an exotic style: a flag whose name appears in an error message as well as in the read
# wants to be a const exactly so the two cannot drift.
#
# Resolved, not guessed: collect `const|static NAME: &str = "MEMRA_…"` bindings, then count a
# binding as a read only when NAME is actually passed to an env call somewhere in the same
# dirs. A const that is merely declared stays out, so this widens detection without inventing
# flags. `&'static str` and a leading `&`/module path at the call site are both tolerated.
# --no-filename as well as -N: rg suppresses line numbers with -N but still prefixes the path
# when given more than one search dir, and a const name read as `path/to.rs:NAME` matches
# nothing at the call site — a silent no-op that would have looked exactly like a clean pass.
rg -N --no-filename -o --replace '$1 $2' \
    '(?:const|static)\s+([A-Z][A-Z0-9_]*)\s*:\s*&(?:'"'"'static\s+)?str\s*=\s*"(MEMRA_[A-Z0-9_]+)"' \
    "${runtime_dirs[@]}" --glob '*.rs' 2>/dev/null | sort -u > "$const_map" || true

while read -r const_name flag_name; do
    [[ -n "${const_name:-}" && -n "${flag_name:-}" ]] || continue
    # Fail loudly rather than silently skipping if the capture ever regresses again.
    [[ "$const_name" == +([A-Z0-9_]) ]] || {
        echo "check-flags: malformed const capture: '$const_name'" >&2; exit 2; }
    if rg -q -N \
        "(env::var(_os)?|(option_)?env!)[[:space:]]*\([[:space:]]*&?([A-Za-z0-9_]+::)*${const_name}\b" \
        "${runtime_dirs[@]}" --glob '*.rs' 2>/dev/null
    then
        printf '%s\n' "$flag_name" >> "$literal_file"
    fi
done < "$const_map"

sort -u "$literal_file" > "$reads_file"

if (( LIST_ONLY )); then
    # Refuse to print an EMPTY census as a success: an empty list is what every upstream
    # failure in this script degrades to, and a consumer asserting "MEMRA_X in the list" would
    # read that as "the flag is not read" rather than "the census broke".
    if [[ ! -s "$reads_file" ]]; then
        echo "check-flags --list: census is EMPTY across ${#runtime_dirs[@]} runtime dir(s)" >&2
        exit 2
    fi
    cat "$reads_file"
    exit 0
fi

# A trailing * is a documented prefix row, e.g. MEMRA_PP_* covers every PP seam.
rg -o 'MEMRA_[A-Z0-9_]+\*?' "$flag_doc" | sort -u > "$patterns_file" || true

is_documented() {
    local name=$1 pattern prefix
    while IFS= read -r pattern; do
        case "$pattern" in
            *\*)
                prefix=${pattern%\*}
                [[ "$name" == "$prefix"* ]] && return 0
                ;;
            "$name")
                return 0
                ;;
        esac
    done < "$patterns_file"
    return 1
}

: > "$uncovered_file"
while IFS= read -r name; do
    if ! is_documented "$name"; then
        printf '%s\n' "$name" >> "$uncovered_file"
    fi
done < "$reads_file"

echo "check-flags: runtime literal reads=$(wc -l < "$reads_file" | tr -d ' ')"
# EVERY uncovered name fails now. There is no "known and new" split any more, because there is
# no list of known-and-forgiven names — that distinction WAS the hole: the split printed both
# classes and then failed on only one of them.
if [[ -s "$uncovered_file" ]]; then
    echo "check-flags: UNCOVERED runtime names — add a row to $flag_doc for each:" >&2
    sed 's/^/  /' "$uncovered_file" >&2
    echo "" >&2
    echo "    A row goes in the SAME COMMIT as the read that introduced it; one prefix row" >&2
    echo "    (a backticked name plus a trailing asterisk) covers a whole family." >&2
    echo "    There is no baseline to add it to any more — see the retired-grandfather note" >&2
    echo "    at the top of this script." >&2
    exit 1
fi
echo "check-flags: no uncovered runtime names"
echo "check-flags: every runtime MEMRA_* name resolves against '$flag_doc' (no grandfather list)"
