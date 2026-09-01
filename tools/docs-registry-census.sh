#!/usr/bin/env bash
# Docs-registry census: the registry documents against the tree they describe.
#
# docs/FLAGS.md already has its own census (tools/check-flags.sh) and this script does
# not re-derive one line of it. What had NO census until now:
#
#   1. docs/KERNELS.md cites files by path (cu/kernels.cu, mmq_ffi.rs:479-490, build.rs).
#      A renamed or deleted file leaves a row pointing at nothing, and nothing noticed.
#      Every file-shaped token in the doc must resolve somewhere in the tracked tree.
#   2. docs/MODELS.md speaks the three-state support vocabulary defined in
#      crates/memra-gguf/src/model_packs/mod.rs (NativeReference, NativeQualified,
#      NativeTuned; see CLAUDE.md "three positive support states"). Any Native-cased
#      token outside that set is a typo or an invented fourth state, both of which have
#      shipped in prose before anyone could grep for them.
#   3. docs/ROUTER.md must exist and stay at or under 60 lines. The router points at the
#      registries; a router that grows answers is quietly becoming a second registry.
#   4. Flags coverage stays check-flags.sh territory. This census only asserts that
#      tools/check-flags.sh --list still answers (exit 0, non-empty is enforced by the
#      lister itself), so the two gates cannot drift into disagreeing about whose
#      territory a flag row is.
#
# No grandfather list, same reason check-flags.sh has none: an exceptions file with no
# expiry silently absorbs the regression it was never granted for. A row that points at
# a missing file is fixed by fixing the row or restoring the file, not by exempting it.
#
# Output contract: every violation is ONE line on stderr and the script exits 1.
# A clean tree prints the counts it measured and exits 0. Refusals (broken preconditions
# that would make a pass vacuous) exit 2, like check-flags.sh.
set -euo pipefail

# Pin collation for sort/grep, same reason as check-flags.sh.
export LC_ALL=C

cd -- "$(dirname -- "$0")/.."

command -v rg >/dev/null || { echo "docs-registry-census: rg is required" >&2; exit 2; }

kernels_doc=docs/KERNELS.md
models_doc=docs/MODELS.md
router_doc=docs/ROUTER.md
flags_gate=tools/check-flags.sh

[[ -f "$kernels_doc" ]] || { echo "docs-registry-census: missing $kernels_doc" >&2; exit 2; }
[[ -f "$models_doc" ]] || { echo "docs-registry-census: missing $models_doc" >&2; exit 2; }
[[ -x "$flags_gate" ]] || { echo "docs-registry-census: missing or non-executable $flags_gate" >&2; exit 2; }

temp_dir=$(mktemp -d)
trap 'rm -rf "$temp_dir"' EXIT
violations="$temp_dir/violations"
: > "$violations"

# ---------------------------------------------------------------------------------
# 1. docs/KERNELS.md: every file-shaped reference resolves in the tree.
#
# The resolution census is `git ls-files`: exactly what a fresh CI checkout contains,
# so a row can never be kept green by an untracked local scratch file. Tokens are
# file-shaped when they end in a code/doc extension; prose, symbol names, versions
# (v0.94.0), mma shapes (m64n64k16.bf16) and build artifacts (.a/.so) never match, so
# the parser is tolerant by construction rather than by a skip list. A `:line` or
# `:line-range` suffix is excluded by the character class itself (no colon).
#
# A token resolves when any one of these holds:
#   - it is a tracked path verbatim from the repo root (ARCHITECTURE.md, docs/FLAGS.md)
#   - it is a path SUFFIX of a tracked path (cu/kernels.cu and bare mmq_ffi.rs both
#     resolve under crates/memra-engine/; the doc's own header says line references
#     resolve against a pinned commit, so rows deliberately cite crate-relative paths)
# Deliberately NO working-tree existence check: `[[ -e ]]` consults the checkout, so an
# untracked local scratch file would keep a row green here that a fresh CI checkout
# reds (the exact run-too-late split this census exists to close). Resolution comes
# from the ls-files census above and nowhere else.
# ---------------------------------------------------------------------------------
files_list="$temp_dir/files"
git ls-files > "$files_list"
[[ -s "$files_list" ]] || { echo "docs-registry-census: git ls-files returned nothing" >&2; exit 2; }

kernel_tokens="$temp_dir/kernel-tokens"
rg -o --no-filename '[A-Za-z0-9_][A-Za-z0-9_./-]*\.(cuh|cu|rs|md|py|sh|cpp|cc|hpp|h|toml|json)\b' \
    "$kernels_doc" | sort -u > "$kernel_tokens" || true

# An empty token census is what every upstream parse failure degrades to, and it would
# read as "all rows resolve". The real doc cites dozens of files; refuse the vacuous pass.
if [[ ! -s "$kernel_tokens" ]]; then
    echo "docs-registry-census: extracted ZERO file-shaped tokens from $kernels_doc (parser broke, or the doc was gutted)" >&2
    exit 2
fi

kernel_ref_count=0
while IFS= read -r tok; do
    kernel_ref_count=$((kernel_ref_count + 1))
    # Escape the only regex-special character the token class admits, then anchor:
    # exact tracked path, or suffix at a path-component boundary.
    esc=${tok//./\\.}
    grep -q -e "^${esc}\$" -e "/${esc}\$" "$files_list" && continue
    printf '%s: referenced file not in tree: %s\n' "$kernels_doc" "$tok" >> "$violations"
done < "$kernel_tokens"

# ---------------------------------------------------------------------------------
# 2. docs/MODELS.md: support-state vocabulary.
#
# Matched by SHAPE (a Native-prefixed CamelCase token), not by a hand list of places
# the doc currently says it, so the check holds no matter which table or field the
# states land in. Today the doc states support in prose and carries zero such tokens;
# zero is a legitimate pass for a vocabulary check (there is nothing off-vocabulary),
# and the measured count is printed so a reader can see which case held. A zero
# census is also VACUOUS, and a pass that inspected nothing must not read as
# coverage: it warns on stderr, every run, until the doc speaks the enum.
# ---------------------------------------------------------------------------------
state_tokens="$temp_dir/state-tokens"
rg -o --no-filename '\bNative[A-Z][A-Za-z0-9]*\b' "$models_doc" | sort > "$state_tokens" || true
state_count=$(grep -c . "$state_tokens" || true)

if (( state_count == 0 )); then
    echo "docs-registry-census: WARNING: MODELS.md support-state arm is VACUOUS: $models_doc carries zero Native* tokens (support stated in prose), so this arm verifies nothing until the doc adopts the NativeReference/NativeQualified/NativeTuned vocabulary" >&2
fi

while IFS= read -r state; do
    [[ -n "$state" ]] || continue
    case "$state" in
        NativeReference|NativeQualified|NativeTuned) ;;
        *)
            printf '%s: unknown support state %s (allowed: NativeReference NativeQualified NativeTuned; defined in crates/memra-gguf/src/model_packs/mod.rs)\n' \
                "$models_doc" "$state" >> "$violations"
            ;;
    esac
done < <(sort -u "$state_tokens")

# ---------------------------------------------------------------------------------
# 3. docs/ROUTER.md: present, and at or under 60 lines.
# ---------------------------------------------------------------------------------
router_lines=0
if [[ ! -f "$router_doc" ]]; then
    printf '%s: missing (the docs router must exist; it maps each question to its owning registry document)\n' \
        "$router_doc" >> "$violations"
else
    router_lines=$(wc -l < "$router_doc" | tr -d ' ')
    if (( router_lines > 60 )); then
        printf '%s: %s lines exceeds the 60-line cap (a router that grows answers is becoming a second registry; move content into the owning document)\n' \
            "$router_doc" "$router_lines" >> "$violations"
    fi
fi

# ---------------------------------------------------------------------------------
# 4. Flags coverage is check-flags.sh territory; assert only that it answers.
# Captured, then gated: never chained past a pipe (a pipe swallows the gate's exit).
# ---------------------------------------------------------------------------------
flags_rc=0
flags_count=0
flags_list=$("$flags_gate" --list 2>"$temp_dir/flags-err") || flags_rc=$?
if (( flags_rc != 0 )); then
    printf 'tools/check-flags.sh --list exited %s (flags coverage gate is broken; its own message: %s)\n' \
        "$flags_rc" "$(tr '\n' ' ' < "$temp_dir/flags-err")" >> "$violations"
else
    flags_count=$(printf '%s\n' "$flags_list" | grep -c . || true)
fi

# ---------------------------------------------------------------------------------
# Verdict.
# ---------------------------------------------------------------------------------
if [[ -s "$violations" ]]; then
    cat "$violations" >&2
    exit 1
fi

echo "docs-registry-census: KERNELS.md file references=$kernel_ref_count, all resolve"
echo "docs-registry-census: MODELS.md support-state tokens=$state_count, all in the three-state vocabulary"
echo "docs-registry-census: ROUTER.md lines=$router_lines (cap 60)"
echo "docs-registry-census: check-flags --list answered with $flags_count runtime names (coverage enforced by check-flags.sh itself)"
