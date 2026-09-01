#!/usr/bin/env bash
# assert-drafter-attached.sh — a gate that was HANDED a drafter must prove the drafter LOADED.
#
# THE SILENT PASS THIS CLOSES (96GB window, 2026-08-19). A qwen 27B cell was launched with
# `MEMRA_DRAFT=<frspec.gguf>` — the GEMMA assistant-drafter seam — instead of
# `MEMRA_MTP_DRAFT=` / the `MEMRA_MODELS="name=<trunk>+<draft>"` spelling. The server logged
# `models config = [(…, None)]` and NO `[mtp-draft]` line: the drafter was never loaded. Spec
# engaged anyway, off the trunk's OWN embedded MTP head, so every assertion in the cell went
# green while "the frspec drafter loads" was never tested. Two things made it invisible:
#
#   * an unused MEMRA_DRAFT on a non-gemma model is not an error (it is the gemma seam, and
#     it silently re-keys wkv_on() / fa_f16pv_on() / the MMQ-SK form on top — docs/FLAGS.md);
#   * a qwen trunk with nextn>0 has a working drafter WITHOUT the file, so the negative
#     control was unavailable by construction.
#
# The rule, enforced here: **an attach is a LOG LINE, never the absence of an error, and
# never `usage.spec.accepted > 0`** (which on a qwen trunk proves the embedded head ran).
#
#   usage: assert-drafter-attached.sh <server.log> [expected-path-substring]
#
# exit 0 = an external drafter attach line is present (and matches the path if one was given)
# exit 1 = NO attach line: the run served on the trunk's own head, whatever it was handed
#
# Both spellings count, because both are real attaches of the same mechanism:
#   MEMRA_MTP_DRAFT=<f>            -> "[mtp-draft] loading external MTP draft: <f>"
#   MEMRA_MODELS="n=<trunk>+<f>"   -> "[worker] n: regime draft attached (<f>)"
# Gemma's assistant drafter is a DIFFERENT mechanism with a different log line; a gemma arm
# passes `--gemma` so this tool asserts that one instead of the MTP pair.
set -uo pipefail
GEMMA=0
if [ "${1:-}" = "--gemma" ]; then GEMMA=1; shift; fi
LOG=${1:-}
WANT=${2:-}
[ -n "$LOG" ] || { echo "usage: $0 [--gemma] <server.log> [expected-path-substring]" >&2; exit 2; }
[ -r "$LOG" ] || { echo "DRAFTER-ATTACH: FAIL — log $LOG unreadable" >&2; exit 1; }

if [ "$GEMMA" = 1 ]; then
    PAT='gemma.*draft|draft.*attached'
    WHAT="gemma assistant drafter"
else
    PAT='\[mtp-draft\] loading external MTP draft:|regime draft attached \('
    WHAT="external MTP drafter"
fi

LINES=$(grep -E "$PAT" "$LOG" || true)
if [ -z "$LINES" ]; then
    echo "DRAFTER-ATTACH: FAIL — no $WHAT attach line in $LOG."
    echo "  This run served on the trunk's OWN head. If a drafter path was passed, the"
    echo "  WRONG SEAM was used: MEMRA_MTP_DRAFT (qwen/step35 MTP) vs MEMRA_DRAFT (gemma"
    echo "  assistant drafter) — docs/FLAGS.md 'SEAM TRAP'. Spec engaging is NOT proof."
    echo "  models-config line, for the record:"
    grep -E "models config = " "$LOG" | tail -1 | sed 's/^/    /'
    exit 1
fi
if [ -n "$WANT" ] && ! printf '%s\n' "$LINES" | grep -qF -- "$WANT"; then
    echo "DRAFTER-ATTACH: FAIL — attach line present but not for the expected artifact."
    echo "  expected to contain: $WANT"
    printf '%s\n' "$LINES" | sed 's/^/    got: /'
    exit 1
fi
printf '%s\n' "$LINES" | sed 's/^/  ok: drafter attached: /'
exit 0
