#!/usr/bin/env bash
# ci-change-class.sh: does this CI trigger's change set touch anything a compiler, linker or
# packager reads? Prints two GITHUB_OUTPUT lines, `code=true|false` and `reason=<why>`.
# ci.yml gates its compile jobs (build, clippy, server-tests, engine-tests, the arch mirrors,
# the publish dry-run) on the answer; the text gates and the boundary check run regardless.
#
# WHY (2026-09-02). ci.yml wall time was 42 min per push (run 33582547232), and a large share
# of pushes on this repo change only research receipts, docs and corpus text: a lane banking a
# cell result paid a full CUDA build of every arch to learn that nothing it touched compiles.
# The text gates already answer every question a docs change can raise (flags census, docs
# registry census, public boundary, allowlist drift); the compile jobs answer nothing for it.
#
# FAIL CLOSED. Every doubt is code=true: unknown event, unreachable or zero base (first push
# of a branch, force-push), empty diff, bad arguments, any git error. The ONLY way to skip a
# compile is a non-empty diff made solely of documentation paths. This script never exits
# non-zero: a red classification step would make the compile jobs' `needs` fail and skip them,
# which is the fail-open shape; ci.yml additionally gates on `code != 'false'` (not `== 'true'`)
# so a missing output still compiles. Teeth: tools/test_ci_change_class.sh.
#
# DOCUMENTATION PATHS (everything else is code):
#   docs/**            registry text; docs/FLAGS.md and docs/KERNELS.md are read by the text
#                      gates, which always run
#   research/**        lane receipts and tune data; nothing under crates/ or tools/ reads them
#                      at compile or test time (checked 2026-09-02: zero "research/" literals
#                      in crates/*/src)
#   agent-knowledge/** corpus text
#   *.md               anywhere EXCEPT under crates/ (a crate README is a cargo package input)
#   LICENSE, .github/ISSUE_TEMPLATE/**
#
# Usage: ci-change-class.sh <event_name> <pr_base_sha> <push_before_sha> <head_sha> [repo_dir]
set -u

event=${1:-}
pr_base=${2:-}
push_before=${3:-}
head=${4:-}
repo=${5:-.}

emit() { printf 'code=%s\nreason=%s\n' "$1" "$2"; exit 0; }

[ -n "$event" ] && [ -n "$head" ] || emit true "missing-args"
cd "$repo" 2>/dev/null || emit true "repo-dir-unreadable"
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || emit true "not-a-git-repo"

case "$event" in
  pull_request)
    base=$pr_base
    [ -n "$base" ] || emit true "pull_request-without-base"
    git cat-file -e "$base^{commit}" 2>/dev/null || emit true "pr-base-unreachable"
    git cat-file -e "$head^{commit}" 2>/dev/null || emit true "head-unreachable"
    # Three-dot: files changed on the PR side since the merge base, never the base's own drift.
    files=$(git diff --name-only "$base...$head" 2>/dev/null) || emit true "diff-failed"
    ;;
  push)
    base=$push_before
    case "$base" in
      ''|0000000000000000000000000000000000000000) emit true "push-without-before" ;;
    esac
    git cat-file -e "$base^{commit}" 2>/dev/null || emit true "push-before-unreachable"
    git cat-file -e "$head^{commit}" 2>/dev/null || emit true "head-unreachable"
    files=$(git diff --name-only "$base" "$head" 2>/dev/null) || emit true "diff-failed"
    ;;
  *)
    emit true "event-$event"
    ;;
esac

[ -n "$files" ] || emit true "empty-diff"

while IFS= read -r f; do
  [ -n "$f" ] || continue
  case "$f" in
    crates/*) emit true "code-path:$f" ;;
  esac
  if printf '%s\n' "$f" | grep -qE '^(docs/|research/|agent-knowledge/|\.github/ISSUE_TEMPLATE/)|\.md$|^LICENSE$'; then
    continue
  fi
  emit true "code-path:$f"
done <<< "$files"

emit false "docs-only:$(printf '%s\n' "$files" | grep -c .)-files"
