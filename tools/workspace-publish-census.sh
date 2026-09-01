#!/usr/bin/env bash
# Workspace publish census: refuses a publish.yml crate list that does not match the workspace.
#
# WHY THIS EXISTS. publish.yml publishes crates by iterating a HARDCODED name list. On
# 2026-08-22 `d143604b0a` added two workspace members (crates/memra-reference,
# crates/memra-cli), took the workspace from 10 members to 12, and made memra-reference a
# hard `[dependencies]` entry of memra-engine — without touching publish.yml. The list still
# said 9 names for 11 publishable members. Nothing noticed until the v0.106.0 tag ran the real
# publish: six crates uploaded, then
#     error: failed to prepare local package for uploading
#     Caused by: no matching package named `memra-reference` found
# and crates.io versions are immutable, so that number is permanently half-published.
#
# WHY A DRY-RUN IS NOT ENOUGH (measured, not assumed): `cargo publish --workspace --dry-run`
# packages every member from a local overlay and never reads publish.yml, so it passes with a
# stale list. Only comparing the list to the member set catches this class. Both checks are
# wired — the dry-run for packaging drift, this census for list rot.
#
# Five refusals, all fail-closed, all read the REAL Cargo.toml and the REAL publish.yml (the
# lesson of the same night: tools/test_release_guard.sh's five arms all passed while main was
# unreleasable, because every one of them inspected a fixture):
#   1. a publishable member missing from the list      (memra-reference, memra-cli: the defect)
#   2. a list entry that is not a workspace member     (rot in the other direction)
#   3. a list entry whose manifest says publish = false
#   4. the list is not in topological order            (a crate before its own workspace dep)
#   5. a publishable member depends on a publish = false member (making an unpublishable crate
#      load-bearing breaks publish with an error no reordering can fix)
#
# Usage: tools/workspace-publish-census.sh [manifest] [workflow]
#   defaults: manifest=Cargo.toml, workflow=.github/workflows/publish.yml
# Callers: ci.yml (every push — this is the one that matters), release.yml guard job,
#          publish.yml, tools/test_workspace_publish_census.sh (teeth).
set -euo pipefail

manifest=${1:-Cargo.toml}
workflow=${2:-.github/workflows/publish.yml}
root=$(dirname "$manifest")
[ "$root" = "" ] && root=.

fail() { echo "::error::publish-census: $*"; exit 1; }

# --- the workspace side -------------------------------------------------------------------
# members = the [workspace] members array, one path per line -> crate dir names.
members=$(awk '
  /^\[workspace\]/          { s = 1 }
  s && /^members[ =]/       { inm = 1 }
  inm                       { line = line $0 }
  inm && /\]/               { print line; exit }
' "$manifest" | grep -o '"[^"]*"' | tr -d '"')

[ -n "$members" ] || fail "no [workspace] members found in $manifest"

publishable=""
declare -A deps_of
for path in $members; do
  cm="$root/$path/Cargo.toml"
  [ -f "$cm" ] || fail "workspace member $path has no Cargo.toml at $cm"
  name=$(awk -F'"' '/^\[package\]/{p=1;next} /^\[/{p=0} p && /^name[ =]/{print $2; exit}' "$cm")
  [ -n "$name" ] || fail "workspace member $path has no [package] name"
  # publish = false anywhere in the [package] table means unpublishable.
  if awk '/^\[package\]/{p=1;next} /^\[/{p=0} p && /^publish[ =]/{print; exit}' "$cm" \
       | grep -q 'false'; then
    unpublishable="${unpublishable:-} $name"
  else
    publishable="$publishable $name"
  fi
  # internal deps: memra-* entries in [dependencies] and [build-dependencies]. dev-deps are
  # deliberately excluded (they are not part of the publish dependency order); the census
  # asserts below that none exist, so the exclusion cannot silently start mattering.
  deps_of[$name]=$(awk '
    /^\[dependencies\]/       { d = 1; next }
    /^\[build-dependencies\]/ { d = 1; next }
    /^\[/                     { d = 0 }
    d && /^memra-/            { printf "%s ", $1 }
  ' "$cm")
  if awk '/^\[dev-dependencies\]/{d=1;next} /^\[/{d=0} d && /^memra-/{print}' "$cm" | grep -q .; then
    fail "$name has an internal [dev-dependencies] entry; this census only orders [dependencies] and [build-dependencies] — extend it before landing that"
  fi
done

# --- the workflow side --------------------------------------------------------------------
# The publish loop: everything between `for crate in` and `; do`, backslash continuations
# folded. Parsed from the real workflow file, never from a copy of the list.
list=$(sed -n '/for crate in/,/; do/p' "$workflow" \
       | tr '\n' ' ' | sed -e 's/.*for crate in//' -e 's/; do.*//' -e 's/\\/ /g')
list=$(echo $list)   # collapse whitespace
[ -n "$list" ] || fail "could not parse the crate list out of $workflow (looked for 'for crate in … ; do')"

# --- refusal 1: publishable member missing from the list ----------------------------------
for name in $publishable; do
  case " $list " in
    *" $name "*) ;;
    *) missing="${missing:-} $name" ;;
  esac
done
[ -z "${missing:-}" ] || fail "publishable workspace member(s)$missing absent from the crate list in $workflow — publish would fail (or silently skip the crate); add them in topological order"

# --- refusals 2 and 3: ghost entries and publish=false entries ----------------------------
for name in $list; do
  case " $publishable $unpublishable " in
    *" $name "*) ;;
    *) ghost="${ghost:-} $name" ;;
  esac
  case " ${unpublishable:-} " in
    *" $name "*) unpub_listed="${unpub_listed:-} $name" ;;
  esac
done
[ -z "${ghost:-}" ] || fail "crate list in $workflow names$ghost, which is not a workspace member"
[ -z "${unpub_listed:-}" ] || fail "crate list in $workflow names$unpub_listed, whose manifest says publish = false"

# --- refusal 4: topological order ---------------------------------------------------------
seen=""
for name in $list; do
  for d in ${deps_of[$name]:-}; do
    case " $publishable " in *" $d "*) ;; *) continue ;; esac
    case " $seen " in
      *" $d "*) ;;
      *) fail "$name is listed before its workspace dependency $d in $workflow — cargo strips the path and requires $d from the registry first; the list must be topological" ;;
    esac
  done
  seen="$seen $name"
done

# --- refusal 5: publishable crate depending on a publish=false crate ----------------------
for name in $publishable; do
  for d in ${deps_of[$name]:-}; do
    case " ${unpublishable:-} " in
      *" $d "*) fail "$name is publishable but depends on $d, whose manifest says publish = false — publish cannot succeed for $name at any position in the list; either publish $d or drop the dependency" ;;
    esac
  done
done

n_pub=$(echo $publishable | wc -w)
n_list=$(echo $list | wc -w)
echo "publish-census: OK — $n_pub publishable members, $n_list listed, order topological, unpublishable:${unpublishable:- none}"
