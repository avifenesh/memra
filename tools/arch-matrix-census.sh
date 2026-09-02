#!/usr/bin/env bash
# Arch-matrix census: refuses a CI that compiles fewer arches than the release matrix builds.
#
# WHY THIS EXISTS. release.yml built a 6-cell matrix (2 glibc floors x 3 CUDA arches) at the
# time of the incident this census was born from; since 2026-09-02 it builds 120a only, with
# 100a compile-covered here in ci.yml and not shipped. ci.yml
# built ONE arch — and not even explicitly: it set MEMRA_NVCC and left MEMRA_CUDA_ARCH unset, so
# build.rs::detect_arch() fell back to 120a on every GPU-less runner. crates/memra-engine/
# build.rs swaps in DIFFERENT SOURCE FILES for the other arches (cu/*_stub.cu on 89 and 90a,
# and it skips cu/qmatvec_gemm.cu on 100a), so two of the three release arches were compiled
# for the FIRST TIME by the tag itself. A stub that had drifted out of ABI parity therefore
# could not be discovered before the number was already public — and was not: v0.105.0 (run
# 32581590491) and v0.106.0 (run 32608092743) both died on `rust-lld: error: undefined symbol`
# in the sm_89 cell, while main sat green.
#
# The rule this encodes: ANY arch the release matrix builds must be compiled at merge time.
# The glibc/OS axis is deliberately NOT mirrored — it changes the libc floor of the shipped
# binary, not the source set or the symbols, so it cannot produce a compile or link failure
# that one OS would miss. The arch axis is the one that changes what source is compiled.
#
# Measured cost of the arch cells this census enforces (ci run 32607039282, the real numbers):
# the existing build job is 1373 s wall, of which the 120a compile is 447 s. A compile-only
# arch cell MEASURED IN CI (run 32609839631): sm_90a 553 s, sm_89 558 s. Run as PARALLEL matrix
# jobs they finish ~13.6 min before the existing job does, so added wall-clock latency is ZERO
# and the cost is ~18 runner-minutes. That matters: a gate that lengthens every merge gets
# bypassed, and a bypass habit is worse than the bug.
#
# Two refusals, both fail-closed, both reading the REAL workflow files:
#   1. an arch in release.yml's matrix that ci.yml never compiles
#   2. an arch compiled by either workflow that build.rs does not accept (typo protection —
#      build.rs asserts the set, so a bad value fails only once that cell runs)
#
# Usage: tools/arch-matrix-census.sh [ci.yml] [release.yml] [build.rs]
# Callers: ci.yml, release.yml guard job, tools/test_arch_matrix_census.sh (teeth).
set -euo pipefail

ci=${1:-.github/workflows/ci.yml}
rel=${2:-.github/workflows/release.yml}
build_rs=${3:-crates/memra-engine/build.rs}

fail() { echo "::error::arch-census: $*"; exit 1; }

for f in "$ci" "$rel" "$build_rs"; do
  [ -f "$f" ] || fail "missing $f"
done

# release.yml: the cuda_arch matrix axis, e.g.  cuda_arch: ["120a", "90a", "89"]
rel_arches=$(grep -m1 'cuda_arch:' "$rel" | grep -o '"[^"]*"' | tr -d '"' | tr '\n' ' ')
[ -n "$rel_arches" ] || fail "could not parse the cuda_arch matrix axis out of $rel"

# ci.yml: every explicit MEMRA_CUDA_ARCH value. Explicit is required — an unset value means
# detect_arch() decides, which reads nvidia-smi, which means CI's coverage would silently
# change the day a runner has a GPU. The census refuses to infer it.
ci_arches=$( { grep -oE 'MEMRA_CUDA_ARCH:[[:space:]]*"?[0-9a-z]+"?' "$ci" || true; } \
            | sed -E 's/.*:[[:space:]]*"?([0-9a-z]+)"?/\1/' | sort -u | tr '\n' ' ')
ci_matrix=$( { sed -n '/cuda_arch:/p' "$ci" | grep -o '"[^"]*"' || true; } | tr -d '"' | tr '\n' ' ')
ci_arches="$ci_arches $ci_matrix"

# build.rs: the accepted set, from the assert that guards it.
ok_arches=$(grep -o 'matches!(cuda_arch.as_str(),[^)]*)' "$build_rs" \
            | head -1 | grep -o '"[^"]*"' | tr -d '"' | tr '\n' ' ')
[ -n "$ok_arches" ] || fail "could not parse the accepted arch set out of $build_rs"

# --- refusal 1 -----------------------------------------------------------------------------
for a in $rel_arches; do
  case " $ci_arches " in
    *" $a "*) ;;
    *) uncovered="${uncovered:-} $a" ;;
  esac
done
[ -z "${uncovered:-}" ] || fail "release.yml builds arch(es)$uncovered that $ci never compiles — build.rs substitutes different sources per arch, so those cells are first compiled by the TAG (this is how v0.105.0 and v0.106.0 both died). Add a compile-only matrix cell per arch; it costs ~510 s in PARALLEL, i.e. no added wall time"

# --- refusal 2 -----------------------------------------------------------------------------
for a in $ci_arches $rel_arches; do
  case " $ok_arches " in
    *" $a "*) ;;
    *) bad="${bad:-} $a" ;;
  esac
done
[ -z "${bad:-}" ] || fail "arch(es)$bad appear in a workflow but $build_rs does not accept them (accepted: $ok_arches)"

# --- refusal 3: an ADVISORY arch must never be shipped ---------------------------------------
# tools/fatbin-census-advisory.txt lists arches whose fatbin-vs-lookup census is non-blocking
# because we already know kernels are missing (sm_89 was the live case on 2026-08-23: 20 of
# them, measured). That
# downgrade is only defensible while such an arch is COMPILE COVERAGE ONLY. The moment it
# appears in release.yml's matrix it becomes a published tarball that panics at
# `Engine::func` on first use — which is exactly what v0.107.0's sm_89 asset does, and what
# this refusal exists to stop happening again. An arch is compile-covered OR shipped, never both.
advisory_file=${4:-tools/fatbin-census-advisory.txt}
if [ -f "$advisory_file" ]; then
  adv=$(sed 's/#.*//' "$advisory_file" | tr -d '[:blank:]' | grep -v '^$' || true)
  for a in $adv; do
    case " $rel_arches " in
      *" $a "*) shipped_advisory="${shipped_advisory:-} $a" ;;
    esac
    # And a listed arch that CI does not compile is a dead entry: nothing is measuring it.
    case " $ci_arches " in
      *" $a "*) ;;
      *) dead_advisory="${dead_advisory:-} $a" ;;
    esac
  done
  [ -z "${shipped_advisory:-} " ] 2>/dev/null || true
  if [ -n "${shipped_advisory:-}" ]; then
    fail "arch(es)$shipped_advisory are listed advisory in $advisory_file (their fatbin census is non-blocking because kernels are KNOWN missing) but release.yml still BUILDS them — that publishes a tarball which panics at Engine::func on first use. Remove them from release.yml's matrix, or fix the kernels and delete the advisory line"
  fi
  if [ -n "${dead_advisory:-}" ]; then
    fail "arch(es)$dead_advisory are listed advisory in $advisory_file but $ci never compiles them — a non-blocking census over an arch nobody builds measures nothing; either add a ci cell or delete the line"
  fi
fi

echo "arch-census: OK — release builds [$rel_arches], ci compiles [$(echo $ci_arches)], build.rs accepts [$ok_arches], advisory [${adv:-none}]"
