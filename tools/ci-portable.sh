#!/usr/bin/env bash
# ci-portable.sh: a NAME, not an executor. PR #590 (2026-09-21) landed this file as a second
# runner of the tier, KV and onboarding-CLI suites (`cargo test --release --locked`, no skip
# census, no floor, no teeth) next to tools/portable-suites.sh from PR #592; main carried two
# ci.yml jobs named portable-suites and GitHub refused the file. Folded on day 14 of
# lane/spill-d: tools/portable-suites.sh is the ONE entry point (skip census at budget 0,
# --min-passed floor, banked raw log, tools/test_portable_suites.sh teeth). This file only
# forwards, so anything still holding the #590 name lands in the executor; nothing tracked
# calls it (ci.yml and tools/local-ci.sh call the wrapper directly), and the teeth assert it
# runs no cargo of its own. CPU execution, never GPU qualification.
set -euo pipefail
exec "$(dirname -- "$0")/portable-suites.sh" "$@"
