set -euo pipefail
tools/unittest-floor.sh research/mtp-calibrated-depth-20260920 'test_*.py' 6
tools/unittest-floor.sh research/mtp-continuing-session-20260921 'test_*.py' 17

