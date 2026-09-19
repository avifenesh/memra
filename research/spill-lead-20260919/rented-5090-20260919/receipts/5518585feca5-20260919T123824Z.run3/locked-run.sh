#!/usr/bin/env bash
set -euo pipefail
exec flock --close -n -x /tmp/memra-5090.lock "$@"
