#!/usr/bin/env bash
set -euo pipefail
exec flock -n -x /tmp/memra-5090.lock "$@"
