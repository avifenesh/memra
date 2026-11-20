#!/usr/bin/env bash
# exit 0 when MemAvailable >= 108 GiB and the GPU lists no compute process
m=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
[ "$m" -ge 108 ] && [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
