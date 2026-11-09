#!/usr/bin/env bash
# exit 0 when the RTX 5090 reads under 75 C, 1 otherwise (waitfor --until probe)
t=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
case "$t" in ''|*[!0-9]*) exit 2 ;; esac
if [ "$t" -lt 75 ]; then exit 0; fi
exit 1
