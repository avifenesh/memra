#!/usr/bin/env bash
# Two more cold builds per arm, interleaved BAAB, after the CPU suites.
set -u
cd ~/.cache/memra-relprofile
python3 measure_build.py B cold 2 > /dev/null 2>&1
python3 measure_build.py A cold 2 > /dev/null 2>&1
python3 measure_build.py A cold 3 > /dev/null 2>&1
python3 measure_build.py B cold 3 > /dev/null 2>&1
echo queue4 done
