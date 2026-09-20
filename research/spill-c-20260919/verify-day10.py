#!/usr/bin/env python3
"""Replay the N=1 day-ten repeat cell against the sealed day-nine spec controls."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import re

LANE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('day9', LANE / 'verify-day9.py')
DAY9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY9)
DAY7 = DAY9.DAY7
require = DAY7.require
RAW10 = LANE / 'pro-single-day10'
ARGV = ['env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32', 'MEMRA_MOE_SLOTS=9986',
        '/root/wt-c/target/release/run-spec',
        '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', '55', '88', '13', '--experts-via-tier']
BINARIES = {'run-gen': 'b1c4090cb32bfba5f65bddd64ccafe407516a51dd8142b77e73184e2ad3f6d22',
            'run-spec': 'ded88cc84bfd54319b8a6bd6ee08733fedfff4b75be769c47a0b9cb29bb6ba0d'}


def replay(path):
    capture = json.loads((path / 'command.capture.json').read_text())
    DAY7.hashes(path, capture)
    require(capture['exit_code'] == 0 and not capture['timed_out'] and
            capture['parse_error'] is None, 'failed cell: ' + path.name)
    require(capture['qualification'] is False and
            capture['status'] == 'executed-not-qualified', 'collector is not qualification')
    require(capture['gpu_telemetry']['interval_ms'] == 250, 'telemetry cadence')
    require(capture['gpu_power_limits'] == [{'device': '0', 'power.limit': '600.00 W',
            'power.max_limit': '600.00 W'}], 'power envelope changed')
    require(json.loads((path / 'lock.json').read_text()) ==
            {'rig': 'pro-single', 'lock': '/tmp/memra-gpu.lock', 'acquired': True}, 'lock')
    require(capture['command'] == ARGV, 'command identity changed')
    log = (path / 'command.log').read_text()
    tape = DAY7.tokens(log)
    acceptance = re.findall(r'acceptance: ([^\n]+)', log)
    require(re.findall(r'\[generate_spec K=(\d)\]', log) == list('12345678'), 'K ladder')
    require(len(acceptance) == 8 and all('self-consistency: PASS' in r for r in acceptance), 'K row')
    verdict = '=== SELF-CONSISTENCY PASS ==='
    require(verdict in log, 'verdict absent')
    rows = re.findall(r'^\[expert-host-slru\] key=(\d+:\d+:\d+) bytes=(\d+) slot=(\d+) hit=(true|false) victim=(-|\d+:\d+:\d+)$', log, re.M)
    require(rows and len(rows) == log.count('[expert-host-slru]'), 'host trace')
    seen = set()
    misses = rereads = evictions = 0
    for key, size, slot, hit, victim in rows:
        require(int(size) <= 860160 and int(slot) < 16, 'host extent')
        misses += hit == 'false'
        rereads += hit == 'false' and key in seen
        evictions += victim != '-'
        seen.add(key)
    physical = re.search(r'physical_reads=(\d+) owner_close=Ok\(\(\)\)', log)
    require(physical and int(physical[1]) == misses, 'reads/drain')
    gpu = re.search(r'\[expert-gpu-slru\] slots=(\d+) allocated_bytes=(\d+) evictions=(\d+)', log)
    require(gpu and int(gpu[2]) == int(gpu[1]) * 860168, 'GPU extents')
    require(int(gpu[1]) == 9986 and int(gpu[3]) > 0 and rereads > 0, 'pressure absent')
    result = {'case': path.name, 'n': 1, 'verdict': verdict, 'gpu_evictions': int(gpu[3]),
              'host_evictions': evictions, 'physical_reads': misses, 'rereads': rereads,
              'gpu_slots': int(gpu[1]), 'ended_utc': capture['ended_utc']}
    return result, tape, acceptance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cell', help='replay one receipt dir under --root only (self-test)')
    parser.add_argument('--root', type=Path, default=RAW10)
    args = parser.parse_args()
    if args.cell:
        result, tape, acceptance = replay(args.root / args.cell)
    else:
        status = json.loads((args.root / 'repeat-status.json').read_text())
        require(status['state'] == 'complete', 'repeat cell incomplete')
        directory = Path(status['directory'])
        require(directory.name == str(directory), 'escaping receipt')
        result, tape, acceptance = replay(args.root / directory)
        post = json.loads((args.root / 'binary-postcheck.json').read_text())
        require(post['binary_sha256'] == BINARIES, 'post-cell binary mismatch')
        require(datetime.datetime.fromisoformat(result['ended_utc']) <=
                datetime.datetime.fromisoformat(post['checked_utc']), 'postcheck preceded cell')
        result['attempts'] = status['attempts']
    control = DAY9.replay('default-spec-off')
    require((tape, acceptance) == control[1:], 'tape/acceptance differ from day-nine native control')
    prior = DAY9.replay('pressure-spec-on')
    require((tape, acceptance) == prior[1:], 'tape/acceptance differ from day-nine pressure-spec-on')
    result['day9_pressure_spec_on'] = {k: prior[0][k] for k in
                                       ['gpu_evictions', 'host_evictions', 'physical_reads', 'rereads']}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
