#!/usr/bin/env python3
"""Replay target-card evidence; CPU success never fills missing native cells."""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
LANE = ROOT / 'research/spill-c-20260919'
RAW = LANE / 'pro-single-day9'
SPEC = importlib.util.spec_from_file_location('day7', LANE / 'verify-day7.py')
DAY7 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY7)
require = DAY7.require
CASES = ['default-gen-off', 'default-spec-off', 'default-gen-on',
         'default-spec-on-retry', '8g-gen-on-final', '8g-spec-on-final',
         '8g-gen-off-final', '8g-spec-off-final']


def replay(case):
    path = RAW / case
    capture = json.loads((path / 'command.capture.json').read_text())
    DAY7.hashes(path, capture)
    require(capture['exit_code'] == 0 and not capture['timed_out'] and
            capture['parse_error'] is None, 'failed cell: ' + case)
    require(capture['qualification'] is False and
            capture['status'] == 'executed-not-qualified', 'collector is not qualification')
    require(capture['gpu_telemetry']['interval_ms'] == 250, 'telemetry cadence')
    require(capture['gpu_power_limits'] == [{'device': '0', 'power.limit': '600.00 W',
            'power.max_limit': '600.00 W'}], 'power envelope changed')
    require(json.loads((path / 'lock.json').read_text()) ==
            {'rig': 'pro-single', 'lock': '/tmp/memra-gpu.lock', 'acquired': True}, 'lock')
    gate = 'spec' if 'spec' in case else 'gen'
    banked = '-on' in case
    argv = ['env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32']
    if case.startswith('8g-'):
        argv.append('MEMRA_MOE_SLOTS=9986')
    argv += [f'/root/wt-c/target/release/run-{gate}',
             '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', '55', '88', '13']
    if banked:
        argv.append('--experts-via-tier')
    require(capture['command'] == argv, 'command identity changed')
    log = (path / 'command.log').read_text()
    tape = DAY7.tokens(log)
    acceptance = re.findall(r'acceptance: ([^\n]+)', log)
    if gate == 'gen':
        verdict = 'prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH'
    else:
        verdict = '=== SELF-CONSISTENCY PASS ==='
        require(re.findall(r'\[generate_spec K=(\d)\]', log) == list('12345678'), 'K ladder')
        require(len(acceptance) == 8 and all('self-consistency: PASS' in r for r in acceptance), 'K row')
    require(verdict in log, 'verdict absent')
    result = {'case': case, 'verdict': verdict}
    if banked:
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
        if case.startswith('8g-'):
            require(int(gpu[1]) == 9986 and int(gpu[3]) > 0 and rereads > 0, 'pressure absent')
        result.update(gpu_evictions=int(gpu[3]), host_evictions=evictions,
                      physical_reads=misses, rereads=rereads, gpu_slots=int(gpu[1]))
    return result, tape, acceptance


def cpu():
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out = LANE / 'raw/day9-cpu' / stamp
    out.mkdir(parents=True)
    commands = {
        'fmt': ['cargo', 'fmt', '--all', '--', '--check'],
        'mac-check': ['cargo', 'check', '-p', 'memra-tier', '--offline', '--all-targets'],
        'linux-check': ['cargo', 'check', '-p', 'memra-tier', '--offline', '--all-targets', '--target', 'x86_64-unknown-linux-gnu'],
        'tests': ['cargo', 'test', '-p', 'memra-tier', '--offline', '--no-fail-fast'],
        'clippy': ['cargo', 'clippy', '-p', 'memra-tier', '--offline', '--all-targets', '--', '-D', 'warnings'],
        'engine-linux-clippy': ['cargo', 'clippy', '-p', 'memra-engine', '--offline', '--lib', '--bin', 'run-gen', '--bin', 'run-spec', '--target', 'x86_64-unknown-linux-gnu', '--', '-D', 'warnings'],
        'diff': ['git', 'diff', '--check'],
        'flags': ['bash', 'tools/check-flags.sh'],
    }
    results = []
    for name, argv in commands.items():
        env = os.environ.copy()
        if name == 'engine-linux-clippy':
            env.update(DOCS_RS='1', MEMRA_MMQ_ARCHIVE_HASH='0000000000000000')
        path = out / (name + '.log')
        with path.open('wb') as log:
            run = subprocess.run(argv, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=300)
        results.append({'name': name, 'command': argv, 'exit': run.returncode,
                        'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        print(name, run.returncode, flush=True)
    (out / 'checks.json').write_text(json.dumps({'source': subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(), 'checks': results,
        'engine_scope': 'Linux compile only, DOCS_RS placeholders'}, indent=2) + '\n')
    require(all(r['exit'] == 0 for r in results), 'CPU failure; inspect raw logs')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--case', choices=CASES)
    args = parser.parse_args()
    if args.cpu:
        cpu()
        return
    require((RAW / 'build/build.exit').read_text().strip() == '0', 'native build failed')
    require((RAW / 'build/source.commit').read_text().strip() ==
            '148e7f0e9994a1c35dd3e0891dae559c377561e0', 'source identity')
    results = {case: replay(case) for case in ([args.case] if args.case else CASES)}
    if not args.case:
        for gate in ['gen', 'spec']:
            control = results[f'default-{gate}-off']
            for name, row in results.items():
                if gate in name:
                    require(row[1:] == control[1:], 'same-card tapes/acceptance differ: ' + name)
    print(json.dumps([row[0] for row in results.values()], indent=2))


if __name__ == '__main__':
    main()
