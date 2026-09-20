#!/usr/bin/env python3
"""Replay the two N=1 GPU bank budget cells against the sealed day-nine gen control."""
import argparse
import datetime
import importlib.util
import json
from pathlib import Path
import re
import subprocess

LANE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location('day9', LANE / 'verify-day9.py')
DAY9 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DAY9)
DAY7 = DAY9.DAY7
require = DAY7.require
RAW = LANE / 'pro-single-day10-budget'
RECORD = 860160
SLOT = RECORD + 8
MINIMUM = 8 * SLOT
ROOT = LANE.parents[1]
RUNTIME_SOURCE = '79353d53d7f69713177caf2e9830b48f9e600fa3'
BINARIES = {'run-gen': 'ec513d94dfed9c94ec419134406b0fe8be190242a6970ed8f5972a9795c2ea98',
            'run-spec': '0685dfa17d4835c0d98143ac75840b598db37ba84cb8e910ed29e4bd22aed1ac'}
FROZEN = {'run-gen': 'b1c4090cb32bfba5f65bddd64ccafe407516a51dd8142b77e73184e2ad3f6d22',
          'run-spec': 'ded88cc84bfd54319b8a6bd6ee08733fedfff4b75be769c47a0b9cb29bb6ba0d'}
REFUSED = re.compile(r'^REFUSED: experts-via-tier GPU bank budget cannot hold the eight-slot minimum '
                     r'\(requested (\d+), minimum (\d+), ceiling (\d+)\)$')
HOST_REFUSED = re.compile(r'^REFUSED: experts-via-tier host bank budget cannot hold one expert record '
                          r'\(requested (\d+), minimum (\d+), ceiling (\d+)\)$')
WRAPPER = ['python3', '/root/wt-c/research/spill-c-20260919/pressure-refusal.py']


def argv(budget, flag='--expert-bank-gpu-bytes'):
    return ['env', 'MEMRA_MOE_RESIDENT=0', 'MEMRA_NGEN=32', '/root/wt-c/target-day10/release/run-gen',
            '/root/artifacts/Qwen3.6-35B-A3B-UD-IQ4_XS.gguf', '55', '88', '13',
            '--experts-via-tier', f'{flag}={budget}']


def envelope(path, capture):
    DAY7.hashes(path, capture)
    require(capture['qualification'] is False, 'collector is not qualification')
    require(not capture['timed_out'] and capture['parse_error'] is None, 'timed out or unparsable')
    require(capture['gpu_telemetry']['interval_ms'] == 250, 'telemetry cadence')
    require(capture['gpu_power_limits'] == [{'device': '0', 'power.limit': '600.00 W',
            'power.max_limit': '600.00 W'}], 'power envelope changed')
    require(json.loads((path / 'lock.json').read_text()) ==
            {'rig': 'pro-single', 'lock': '/tmp/memra-gpu.lock', 'acquired': True}, 'lock')


def replay_refusal(path):
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 2 and capture['status'] == 'refused', 'not a collector refusal')
    require(capture['command'] == argv(7 * SLOT), 'command identity changed')
    log = (path / 'command.log').read_text()
    lines = log.splitlines()
    require(lines and lines[-1] == capture['failure_quote'], 'refusal is not the final line')
    match = REFUSED.match(lines[-1])
    require(match, 'refusal text changed')
    requested, minimum, ceiling = (int(v) for v in match.groups())
    require(requested == 7 * SLOT and minimum == MINIMUM and ceiling >= minimum, 'refusal arithmetic')
    for marker in ['[expert-host-slru]', '[expert-gpu-slru]', '[experts-via-tier] installed',
                   '[experts-via-tier] gpu_bank_budget', '\nloaded ', 'prompt tokens:', 'tokens: [', 'MATCH']:
        require(marker not in log, 'work after refusal: ' + marker.strip())
    return {'case': path.name, 'n': 1, 'verdict': lines[-1], 'requested_bytes': requested,
            'minimum_bytes': minimum, 'hard_ceiling_bytes': ceiling, 'ended_utc': capture['ended_utc']}


def replay_host_refusal(path):
    """The wrapper red arm passes the native token through verbatim; the native exit is 2."""
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 2 and capture['status'] == 'refused', 'not a collector refusal')
    require(capture['command'] == WRAPPER + argv(1, '--expert-bank-host-bytes'), 'command identity changed')
    log = (path / 'command.log').read_text()
    lines = log.splitlines()
    require(len(lines) >= 3 and lines[-1] == capture['failure_quote'], 'refusal is not the final line')
    require(lines[-2] == 'native_exit_code=2' and lines[-3] == lines[-1], 'wrapper did not pass the native token')
    match = HOST_REFUSED.match(lines[-1])
    require(match, 'refusal text changed')
    requested, minimum, ceiling = (int(v) for v in match.groups())
    require((requested, minimum, ceiling) == (1, RECORD, 256 * 1024 * 1024), 'refusal arithmetic')
    require('FAIL:' not in log and 'Error:' not in log, 'legacy error shape present')
    for marker in ['[expert-host-slru]', '[expert-gpu-slru]', '[experts-via-tier] installed',
                   '[experts-via-tier] gpu_bank_budget', '\nloaded ', 'prompt tokens:', 'tokens: [', 'MATCH']:
        require(marker not in log, 'work after refusal: ' + marker.strip())
    return {'case': path.name, 'n': 1, 'verdict': lines[-1], 'requested_bytes': requested,
            'minimum_bytes': minimum, 'ceiling_bytes': ceiling, 'ended_utc': capture['ended_utc']}


def replay_exact(path):
    capture = json.loads((path / 'command.capture.json').read_text())
    envelope(path, capture)
    require(capture['exit_code'] == 0 and capture['status'] == 'executed-not-qualified', 'failed cell')
    require(capture['command'] == argv(MINIMUM), 'command identity changed')
    log = (path / 'command.log').read_text()
    budget = re.search(r'^\[experts-via-tier\] gpu_bank_budget bytes=(\d+) slots=(\d+) hard_ceiling=(\d+)$', log, re.M)
    installed = re.search(r'^\[experts-via-tier\] installed artifact_sha256=([0-9a-f]{64}) host_slots=(\d+) max_expert_bytes=(\d+)$', log, re.M)
    require(budget and installed and budget.start() < installed.start(), 'budget line missing or after install')
    require((int(budget[1]), int(budget[2])) == (MINIMUM, 8) and int(budget[3]) >= MINIMUM, 'budget arithmetic')
    require(installed[1] == 'df27a780435b7b45c2597536112ea3cb091f8544c3d0c3318d9f4258b31f7adf'
            and (int(installed[2]), int(installed[3])) == (16, RECORD), 'install identity')
    verdict = 'prefill argmax=198  decode argmax=198  logit maxdiff=6.482e-1  MATCH'
    require(verdict in log, 'verdict absent')
    tape = DAY7.tokens(log)
    rows = re.findall(r'^\[expert-host-slru\] key=(\d+:\d+:\d+) bytes=(\d+) slot=(\d+) hit=(true|false) victim=(-|\d+:\d+:\d+)$', log, re.M)
    require(rows and len(rows) == log.count('[expert-host-slru]'), 'host trace')
    seen = set()
    misses = rereads = evictions = 0
    for key, size, slot, hit, victim in rows:
        require(int(size) <= RECORD and int(slot) < 16, 'host extent')
        misses += hit == 'false'
        rereads += hit == 'false' and key in seen
        evictions += victim != '-'
        seen.add(key)
    physical = re.search(r'physical_reads=(\d+) owner_close=Ok\(\(\)\)', log)
    require(physical and int(physical[1]) == misses, 'reads/drain')
    gpu = re.search(r'\[expert-gpu-slru\] slots=(\d+) allocated_bytes=(\d+) evictions=(\d+)', log)
    require(gpu and (int(gpu[1]), int(gpu[2])) == (8, MINIMUM), 'GPU extents')
    require(int(gpu[3]) > 0 and rereads > 0, 'pressure absent')
    result = {'case': path.name, 'n': 1, 'verdict': verdict, 'gpu_slots': 8, 'gpu_evictions': int(gpu[3]),
              'host_evictions': evictions, 'physical_reads': misses, 'rereads': rereads,
              'hard_ceiling_bytes': int(budget[3]), 'ended_utc': capture['ended_utc']}
    return result, tape


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=RAW)
    parser.add_argument('--cell', help='replay one receipt dir under --root only (self-test)')
    args = parser.parse_args()
    if args.cell:
        path = args.root / args.cell
        if args.cell.startswith('budget-refuse'):
            result = replay_refusal(path)
        elif args.cell.startswith('budget-host-refuse'):
            result = replay_host_refusal(path)
        else:
            result = replay_exact(path)[0]
        print(json.dumps(result, indent=2))
        return
    status = json.loads((args.root / 'budget-status.json').read_text())
    require(status['state'] == 'complete', 'budget cells incomplete')
    cells = {row['case']: row for row in status['completed']}
    require(set(cells) == {'budget-refuse-7', 'budget-exact-8', 'budget-host-refuse-1'}, 'cell set')
    for row in cells.values():
        require(Path(row['directory']).name == row['directory'], 'escaping receipt')
    build = json.loads((args.root / 'build.json').read_text())
    require(build['source'] == RUNTIME_SOURCE and build['binary_sha256'] == BINARIES
            and build['frozen_binary_sha256'] == FROZEN, 'build receipt identity')
    require(build['cargo_target_dir'] == '/root/wt-c/target-day10', 'build wrote into the frozen target dir')
    DAY7.hashes(args.root, {'path': 'build.log', 'bytes': (args.root / 'build.log').stat().st_size,
                            'sha256': build['build_log_sha256']})
    manifest = json.loads((args.root / 'binary-manifest.json').read_text())
    post = json.loads((args.root / 'binary-postcheck.json').read_text())
    for record in (manifest, post):
        require(record['binary_sha256'] == BINARIES, 'day-ten binary identity')
        require(record['frozen'] == {'runtime_source': '148e7f0e9994a1c35dd3e0891dae559c377561e0',
                                     'binary_sha256': FROZEN}, 'frozen day-nine binaries changed')
        # The checkout may carry later docs/script commits; the engine crates must be the built source.
        diff = subprocess.run(['git', 'diff', '--quiet', RUNTIME_SOURCE, record['worktree_head'], '--', 'crates/'],
                              cwd=ROOT)
        require(diff.returncode == 0, 'worktree engine sources differ from the built source')
    require(BINARIES['run-gen'] != FROZEN['run-gen'], 'day-ten build is the frozen build')
    refusal = replay_refusal(args.root / cells['budget-refuse-7']['directory'])
    host = replay_host_refusal(args.root / cells['budget-host-refuse-1']['directory'])
    exact, tape = replay_exact(args.root / cells['budget-exact-8']['directory'])
    control = DAY9.replay('default-gen-off')
    require(tape == control[1], 'tape differs from the day-nine native control')
    checked = datetime.datetime.fromisoformat(post['checked_utc'])
    for row in (refusal, host, exact):
        require(datetime.datetime.fromisoformat(row['ended_utc']) <= checked, 'postcheck preceded cell')
    refusal['attempts'] = cells['budget-refuse-7']['attempts']
    host['attempts'] = cells['budget-host-refuse-1']['attempts']
    exact['attempts'] = cells['budget-exact-8']['attempts']
    exact['tape_matches_day9_default_gen_off'] = True
    print(json.dumps({'gpu_refused': refusal, 'host_refused': host, 'exact': exact,
                      'runtime_source': RUNTIME_SOURCE}, indent=2))


if __name__ == '__main__':
    main()
