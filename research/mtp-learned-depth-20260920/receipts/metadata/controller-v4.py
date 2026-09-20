"""Continue fixed full-head MTP comparisons, committing only complete paired sets."""
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path('/root/mtp-depth-records')
LANE = Path('/root/memra-mtp-depth/research/mtp-learned-depth-20260920')
BIN = Path('/root/target-learned-d/release')
SOURCE = 'a37e30a6dd9637ece727c040f12a115dd6d96517'
sys.path.insert(0, str(LANE))
import run_study as study

selected = {'qwen': [], 'gemma': []}


def status(**data):
    data.update(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                pid=os.getpid(), source_commit=SOURCE,
                completed_sets={f: len(v) for f, v in selected.items()})
    path = ROOT / 'status.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def wait_quiet(family, phase):
    status(state='waiting', phase=phase, family=family)
    quiet = None
    while quiet is None or time.monotonic() - quiet < 30:
        busy = study.gpu_processes()
        locks = subprocess.check_output(['lslocks', '--output', 'COMMAND,PATH'], text=True)
        building = any('cargo' in line and '.cargo-build-lock' in line for line in locks.splitlines())
        if busy or building:
            quiet = None
        elif quiet is None:
            quiet = time.monotonic()
        time.sleep(5)


def validate_set(folder, family, cycle):
    base = 20261200 if family == 'qwen' else 20261300
    seed = base + cycle
    rows = [r for r in json.loads((folder / 'runs.json').read_text()) if r['seed'] == seed]
    assert [r['arm'] for r in rows] == list(study.select_cycles(start=cycle, stop=cycle+1)[0][1])
    assert all(r['elapsed_s'] >= 60 and not r['correctness_only'] for r in rows)
    assert not any((folder / f'{seed}-{r["arm"]}.contamination.txt').exists() for r in rows)
    study.audit_control_ids(folder, seed, ['measured'], turns=8 if family == 'qwen' else 16)
    return {'cycle': cycle, 'seed': seed, 'receipt_dir': folder.name, 'records': rows}


def save_family(family):
    records = [r for item in selected[family] for r in item['records']]
    (ROOT / f'{family}-selected-sets.json').write_text(json.dumps(selected[family], indent=2) + '\n')
    if len(selected[family]) == 8:
        assert [x['cycle'] for x in selected[family]] == list(range(8))
        (ROOT / f'{family}-final-summary.json').write_text(json.dumps(study.summarize(records), indent=2) + '\n')


def run_attempt(family, kind, cycle, attempt):
    key = f'cycle-{cycle}' if cycle is not None else kind
    name = f'{family}-mtp-{key}-attempt-{attempt}-v4'
    output = ROOT / name
    cache = Path('/data/models/qwen-depth-20260920' if family == 'qwen'
                 else '/data/models/gemma-depth-20260920')
    manifest = cache / 'mtp-depth-artifacts.json'
    shutil.copyfile(LANE / f'{family}-artifacts.lock.json', manifest)
    wait_quiet(family, name)
    cmd = ['python3', str(LANE / 'run_study.py'), '--family', family,
           '--binary', str(BIN / ('mtp-depth-study' if family == 'qwen' else 'gemma-depth-study')),
           '--target', str(cache / 'target.gguf'), '--artifact-manifest', str(manifest),
           '--source-commit', SOURCE, '--out', str(output), '--wait-lock',
           '--lock', '/tmp/memra-5090.lock', '--seed', '20261200' if family == 'qwen' else '20261300',
           '--workload', str(LANE / ('workload-short.txt' if kind == 'short' else 'workload-code.txt')),
           '--max-new', '256' if kind == 'short' else '1024',
           '--ctx', '16384' if kind == 'short' else '49152']
    if family == 'gemma':
        cmd += ['--draft', str(cache / 'assistant.gguf')]
    if cycle is None:
        cmd += ['--gate']
    else:
        cmd += ['--cycle-start', str(cycle), '--cycle-stop', str(cycle+1)]
    status(state='running', family=family, phase=name, cycle=cycle, attempt=attempt, command=cmd)
    driver_log = ROOT / (name + '.driver.log')
    with driver_log.open('w') as log:
        result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    if output.exists():
        shutil.copyfile(driver_log, output / 'driver.log')
        subprocess.run(['tar', '-czf', str(ROOT / (name + '.tar.gz')), '-C', str(ROOT), name], check=True)
    if result.returncode:
        message = driver_log.read_text()
        if any(s in message for s in ['concurrent GPU processes detected',
                                      'GPU became busy before', 'GPU has an existing workload']):
            print('REJECTED_GPU_INTERFERENCE', name, flush=True)
            return None
        raise RuntimeError(f'{name} failed; see {driver_log}')
    if cycle is not None:
        return validate_set(output, family, cycle)
    assert json.loads((output / 'summary.json').read_text())['status'] == 'greedy-identity-pass'
    return output


def run_clean(family, kind, cycle=None):
    for attempt in range(1, 4):
        result = run_attempt(family, kind, cycle, attempt)
        if result is not None:
            return result
    raise RuntimeError('three GPU-interrupted attempts for one set; an exclusive window is needed')


def main():
    prior = ROOT / 'qwen-mtp-oracle-v1'
    receipt = json.loads((prior / 'identity.json').read_text())
    assert receipt['binary_sha256'] == study.digest(BIN / 'run-spec')
    assert '=== SELF-CONSISTENCY PASS ===' in (prior / 'oracle.log').read_text()
    for name in ['qwen-mtp-short-v2', 'qwen-mtp-code-v2', 'qwen-mtp-sampled-v3']:
        folder = ROOT / name
        ident = json.loads((folder / 'identity.json').read_text())
        assert ident['binary_sha256'] == study.digest(BIN / 'mtp-depth-study')
        workload = LANE / ('workload-short.txt' if 'short' in name else 'workload-code.txt')
        assert ident['workload_sha256'] == study.digest(workload)
        if 'sampled' not in name:
            assert json.loads((folder / 'summary.json').read_text())['status'] == 'greedy-identity-pass'
    selected['qwen'] = [validate_set(ROOT / 'qwen-mtp-sampled-v3', 'qwen', cycle) for cycle in range(3)]
    save_family('qwen')
    for cycle in range(3, 8):
        selected['qwen'].append(run_clean('qwen', 'sampled', cycle))
        save_family('qwen')
    for kind in ['short', 'code']:
        run_clean('gemma', kind)
    for cycle in range(8):
        selected['gemma'].append(run_clean('gemma', 'sampled', cycle))
        save_family('gemma')
    status(state='complete')


if __name__ == '__main__':
    (ROOT / 'controller.pid').write_text(str(os.getpid()) + '\n')
    try:
        main()
    except BaseException as error:
        old = json.loads((ROOT / 'status.json').read_text()) if (ROOT / 'status.json').exists() else {}
        status(state='failed', phase=old.get('phase'), error=str(error))
        raise
