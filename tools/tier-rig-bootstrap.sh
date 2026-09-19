#!/usr/bin/env bash
# Send this script over an operator-owned SSH identity; never embed credentials.
# BRANCH=lane/spill-integ-... bash tier-rig-bootstrap.sh [--dry-run --out DIR]
set -euo pipefail
exec python3 - "$@" <<'PY'
"""Fresh non-serving 5090 bootstrap. Dry-run stubs external effects, not checks."""
import argparse
import datetime
from decimal import Decimal, InvalidOperation
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--dry-run', action='store_true')
p.add_argument('--out', type=Path, help='new receipt directory (required for dry-run)')
p.add_argument('--repo', type=Path)
p.add_argument('--resume', action='store_true', help='read prior receipt; revalidate in a new attempt')
p.add_argument('--provider', choices=['auto', 'runpod', 'vast', 'other'], default='auto')
p.add_argument('--persistent-root', type=Path, help='operator-confirmed provider persistent volume')
p.add_argument('--nvme-root', type=Path, help='candidate local NVMe mount; verified before use')
p.add_argument('--set-power', action='store_true', help='request expected limit; refusal is diagnostic')
p.add_argument('--hourly-cost', type=Decimal, help='optional private operator rate; never publish raw')
p.add_argument('--expected-power', type=int, default=600)
p.add_argument('--gap-seconds', type=int, default=60)
p.add_argument('--jobs', type=int, default=4)
a = p.parse_args()
minimum_commit = '020d20479cd686835c0fb7743040947d0fc2723b'
branch = os.environ.get('BRANCH', '')
if not re.fullmatch(r'lane/spill-[A-Za-z0-9._/-]+', branch) or '..' in branch or branch.endswith('/'):
    p.error('BRANCH must explicitly name a lane/spill-* remote branch; main/default forbidden')
if a.gap_seconds < 60 or a.expected_power < 600 or a.jobs < 1:
    p.error('gap >=60 seconds, power >=600 W and jobs >=1 required')
if a.dry_run and a.out is None:
    p.error('--dry-run requires --out (never writes a live checkout)')
provider = a.provider
if provider == 'auto':
    provider = 'runpod' if os.environ.get('RUNPOD_POD_ID') else ('vast' if os.environ.get('CONTAINER_ID') else 'other')
if a.resume and a.out is None:
    p.error('--resume requires --out pointing to the prior receipt directory')
if a.hourly_cost is not None and (not a.hourly_cost.is_finite() or a.hourly_cost < 0):
    p.error('--hourly-cost must be finite and nonnegative')
if provider == 'vast' and a.persistent_root is None:
    p.error('Vast persistence is offer-specific: --persistent-root must be operator-confirmed')
persistent = (a.persistent_root or (Path('/workspace') if provider == 'runpod' else Path.home())).resolve()
repo = a.repo.resolve() if a.repo else persistent / 'memra-spill'
instance = os.environ.get('RUNPOD_POD_ID') if provider == 'runpod' else (os.environ.get('CONTAINER_ID') if provider == 'vast' else None)
bootstrap_start = time.monotonic_ns()
utc = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
host = re.sub('[^A-Za-z0-9_.-]', '_', os.uname().nodename)
# Raw host identifiers stay machine-local; sanitize before publishing evidence.
report = {'schema_version': 1, 'kind': 'cpu-stub' if a.dry_run else 'rig-bootstrap',
          'status': 'incomplete', 'branch': branch, 'minimum_commit': minimum_commit, 'started_utc': utc,
          'expected_power_w': a.expected_power, 'gap_seconds': a.gap_seconds,
          'lock': '/tmp/memra-5090.lock', 'qualification': False, 'steps': [],
          'provider': provider, 'provider_instance_id': instance,
          'hourly_cost': str(a.hourly_cost) if a.hourly_cost is not None else None,
          'private_receipt': True}
CUDA = r'''
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"ERROR: %s: %s\n",#x,cudaGetErrorString(e)); return 2; } } while(0)
int main() {
  int count=0; CHECK(cudaGetDeviceCount(&count));
  if(count!=1) { std::fprintf(stderr,"ERROR: expected exactly one CUDA device, got %d\n",count); return 3; }
  CHECK(cudaSetDevice(0));
  const size_t bytes=size_t(8)*1024*1024*1024, chunk=4*1024*1024;
  unsigned char *d=nullptr; CHECK(cudaMalloc((void**)&d,bytes));
  CHECK(cudaMemset(d,0xa5,bytes)); CHECK(cudaDeviceSynchronize());
  std::vector<unsigned char> host(chunk);
  for(size_t offset=0;offset<bytes;offset+=chunk) {
    CHECK(cudaMemcpy(host.data(),d+offset,chunk,cudaMemcpyDeviceToHost));
    for(size_t i=0;i<chunk;i++) if(host[i]!=0xa5) {
      std::fprintf(stderr,"ERROR: readback mismatch at %zu\n",offset+i); return 4;
    }
  }
  CHECK(cudaFree(d));
  std::puts("ACCEPT: 8589934592 bytes cudaMalloc+memset+full-readback MATCH");
  return 0;
}
'''

def check(ok, message):
    if not ok:
        raise RuntimeError(message)

# Persist from the first command; SIGKILL/spot loss must not strand evidence in /tmp.
# A resume never reuses stale GPU acceptance: every hardware/source check runs again.
destination = a.out.resolve() if a.out else persistent/'spill-bootstrap-receipts'/f'{host}-{utc}'
if a.resume:
    previous_path = destination/'BOOTSTRAP.json'
    previous = json.loads(previous_path.read_text())
    check(previous['branch'] == branch and previous['kind'] == report['kind'], 'resume branch/class mismatch')
    report['resume_from'] = {'path': str(previous_path), 'sha256': hashlib.sha256(previous_path.read_bytes()).hexdigest(),
                             'last_step': previous['steps'][-1]['name'] if previous['steps'] else None}
    destination = destination/'attempts'/utc
if destination.exists():
    p.error('--out already exists; use --resume to preserve evidence in a new attempt')
destination.mkdir(parents=True, exist_ok=False)
scratch = destination
if True:
    def checkpoint(event):
        row = {'kind': 'BOOTSTRAP', 'event': event, 'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'qualification': False, 'last_step': report['steps'][-1] if report['steps'] else None,
               'active_step': report.get('active_step')}
        with (destination/'BOOTSTRAP.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n'); stream.flush(); os.fsync(stream.fileno())
        temporary = destination/'BOOTSTRAP.partial'
        with temporary.open('w') as stream:
            json.dump(report, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        temporary.replace(destination/'BOOTSTRAP.json')
    checkpoint('start')
    def run(label, argv, stub='', timeout=120, allowed=False, cwd=None):
        path = scratch / (label + '.log')
        start = time.monotonic_ns()
        start_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['active_step'] = {'name': label, 'command': argv, 'started_utc': start_utc, 'raw_log': path.name}
        checkpoint('cell-start')
        if a.dry_run:
            path.write_text(stub)
            code = 0
        else:
            with path.open('xb') as log:
                try:
                    proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
                                            cwd=cwd, start_new_session=True)
                    try:
                        code = proc.wait(timeout=timeout)
                    except subprocess.TimeoutExpired:
                        import signal
                        os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                        log.write(b'ERROR: bootstrap command timed out\n'); code = 124
                except OSError as error:
                    log.write(('ERROR: ' + str(error) + '\n').encode()); code = 127
        # Never parse a pipe: raw command output is closed first.
        text = path.read_text(errors='replace')
        report['steps'].append({'name': label, 'command': argv, 'exit_code': code,
                                'duration_ns': time.monotonic_ns()-start,
                                'started_utc': start_utc, 'ended_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                'provider_instance_id': instance,
                                'estimated_cost': str(a.hourly_cost * Decimal(time.monotonic_ns()-start) / Decimal(3600000000000)) if a.hourly_cost is not None else None,
                                'stubbed': a.dry_run, 'raw_log': path.name,
                                'failure_quote': next((line for line in text.splitlines() if re.search(
                                    r'ERROR|error:|fatal|panic|out of memory|CUDA_ERROR', line)),
                                    'died, cause unknown — repro needed') if code != 0 else None,
                                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        report.pop('active_step', None)
        checkpoint('cell-end')
        print(f'{label}: exit {code}', flush=True)
        check(code == 0 or allowed, f'{label} failed, exit {code}; exact output: {path.name}')
        return text if code == 0 else ''

    def version(text, pattern):
        m = re.search(pattern, text)
        return tuple(map(int, m.groups())) if m else None

    try:
        check(a.dry_run or sys.platform == 'linux', 'Linux VM/container required')
        distro = run('distro', ['cat', '/etc/os-release'], 'ID=ubuntu\nVERSION_ID="24.04"\n')
        report['distro'] = distro
        # In the FIRST minute, before installs/builds. NVML is inventory, not acceptance.
        power = run('power', ['nvidia-smi', '--query-gpu=name,memory.total,power.limit,power.max_limit',
                             '--format=csv,noheader,nounits'], 'NVIDIA GeForce RTX 5090, 32607, 600.00, 600.00\n')
        rows = [line.split(',') for line in power.splitlines() if line.strip()]
        check(len(rows) == 1 and len(rows[0]) == 4 and 'RTX 5090' in rows[0][0], 'exactly one RTX 5090 required')
        try:
            memory, limit, maximum = map(lambda s: Decimal(s.strip()), rows[0][1:])
            check(all(x.is_finite() for x in (memory,limit,maximum)) and memory >= 31000
                  and maximum >= a.expected_power,
                  'power.max_limit must meet expected power; >=31,000 MiB required')
        except InvalidOperation:
            raise RuntimeError('unparseable power/memory query; no acceptance')
        report['power_verbatim'] = power
        report['power_limit_restricted'] = limit < a.expected_power
        if a.set_power:
            with open('/tmp/memra-5090.lock', 'a') as power_lock:
                fcntl.flock(power_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                occupied = run('compute-before-power', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], 'pid, process_name, used_memory [MiB]\n')
                check(len(occupied.strip().splitlines()) == 1 and occupied.startswith('pid,'),
                      'GPU occupied or inventory malformed; refuse power request')
                run('set-power', ['nvidia-smi', '-pl', str(a.expected_power)], allowed=True)
                report['power_set_exit'] = report['steps'][-1]['exit_code']
                after = run('power-after', ['nvidia-smi', '--query-gpu=power.limit,power.max_limit', '--format=csv,noheader,nounits'], '600.00, 600.00\n')
                values = [Decimal(v.strip()) for v in after.strip().split(',')]
                check(len(values) == 2 and all(v.is_finite() for v in values) and values[1] >= a.expected_power,
                      'invalid power.max_limit after request')
                report['power_after_verbatim'] = after
                report['power_limit_restricted'] = values[0] < a.expected_power
        # Container policy can deny -pl: retain current limit and do not claim expected power.
        # Neither systemd nor service/driver modifications are needed.
        # Verify prerequisites; install only ordinary distro packages, never GPU drivers.
        packages = ['git', 'build-essential', 'pkg-config', 'libssl-dev', 'numactl', 'util-linux', 'curl', 'ca-certificates']
        tools = ('git', 'g++', 'make', 'pkg-config', 'numactl', 'flock', 'lsblk', 'findmnt', 'curl')
        missing = [t for t in tools if not shutil.which(t)] if not a.dry_run else ['numactl']
        run('openssl-development', ['pkg-config', '--exists', 'openssl'], allowed=True)
        if report['steps'][-1]['exit_code'] != 0:
            missing.append('openssl development libraries')
        if missing:
            check('ID=ubuntu' in distro or 'ID=debian' in distro, 'missing '+','.join(missing)+'; auto-install supports Ubuntu/Debian only')
            privilege = [] if a.dry_run or os.geteuid() == 0 else ['sudo', '-n']
            run('apt-update', privilege + ['apt-get', 'update'], timeout=300)
            run('apt-packages', privilege + ['apt-get', 'install', '-y'] + packages, timeout=600)
        run('storage-df', ['df', '-hT'], 'STUB df: no storage qualification\n')
        run('storage-lsblk', ['lsblk', '-J', '-o', 'NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS'], '{"blockdevices": []}\n')
        report['paths'] = {'persistent_root': str(persistent), 'repo': str(repo),
                           'build': str(repo/'target'), 'scratch': None, 'nvme': None,
                           'persistence': 'operator/provider contract, not proven by mount name'}
        if a.nvme_root:
            mount = a.nvme_root.resolve()
            device = run('nvme-mount', ['findmnt', '-n', '-o', 'SOURCE', '-T', str(mount)], '/dev/nvme0n1\n').strip()
            ancestry = run('nvme-ancestry', ['lsblk', '-s', '-r', '-n', '-o', 'KNAME', device], 'nvme0n1\n')
            check(any(re.fullmatch(r'nvme[0-9]+n[0-9]+(?:p[0-9]+)?', line.strip()) for line in ancestry.splitlines()),
                  'local NVMe ancestry not proven; no overlay/network fallback')
            check(a.dry_run or mount.is_dir(), 'NVMe mount absent')
            report['paths'].update(scratch=str(mount), nvme=str(mount))
        checkpoint('paths-selected')
        # Follow engine build.rs: explicit override first, then existing CUDA root;
        # otherwise newest runnable release among PATH and /usr/local/cuda*.
        check(os.environ.get('MEMRA_CUDA_ARCH', '120a') == '120a', 'MEMRA_CUDA_ARCH must be unset or 120a for this rig')
        explicit = os.environ.get('MEMRA_NVCC')
        if explicit is None:
            for key in ('CUDA_HOME', 'CUDA_PATH', 'CUDA_ROOT'):
                value = os.environ.get(key)
                if value and (Path(value)/'bin/nvcc').is_file():
                    explicit = str(Path(value)/'bin/nvcc'); break
        candidates = [explicit] if explicit is not None else (
            [str(Path(d)/'nvcc') for d in os.get_exec_path() if d] +
            ['/usr/local/cuda/bin/nvcc'] + [str(q/'bin/nvcc') for q in Path('/usr/local').glob('cuda-*')])
        if a.dry_run:
            candidates = ['/stub/cuda-12.4/bin/nvcc', '/stub/cuda-13.2/bin/nvcc']
        ranked, seen = [], set()
        for i, candidate in enumerate(candidates):
            # A bare explicit MEMRA_NVCC is resolved by Command through PATH too.
            q = Path(shutil.which(candidate) or candidate) if '/' not in candidate else Path(candidate)
            if not a.dry_run and not q.is_file():
                continue
            q = str(q.resolve())
            if q in seen: continue
            seen.add(q)
            text = run(f'nvcc-version-{i}', [q, '--version'],
                       'Cuda compilation tools, release '+('12.4' if '12.4' in q else '13.2')+', V13.2.0\n', allowed=True)
            v = version(text, r'release (\d+)\.(\d+)')
            if v: ranked.append((v,q))
        ranked.sort(key=lambda x:x[0], reverse=True)
        check(ranked and ranked[0][0] >= (13,0),
              'missing usable CUDA nvcc >=13.x at build.rs-selected path; install NVIDIA cuda-toolkit-13-x (not distro nvidia-cuda-toolkit 12.x); drivers unchanged')
        nvcc = ranked[0][1]
        report['nvcc'] = nvcc
        # Build with the exact compiler/architecture just accepted, even if rustup changes PATH.
        os.environ['MEMRA_NVCC'] = nvcc
        os.environ['MEMRA_CUDA_ARCH'] = '120a'
        # `--list-gpu-arch` prints base arches only (compute_120), never the arch-specific
        # `compute_120a` variant, so a list check is a false negative on every toolkit (seen on
        # CUDA 13.1.2, 2026-09-19). The `cuda-compile` step below with `-arch=sm_120a` is the
        # real proof; here we only require the base Blackwell arch to be listed.
        arches = run('nvcc-arches', [nvcc, '--list-gpu-arch'], 'compute_120\n')
        check('compute_120' in arches.split(), 'selected nvcc lacks compute_120 (Blackwell); install CUDA 13.x toolkit')
        source = scratch/'accept.cu'; source.write_text(CUDA)
        binary = scratch/'accept'
        run('cuda-compile', [nvcc, '-arch=sm_120a', str(source), '-o', str(binary)], timeout=180)
        # Actual lock held even in dry-run; all CUDA consumers serialized, including gap.
        with open('/tmp/memra-5090.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            processes = run('compute-before', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], 'pid, process_name, used_memory [MiB]\n')
            check(len(processes.strip().splitlines()) == 1 and processes.startswith('pid,'),
                  'GPU occupied or compute inventory malformed; refuse acceptance interference')
            try:
                for attempt in (1,2):
                    answer = run(f'accept-{attempt}', [str(binary)], 'ACCEPT: 8589934592 bytes cudaMalloc+memset+full-readback MATCH\n', timeout=180)
                    check('ACCEPT: 8589934592 bytes cudaMalloc+memset+full-readback MATCH' in answer, 'missing actual full readback marker')
                    if attempt == 1:
                        run('accept-gap', ['sleep', str(a.gap_seconds)], timeout=a.gap_seconds+10)
            finally:
                run('compute-after', ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv'], 'pid, process_name, used_memory [MiB]\n')
        report['cuda_acceptance'] = 'stubbed-not-hardware' if a.dry_run else 'two-full-readbacks'
        # Stable is explicit for every build; do not silently use an old repo override.
        cargo_bin = str(Path.home()/'.cargo/bin')
        os.environ['PATH'] = cargo_bin + os.pathsep + os.environ.get('PATH','')
        if a.dry_run or not shutil.which('rustup'):
            installer = scratch/'rustup-init.sh'
            run('rustup-download', ['curl','--fail','--show-error','--location','--proto','=https','--tlsv1.2',
                                    'https://sh.rustup.rs','--output',str(installer)], timeout=180)
            run('rustup-install', ['sh',str(installer),'-y','--profile','minimal','--default-toolchain','none'], timeout=600)
        run('rustup-stable', ['rustup', 'toolchain', 'install', 'stable', '--profile', 'minimal'], timeout=600)
        rust = run('rust-version', ['rustc', '+stable', '--version'], 'rustc 1.97.0 (stub)\n')
        check((version(rust, r'rustc (\d+)\.(\d+)\.(\d+)') or (0,)) >= (1,97,0), 'stable Rust >=1.97.0 required')
        # Resolve the requested head before clone. An unknown branch is NEVER main.
        remote = 'https://github.com/avifenesh/memra'
        refs = run('remote-branch', ['git', 'ls-remote', '--exit-code', '--heads', remote, 'refs/heads/'+branch], '1'*40+'\trefs/heads/'+branch+'\n')
        check(re.fullmatch('[0-9a-f]{40}\\trefs/heads/'+re.escape(branch)+'\\n?', refs) is not None, 'remote branch missing/ambiguous')
        expected = refs.split()[0]
        if a.dry_run or not repo.exists():
            run('clone', ['git', 'clone', '--single-branch', '--branch', branch, remote, str(repo)], timeout=900)
        else:
            check((repo/'.git').exists(), 'existing repo path is not a checkout')
            clean = run('worktree-clean', ['git', 'status', '--porcelain', '--untracked-files=no'], cwd=repo)
            check(not clean.strip(), 'existing checkout dirty; preserve it and refuse bootstrap')
            origin = run('origin', ['git','remote','get-url','origin'], cwd=repo).strip().removesuffix('.git')
            check(origin == remote, 'existing origin differs; refuse')
            run('fetch-branch', ['git','fetch','origin','refs/heads/'+branch], cwd=repo, timeout=900)
            run('checkout', ['git','checkout','--detach', expected], cwd=repo)
        tip = run('source-sha', ['git','rev-parse','HEAD'], expected+'\n', cwd=None if a.dry_run else repo).strip()
        check(tip == expected, 'remote branch changed during clone/fetch; rerun for exact head')
        report['source_commit'] = tip
        run('minimum-source', ['git', 'merge-base', '--is-ancestor', minimum_commit, tip],
            cwd=None if a.dry_run else repo)
        if destination is None:
            destination = repo/'research/spill-d-20260919/raw'/f'{host}-{utc}'
            destination.mkdir(parents=True, exist_ok=False)
        # Keep the exact standalone acceptance executable/source to re-run D1 local checks.
        if a.dry_run: binary.write_bytes(b'CPU STUB NOT EXECUTABLE\n')
        # Source and binary already reside in the durable receipt directory.
        report['accept_binary_sha256'] = hashlib.sha256(binary.read_bytes()).hexdigest()
        if a.dry_run:
            commands = [['nvidia-smi','topo','-m'], ['nvidia-smi','-q'], ['nvidia-smi','topo','-p2p','r'], ['nvidia-smi','topo','-p2p','w'], ['lscpu','--extended=CPU,NODE,SOCKET'], ['numactl','-H'], ['lscpu'], ['df','-hT'], ['lsblk','-J','-o','NAME,TYPE,SIZE,ROTA,TRAN,MOUNTPOINTS'], ['free','-g'], ['findmnt','-J','-T','/scratch']]
            fixture = {}
            for i, command in enumerate(commands):
                text = run(f'topology-{i}',command,'STUB inventory, not hardware\n')
                fixture[' '.join(command)] = {'exit_code':0,'stdout':text,'stderr':'','timed_out':False}
            spec = importlib.util.spec_from_file_location('topology', Path.cwd()/'tools/tier-topology.py')
            topology = importlib.util.module_from_spec(spec); spec.loader.exec_module(topology)
            topo = topology.probe(fixture, storage=True)
            (destination/'TOPOLOGY.json').write_text(json.dumps(topo,indent=2)+'\n')
        else:
            run('topology', ['python3', str(repo/'tools/tier-topology.py'), '--storage', '--out', str(destination/'TOPOLOGY.json')])
        # Explicit crates/targets: no --all-bins (would include unrelated paused lanes).
        builds = [
            ['cargo','+stable','build','--release','--locked','-j',str(a.jobs),'-p','memra-tier','-p','memra-kv'],
            ['cargo','+stable','build','--release','--locked','-j',str(a.jobs),'-p','memra-server','--bin','memra-server'],
            ['cargo','+stable','build','--release','--locked','-j',str(a.jobs),'-p','memra-engine','--bin','storage-bench','--bin','pp-transport-smoke','--bin','qwen4exp_gpu_gate','--bin','run-gen','--bin','run-spec'],
            ['cargo','+stable','test','--release','--locked','-j',str(a.jobs),'-p','memra-engine','--lib','--no-run'],
        ]
        report['builds'] = builds
        for i, cmd in enumerate(builds):
            run(f'build-{i}', cmd, 'STUB build; native worker.rs NOT compiled\n', timeout=3600, cwd=None if a.dry_run else repo)
        report['release_binaries'] = {}
        for name in ('storage-bench','pp-transport-smoke','qwen4exp_gpu_gate','run-gen','run-spec','memra-server'):
            executable = repo/'target/release'/name
            check(a.dry_run or executable.is_file(), 'build did not produce '+str(executable))
            if a.dry_run:
                report['release_binaries'][name] = {'status':'stubbed-not-built'}
            else:
                h = hashlib.sha256()
                with executable.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024*1024), b''): h.update(block)
                report['release_binaries'][name] = {'sha256':h.hexdigest()}
        wrapper = destination/'locked-run.sh'
        wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec flock --close -n -x /tmp/memra-5090.lock "$@"\n')
        wrapper.chmod(0o755)
        report['status'] = 'dry-run-complete-not-qualified' if a.dry_run else 'bootstrap-complete-not-tier-qualified'
    except Exception as error:
        report['status'] = 'blocked'
        report['blocker'] = str(error)
        print('BLOCKED: '+str(error), file=sys.stderr)
    finally:
        report['ended_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        report['duration_ns'] = time.monotonic_ns() - bootstrap_start
        checkpoint('end')
        print('receipt: '+str(destination/'BOOTSTRAP.json'))
    sys.exit(0 if report['status'].startswith(('dry-run-complete','bootstrap-complete')) else 2)
PY
