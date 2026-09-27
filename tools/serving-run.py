#!/usr/bin/env python3
"""Capture one complete reviewed serving scope; seal/verify all scopes for release.

Uses existing collectors and already built ELFs. Does not rent cards, acquire GPU
locks, rebuild, retry inference, choose smaller required scope or publish results.
"""
import argparse
import atexit
from dataclasses import asdict
import hashlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import uuid

# Establish the controller closure before any project import. -B alone still
# reads ignored bytecode. Imports below execute these captured source buffers.
_PY_CACHE = tempfile.TemporaryDirectory(prefix='memra-serving-controller-')
atexit.register(_PY_CACHE.cleanup)
sys.pycache_prefix = _PY_CACHE.name
sys.dont_write_bytecode = True
_ROOT = Path(__file__).resolve().parents[1]
_ENTRY = Path(__file__).absolute()
_FIXED = ('serving-run.py', 'qualify-release.py', 'release_qualification.py',
          'release_inputs.py', 'release_input_view.py', 'check_hardware_gate.py', 'release-coverage.py')


def _controller_files():
    if _ENTRY.is_symlink() or _ENTRY.resolve() != _ROOT/'tools/serving-run.py':
        raise RuntimeError('controller entry origin differs')
    paths = {_ROOT/'tools'/name for name in _FIXED} | set((_ROOT/'tools').glob('serving_*.py'))
    files = {}
    for path in sorted(paths):
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or path.resolve() != path:
            raise RuntimeError('controller helper origin is not a regular source file: '+str(path))
        raw = path.read_bytes()
        files[str(path.relative_to(_ROOT))] = (raw, '100755' if info.st_mode & 0o111 else '100644')
    return files


_CONTROLLER_FILES = _controller_files()


def _controller_source(repo, head):
    if Path(repo).resolve() != _ROOT:
        raise RuntimeError('controller entry/helper origins differ from requested repository')
    if not isinstance(head, str) or len(head) != 40 or any(c not in '0123456789abcdef' for c in head):
        raise RuntimeError('controller source requires an exact commit')
    tree = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-r', '-z', head, '--', 'tools'])
    expected = {}
    for row in tree.split(b'\0'):
        if row:
            metadata, name = row.split(b'\t'); mode, kind, blob = metadata.decode().split()
            name = name.decode()
            if name in {'tools/'+n for n in _FIXED} or (name.startswith('tools/serving_') and name.endswith('.py')):
                expected[name] = (mode, kind, blob)
    observed = {name: (mode, 'blob', hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest())
                for name, (raw, mode) in _CONTROLLER_FILES.items()}
    if observed != expected or _controller_files() != _CONTROLLER_FILES:
        raise RuntimeError('controller source bytes/modes differ from requested Git closure')


_MODULE_FILES = {Path(name).stem: name for name in _CONTROLLER_FILES if Path(name).stem.isidentifier()}
_MODULE_FILES['qualify_release'] = 'tools/qualify-release.py'
_PRELOADED = set(_MODULE_FILES) & sys.modules.keys()


class _ControllerImports(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in _MODULE_FILES:
            return importlib.util.spec_from_loader(fullname, self, origin=str(_ROOT/_MODULE_FILES[fullname]))
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        name = _MODULE_FILES[module.__name__]
        module.__file__ = str(_ROOT/name)
        module.__cached__ = None
        exec(compile(_CONTROLLER_FILES[name][0], module.__file__, 'exec'), module.__dict__)


_IMPORTS = _ControllerImports()
if __name__ == '__main__':
    # CLI preflight happens before even check_hardware_gate/release_qualification
    # can load. Import-only unit callers remain unable to capture preloaded code.
    early = argparse.ArgumentParser(add_help=False)
    early.add_argument('--repo', type=Path, default=_ROOT)
    early.add_argument('--expected-head')
    selected, _ = early.parse_known_args()
    if selected.expected_head is not None:
        _controller_source(selected.repo, selected.expected_head)
    if _PRELOADED:
        raise RuntimeError('controller helpers were loaded before source binding')
sys.meta_path.insert(0, _IMPORTS)

import release_qualification as q
import serving_run as run
from serving_capture import EvidenceStore, collect, encoded
from serving_http import capture_request
from serving_listener import ListenerOwnershipError, prove_listener
from serving_process import OwnedServer
from serving_release import json_object, require

spec = importlib.util.spec_from_loader('qualify_release', _IMPORTS)
producer = importlib.util.module_from_spec(spec); spec.loader.exec_module(producer)


def observed_controller_identity(repo, head, source):
    _controller_source(repo, head)
    require(not _PRELOADED, 'capture requires a fresh process with bound controller imports')
    for name, path in _MODULE_FILES.items():
        module = producer if name == 'qualify_release' else sys.modules.get(name)
        if module is not None:
            require(module.__loader__ is _IMPORTS and module.__file__ == str(_ROOT/path),
                    'loaded controller helper origin differs: '+name)
    observed = {name: {'sha256': q.digest(raw), 'mode': mode}
                for name, (raw, mode) in _CONTROLLER_FILES.items()}
    require(observed == run.controller_identity(repo, head, source), 'controller closure differs from built source')
    return observed


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_bytes(encoded(obj)); os.replace(temporary, path)


def reference(root, name):
    return {'path': q.safe_path(name), 'sha256': q.digest((root / name).read_bytes())}


def readiness(server, endpoint, http, store, scope):
    state = {'schema': 'memra-serving-startup-v1', 'state': 'starting', 'observations': [], 'listeners': [], 'errors': [], 'lifecycle': None}
    store.index(state)
    end = time.monotonic() + server._config['timeouts']['startup']
    try:
        while time.monotonic() < end:
            try:
                proof = prove_listener(server.identity, **endpoint, timeout=min(1, max(.001, end-time.monotonic())))
                state['listeners'].append(store.obj(asdict(proof)))
            except ListenerOwnershipError as error:
                state['listeners'].append(store.obj({'error': str(error), 'observed_ns': time.monotonic_ns()}))
                store.index(state); time.sleep(.02); continue
            samples = []
            owner = server.identity; receipt = server.receipt()
            identity = {'pid': owner.pid, 'start_identity': receipt['boot_id'] + ':' + owner.start_time}
            for path in ('/health', '/readyz'):
                require(time.monotonic() < end, 'startup deadline exceeded')
                observed = capture_request(request_id='startup-' + str(len(state['observations'])+1), method='GET', path=path, body=b'',
                    **endpoint, **{**http, 'wall_timeout': min(http['wall_timeout'], end-time.monotonic())})
                observed.update(path=path, method='GET', server_identity=identity)
                state['observations'].append(store.observation(observed)); store.index(state)
                require(observed['transport_error'] is None, 'startup HTTP transport failed')
                body = json_object(observed['body']); samples.append((observed, body))
            if all(row['status'] == 200 for row, body in samples):
                require([body.get('status') for row, body in samples] == ['ok', 'ready']
                        and all(scope['model'] in body.get('models', []) for row, body in samples), 'wrong startup model/status')
                proof = prove_listener(owner, **endpoint, timeout=min(1, max(.001, end-time.monotonic())))
                state['listeners'].append(store.obj(asdict(proof))); server.mark_ready(proof)
                state['state'] = 'ready'; store.index(state); return state
            store.index(state); time.sleep(.02)
        raise q.GateError('owned serving startup deadline exceeded')
    except BaseException as error:
        state['errors'].append(type(error).__name__ + ': ' + str(error)); state['state'] = 'failed'; store.index(state); raise


def capture_cell(required, scope, policy, args, build, uuids, index, output):
    output.mkdir(parents=True)
    runtime = {'server_identity': None, 'port': args.port, 'cwd': str(args.repo),
        'binary': str(args.build / 'target/release/memra-server'), 'output_path': str(output / 'capture/process/output.log'),
        'client_trace_key': uuid.uuid4().hex if required['scenario'] in run.PHASE else None}
    entry = {'id': required['id'], 'scope': required['scope'], 'scenario': required['scenario'], 'state': 'failed',
        'errors': [], 'runtime': runtime, 'capture': None, 'capture_root': f'cells/{index:03d}/capture',
        'log_identity': None, 'started_ns': time.monotonic_ns(), 'finished_ns': None, 'startup': None}
    server = None; startup = None; startup_store = None
    try:
        program, launch, plan, identities = run.materialize(scope, policy, runtime, uuids, build['binaries'])
        if required['scenario'] in run.GROUP:
            collect(plan, output / 'capture')
            raw = (output/'capture/capture.json').read_bytes(); data = json_object(raw)
            child = q.Evidence(output/'capture'); life = child.obj(data['lifecycle'])
            runtime['server_identity'] = {'pid': life['server']['pid'], 'start_identity': life['boot_id']+':'+life['server']['start_time']}
        else:
            runtime['output_path'] = str(output/'process/output.log')
            program, launch, plan, identities = run.materialize(scope, policy, runtime, uuids, build['binaries'])
            server = OwnedServer(argv=launch['argv'], cwd=launch['cwd'], env=launch['env'], evidence_dir=str(output/'process'),
                **{k+'_timeout':v for k,v in launch['timeouts'].items()})
            server.start(); startup_store = EvidenceStore(output/'startup')
            startup = readiness(server, program['endpoint'], policy['server']['http'], startup_store, scope['scope'])
            receipt = server.receipt(); owner = server.identity
            runtime['server_identity'] = {'pid': owner.pid, 'start_identity': receipt['boot_id']+':'+owner.start_time}
            stat = server.output_path.stat()
            entry['log_identity'] = {'path': str(server.output_path), 'device': stat.st_dev, 'inode': stat.st_ino, 'controller_pid': os.getpid()}
            program, launch, plan, identities = run.materialize(scope, policy, runtime, uuids, build['binaries'])
            if required['scenario'] in run.PHASE:
                from serving_cancel_phase import collect_phase_cancel_cell as collector
            elif required['scenario'] == 'drain':
                from serving_drain_capture import collect_drain_cell as collector
            else:
                from serving_worker_failure_capture import collect_worker_failure_cell as collector
            collector(required, program, server=server, output=output/'capture')
        raw = (output/'capture/capture.json').read_bytes()
        entry['capture'] = {'path': entry['capture_root']+'/capture.json', 'sha256': q.digest(raw)}
        program, launch, plan, identities = run.materialize(scope, policy, runtime, uuids, build['binaries'])
        run.replay_cell(required, program, raw, q.Evidence(output/'capture').read, capture_sha256=q.digest(raw),
            launch=launch, plan=plan, identities=identities, log_identity=entry['log_identity'])
        entry['state'] = 'captured'
    except BaseException as error:
        entry['errors'].append(type(error).__name__ + ': ' + str(error))
    finally:
        if server is not None:
            try:
                lifecycle = server.close(reason='serving_scope_complete' if entry['state']=='captured' else 'serving_scope_failed')
                if startup_store is not None:
                    if startup is None: startup = json_object((startup_store.root/'capture.json').read_bytes())
                    # The startup index owns its closure receipt. It stays separate
                    # from the unmodified phase collector's capture/index.
                    startup['lifecycle'] = startup_store.obj(lifecycle); startup_store.index(startup)
            except BaseException as error:
                entry['errors'].append('owned closure: '+str(error)); entry['state']='failed'
        if startup_store is not None:
            entry['startup'] = reference(args.out, str((startup_store.root/'capture.json').relative_to(args.out)))
        cap = output/'capture/capture.json'
        if cap.exists(): entry['capture'] = reference(args.out, str(cap.relative_to(args.out)))
        entry['finished_ns'] = time.monotonic_ns()
    return entry


def capture(args):
    _controller_source(args.repo, args.expected_head)
    require(not _PRELOADED, 'capture requires a fresh process with bound controller imports')
    require(sys.platform == 'linux', 'native serving capture requires Linux')
    source = producer.clean_source(args.repo)
    require(source['commit'] == args.expected_head, 'source differs from requested commit')
    policy, manifest = run.load_policy(args.repo, args.expected_head, source)
    scopes = [s for s in policy['scopes'] if s['scope']['id'] == args.scope]
    require(len(scopes) == 1, 'unknown source-owned scope')
    scope = scopes[0]
    build = json_object((args.build/'build.json').read_bytes())
    q.validate_build(build, source, build['source'], q.Evidence(args.build))
    args.out.mkdir(parents=True, exist_ok=False)
    record = {'schema': 'memra-native-serving-scope-v1', 'scope_id': args.scope, 'state': 'failed', 'errors': [],
        'source_before': source['inputs_sha256'], 'source_after': None, 'build_sha256': q.digest((args.build/'build.json').read_bytes()),
        'binaries_before': None, 'binaries_after': None, 'models_before': None, 'models_after': None,
        'controllers_before': None, 'controllers_after': None, 'lease_owner': None, 'hardware': None, 'hardware_after': None,
        'numeric_environment': None, 'started_unix': time.time(), 'started_ns': time.monotonic_ns(), 'finished_unix': None,
        'finished_ns': None, 'topology': None, 'lease': {'path':'lease.json','sha256':None}, 'lease_before': None,
        'lease_after': None, 'controller': None, 'cells': []}
    required = run.required_cells(manifest, [scope['scope']])
    source_cells = {c['scenario']:c for c in scope['cells']}
    binaries = lambda: {name: producer.binary_id(args.build/'target/release'/name) for name in q.BINARIES}
    models = lambda: {v['path']:q.file_identity(v['path']) for v in scope['artifacts'].values()}
    def save(): write(args.out/'run.json', record)
    save()
    try:
        observations = []; lease = producer.live_lease(generic=False, observations=observations)
        write(args.out/'lease-before.json', observations[0]); record['lease_before'] = reference(args.out,'lease-before.json')
        record['controller'] = observations[0]['controller']
        record['lease_owner'] = {k:lease[k] for k in ('wrapper_pid','child_pid','requested_uuids')}
        ids = lease['requested_uuids']; record['numeric_environment'] = {'CUDA_VISIBLE_DEVICES':q.digest(','.join(ids).encode())}
        record['hardware'], topology = producer.observe_hardware(ids, generic=False)
        require([{k:d[k] for k in ('name','compute_cap')} for d in record['hardware']['devices']] == scope['hardware']['devices'], 'wrong route hardware')
        (args.out/'topology.txt').write_bytes(topology); record['topology'] = reference(args.out,'topology.txt')
        record['binaries_before'] = binaries(); require(record['binaries_before']==build['binaries'], 'stale server/build ELF')
        record['models_before'] = models()
        require(record['models_before']=={v['path']:{k:v[k] for k in ('bytes','sha256')} for v in scope['artifacts'].values()}, 'wrong reviewed model artifacts')
        record['controllers_before'] = observed_controller_identity(args.repo, args.expected_head, source); save()
        for index, cell in enumerate(required):
            result = capture_cell(cell, scope, source_cells[cell['scenario']], args, build, ids, index, args.out/f'cells/{index:03d}')
            record['cells'].append(result); save()
            require(result['state']=='captured', 'required serving cell failed: '+cell['id'])
        record['state']='captured'
    except BaseException as error:
        record['errors'].append(type(error).__name__+': '+str(error))
    finally:
        found = {c['id'] for c in record['cells']}
        for cell in required:
            if cell['id'] not in found: record['cells'].append({**{k:cell[k] for k in ('id','scope','scenario')}, 'state':'not_attempted'})
        try:
            record['source_after'] = producer.clean_source(args.repo)['inputs_sha256']
            record['binaries_after'] = binaries(); record['models_after'] = models()
            record['controllers_after'] = observed_controller_identity(args.repo,args.expected_head,source)
            observations=[]; lease=producer.live_lease(generic=False,observations=observations)
            record['hardware_after'], _ = producer.observe_hardware(lease['requested_uuids'],generic=False)
            write(args.out/'lease-after.json',observations[0]);record['lease_after']=reference(args.out,'lease-after.json')
        except BaseException as error:record['errors'].append('postcheck: '+str(error))
        record['finished_ns']=time.monotonic_ns();record['finished_unix']=time.time()
        if record['errors']:record['state']='failed'
        save()
    require(record['state']=='captured', 'serving capture failed; complete attempt/unattempted denominator retained')
    print('EXECUTED, UNSEALED serving scope:', args.out)


def seal(args):
    source=json_object((args.build/'source.json').read_bytes()); q.verify_source(source,args.repo,args.expected_head)
    policy,manifest=run.load_policy(args.repo,args.expected_head,source)
    require(len(args.scope_run)==len(args.lease)==len(policy['scopes']), 'all source-owned scopes and closed leases are required')
    args.out.mkdir(parents=True,exist_ok=False)
    for name in ('source.json','build.json','build.log','fetch.log'):shutil.copy2(args.build/name,args.out/name)
    stage={'schema':'memra-serving-release-v1','source':reference(args.out,'source.json'),'build':reference(args.out,'build.json'),'runs':[]}
    for key,raw in [('policy',run.tracked(args.repo,args.expected_head,run.POLICY)),('manifest',manifest)]:
        path='serving/'+key+'.json';(args.out/path).parent.mkdir(exist_ok=True);(args.out/path).write_bytes(raw);stage[key]=reference(args.out,path)
    seen=set()
    for folder,lease_path in zip(args.scope_run,args.lease):
        item=json_object((folder/'run.json').read_bytes());name=item['scope_id']
        require(name not in seen and name in {s['scope']['id'] for s in policy['scopes']}, 'duplicate/foreign serving scope');seen.add(name)
        dest=args.out/'serving/runs'/name;dest.mkdir(parents=True)
        for path in folder.rglob('*'):
            if path.is_file():
                require(not path.is_symlink() and path.resolve().is_relative_to(folder.resolve()),'scope evidence symlink')
                target=dest/path.relative_to(folder);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        require(not (dest/'lease.json').exists(),'capture cannot preselect its completed lease')
        shutil.copy2(lease_path,dest/'lease.json');item['lease']=reference(dest,'lease.json');write(dest/'run.json',item)
        stage['runs'].append(reference(args.out,str((dest/'run.json').relative_to(args.out))))
    write(args.out/'serving/stage.json',stage)
    index={'schema':'memra-serving-stage-v1','source':stage['source'],'build':stage['build'],
        'serving':reference(args.out,'serving/stage.json'),'payloads':{str(p.relative_to(args.out)):q.digest(p.read_bytes())
            for p in sorted(args.out.rglob('*')) if p.is_file()}}
    write(args.out/'stage.json',index)
    run.validate_stage_directory(args.out,args.repo,args.expected_head)
    print('SEALED required serving stage; generic battery and full release v2 still required:',args.out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('capture','seal','verify'))
    parser.add_argument('--repo',type=Path,default=q.ROOT);parser.add_argument('--expected-head',required=True)
    parser.add_argument('--out',type=Path,required=True);parser.add_argument('--build',type=Path)
    parser.add_argument('--scope');parser.add_argument('--port',type=int,default=18426)
    parser.add_argument('--scope-run',type=Path,action='append',default=[])
    parser.add_argument('--lease',type=Path,action='append',default=[])
    args=parser.parse_args();args.repo=args.repo.resolve();args.out=args.out.resolve()
    if args.build is not None:args.build=args.build.resolve()
    try:
        require(q.COMMIT.fullmatch(args.expected_head),'expected-head must be an exact commit')
        if args.mode=='verify': print(json.dumps(run.validate_stage_directory(args.out,args.repo,args.expected_head)['verdicts'],sort_keys=True))
        else:
            require(args.build is not None and (args.mode!='capture' or args.scope),'build/scope missing')
            globals()[args.mode](args)
        return 0
    except (Exception,KeyboardInterrupt) as error:
        print('UNQUALIFIED:',type(error).__name__+':',error,file=sys.stderr);return 1

if __name__=='__main__':sys.exit(main())
