"""Required serving release binding, using the existing eleven scenario adapters.

Policy is read from a fixed tracked Git path, never from a run-selected scope.
Dynamic binding is limited to owned process identity, local paths/port, physical
UUIDs and a correlation key. Payloads, oracles, routes, deadlines, artifact bytes
and hardware classes come from that policy. This is auditable execution evidence,
not a signature scheme; invented records cannot become native proof by hashing.
"""
import copy
import math
from pathlib import PurePosixPath
import re

import release_qualification as q
from serving_completion import _keys, _same
from serving_manifest import required_cells, validate_cell_coverage
from serving_release import json_object, require

POLICY = 'tools/serving-release.programs.json'
MANIFEST = 'tools/serving-release.cells.json'
GROUP = frozenset(('short_prompt', 'long_prompt', 'wire_completion', 'cache_restore', 'concurrent_completion', 'overload_recovery'))
PHASE = frozenset(('cancel_queued', 'cancel_prime', 'cancel_decode'))
SCENARIOS = GROUP | PHASE | {'drain', 'worker_failure_recovery'}
RUNTIME_FIELDS = {'server_identity', 'endpoint', 'identities', 'client_trace_key'}
CONTROLLER_FILES = ('tools/serving-run.py', 'tools/qualify-release.py', 'tools/release_qualification.py',
                    'tools/release_inputs.py', 'tools/release_input_view.py',
                    'tools/check_hardware_gate.py', 'tools/release-coverage.py')


def tracked(repo, head, path):
    # A symlink or publication sidecar must not replace executable policy.
    require(q.safe_path(path) == path and not q.publication_metadata(path), 'invalid tracked policy path')
    tree = q.tree_files(repo, head)
    require(tree.get(path, {}).get('mode') in ('100644', '100755'), 'required tracked policy/input missing: ' + path)
    return q.git(repo, 'show', f'{head}:{path}')


def load_policy(repo, head, source=None):
    raw = tracked(repo, head, POLICY)
    manifest = tracked(repo, head, MANIFEST)
    roster = q.read_roster(tracked(repo, head, 'tools/release-roster.tsv'))
    policy = json_object(raw)
    _keys(policy, {'schema', 'scopes'}, 'serving policy')
    require(policy['schema'] == 'memra-serving-policy-v1' and type(policy['scopes']) is list and policy['scopes'],
            'missing reviewed serving scopes')
    by_model = {r['id']: r for r in roster}
    seen = set(); artifact_census = {}
    for item in policy['scopes']:
        _keys(item, {'roster_id', 'scope', 'artifacts', 'hardware', 'cells'}, 'serving scope policy')
        require(item['roster_id'] in by_model, 'serving policy invents a roster model')
        seen.add(item['roster_id'])
        artifacts = item['artifacts']
        require(type(artifacts) is dict and 'model' in artifacts and 'server_binary' not in artifacts,
                'model artifact closure is missing')
        for name, identity in artifacts.items():
            require(type(name) is str and name, 'invalid artifact name')
            _keys(identity, {'path', 'bytes', 'sha256'}, 'source-owned artifact')
            require(type(identity['path']) is str and PurePosixPath(identity['path']).is_absolute(), 'artifact path is not absolute')
            q.validate_identity(identity, name)
            require(identity["path"] not in artifact_census or _same(artifact_census[identity["path"]], identity), "policy declares conflicting artifact identities")
            artifact_census[identity["path"]] = identity
        require(artifacts['model']['path'] == by_model[item['roster_id']]['path'], 'serving model differs from required roster')
        _keys(item['hardware'], {'devices'}, 'required serving hardware')
        devices = item['hardware']['devices']
        require(type(devices) is list and 1 <= len(devices) <= 32, 'missing reviewed physical device set')
        for device in devices:
            _keys(device, {'name', 'compute_cap'}, 'required device class')
            require(all(type(v) is str and v for v in device.values()), 'invalid device class')
        cells = item['cells']
        require(type(cells) is list and len(cells) == 11 and {c.get('scenario') for c in cells} == SCENARIOS,
                'policy must specify each required scenario exactly once')
        for cell in cells:
            _keys(cell, {'scenario', 'program', 'server'}, 'source-owned cell')
            p = cell['program']
            require(type(p) is dict and not (p.keys() & RUNTIME_FIELDS), 'policy contains dynamic owner bindings')
            require(p.get('cell_id') == item['scope']['id'] + '/' + cell['scenario']
                    and _same(p.get('scope'), item['scope']), 'policy program substitutes scope/cell')
            require(type(p.get('requests')) is list and p['requests'], 'policy request denominator missing')
            _keys(cell['server'], {'env', 'timeouts', 'http'}, 'source-owned server configuration')
            env = cell['server']['env']
            require(type(env) is dict and all(type(k) is str and type(v) is str for k, v in env.items()), 'invalid explicit launch environment')
            require(not any(k.startswith('MEMRA_API_KEY') for k in env)
                    and not {'CUDA_VISIBLE_DEVICES', 'MEMRA_ADDR', 'LD_PRELOAD'} & env.keys(), 'policy overrides controlled launch binding')
            require(env.get('MEMRA_MODELS') == item['scope']['model'] + '=' + artifacts['model']['path'],
                    'served alias/model differs from roster artifact')
            require(not {'skip', 'skipped'} & p.keys(), 'policy cannot skip required work')
    require(seen == by_model.keys(), 'serving policy omits required roster models')
    required_cells(manifest, [x['scope'] for x in policy['scopes']])
    if source is not None:
        for path in (POLICY, MANIFEST, 'tools/release-roster.tsv'):
            require(path in source['files'], 'policy was not an input of the tested build')
    return policy, manifest


def controller_identity(repo, head, source):
    paths = sorted(set(CONTROLLER_FILES) | {p for p in source['files']
        if p.startswith('tools/serving_') and p.endswith('.py')})
    require(all(p in source['files'] for p in paths), 'controller is outside built source closure')
    return {p: {'sha256': q.digest(q.git(repo, 'show', f'{head}:{p}')), 'mode': source['files'][p]['mode']} for p in paths}


def materialize(scope, cell, runtime, uuids, binaries):
    """Only operating identities vary; the reviewed request/oracle program does not."""
    _keys(runtime, {'server_identity', 'port', 'cwd', 'binary', 'output_path', 'client_trace_key'}, 'cell runtime binding')
    require(type(runtime['port']) is int and 1 <= runtime['port'] <= 65535, 'invalid bound loopback port')
    require(all(type(runtime[k]) is str and PurePosixPath(runtime[k]).is_absolute()
                for k in ('cwd', 'binary', 'output_path')), 'invalid runtime path')
    endpoint = {'host': '127.0.0.1', 'port': runtime['port']}
    paths = {name: v['path'] for name, v in scope['artifacts'].items()}
    paths['server_binary'] = runtime['binary']
    identities = {name: {k: v[k] for k in ('bytes', 'sha256')} for name, v in scope['artifacts'].items()}
    identities['server_binary'] = {k: binaries['memra-server'][k] for k in ('bytes', 'sha256')}
    program = copy.deepcopy(cell['program'])
    program['server_identity'] = copy.deepcopy(runtime['server_identity'])
    if cell['scenario'] not in GROUP:
        program['endpoint'] = endpoint
        if cell['scenario'] in PHASE:
            require(type(runtime['client_trace_key']) is str and re.fullmatch('[0-9a-f]{32}', runtime['client_trace_key']), 'missing client correlation key')
            program['client_trace_key'] = runtime['client_trace_key']
        else:
            require(runtime['client_trace_key'] is None, 'unexpected correlation key')
            program['identities'] = paths
        require(_same(program.get('http'), cell['server']['http']), 'program and launch HTTP deadlines differ')
    else:
        require(runtime['client_trace_key'] is None, 'unexpected group correlation key')
    env = {**cell['server']['env'], 'CUDA_VISIBLE_DEVICES': ','.join(uuids),
           'MEMRA_ADDR': '127.0.0.1:' + str(runtime['port'])}
    launch = {'argv': [runtime['binary']], 'cwd': runtime['cwd'], 'env': env,
              'timeouts': cell['server']['timeouts'], 'output_path': runtime['output_path']}
    plan = None
    if cell['scenario'] in GROUP:
        requests = program['requests']
        sent = lambda rs: [{k: r[k] for k in ('id', 'model', 'wire', 'path', 'payload')} for r in rs]
        if cell['scenario'] == 'overload_recovery':
            groups = [{'id': program['cell_id'] + '/pressure', 'mode': 'concurrent', 'requests': sent([r for r in requests if r['role'] != 'recovery'])},
                      {'id': program['cell_id'] + '/recovery', 'mode': 'serial', 'requests': sent([r for r in requests if r['role'] == 'recovery'])}]
        else:
            groups = [{'id': program['cell_id'], 'mode': program['mode'], 'requests': sent(requests)}]
            if cell['scenario'] == 'cache_restore': groups[0]['metrics'] = True
        plan = {'schema': 'memra-serving-capture-plan-v1', 'server': {'argv': launch['argv'], 'cwd': launch['cwd'],
                'env': env, **endpoint, **{k + '_timeout': v for k, v in launch['timeouts'].items()}},
                'identities': paths, 'http': cell['server']['http'], 'groups': groups}
    return program, launch, plan, identities


def replay_cell(required, program, raw, read, *, capture_sha256, launch, plan, identities, log_identity):
    common = {'expected_capture_sha256': capture_sha256}
    scenario = required['scenario']
    if scenario in GROUP:
        if scenario == 'overload_recovery':
            from serving_overload_gate import evaluate_overload_capture as evaluate
        else:
            from serving_group_gate import evaluate_group_capture as evaluate
        return evaluate(required, program, raw, read, expected_plan=plan, expected_identities=identities, **common)
    if scenario in PHASE:
        from serving_cancel_evidence import read_phase_capture
        return read_phase_capture(raw, read, expected_required=required, expected_program=program,
                                  expected_server=launch, expected_log_identity=log_identity, **common)
    if scenario == 'drain':
        from serving_drain_evidence import read_drain_capture
        return read_drain_capture(raw, read, expected_required=required, expected_program=program,
                                  expected_server=launch, expected_identities=identities, **common)
    require(scenario == 'worker_failure_recovery', 'no required scenario adapter')
    from serving_worker_failure_evidence import read_worker_failure_capture
    return read_worker_failure_capture(raw, read, expected_required=required, expected_program=program,
        expected_server=launch, expected_identities=identities, expected_log_identity=log_identity, **common)


def lease_observation(value, lease, controller, hardware):
    """Validate selected raw FLOCK rows and Linux process ancestry, not a boolean."""
    _keys(value, {'started_ns', 'finished_ns', 'unix_ns', 'boot_id', 'controller', 'ancestors', 'locks'}, 'live lease observation')
    require(type(value['started_ns']) is int and type(value['finished_ns']) is int
            and 0 < value['started_ns'] <= value['finished_ns'] and type(value['unix_ns']) is int and value['unix_ns'] > 0, 'invalid lease observation clocks')
    require(_same({k:v for k,v in value['controller'].items() if k != 'state'},
                  {k:v for k,v in controller.items() if k != 'state'}) and controller['identity_source'] == 'linux_proc_start_ticks', 'foreign capture controller')
    import uuid
    require(str(uuid.UUID(value['boot_id'])) == value['boot_id'], 'invalid controller boot identity')
    require(type(controller['pid']) is int and controller['pid'] > 1 and re.fullmatch(r'[0-9]+', controller['start_time']), 'invalid controller birth')
    ancestors = value['ancestors']
    require(type(ancestors) is list and ancestors and ancestors[0] == value['controller'], 'missing raw process ancestry')
    ids = [p['pid'] for p in ancestors]
    require(len(set(ids)) == len(ids) and all(a['ppid'] == b['pid'] for a, b in zip(ancestors, ancestors[1:])), 'broken process ancestry')
    require(lease['child_pid'] in ids and lease['wrapper_pid'] in ids
            and ids.index(lease['child_pid']) < ids.index(lease['wrapper_pid']), 'controller was not inside live lease')
    require(type(value['locks']) is list and len(value['locks']) == len(lease['requested_uuids']), 'live lock denominator differs')
    for row, card in zip(value['locks'], sorted(lease['requested_uuids'])):
        _keys(row, {'uuid', 'path', 'device_major', 'device_minor', 'inode', 'raw'}, 'physical lock observation')
        require(row['uuid'] == card and row['path'] == lease['lock_files'][card], 'foreign lock path')
        fields = row['raw'].split()
        require(len(fields) >= 8 and fields[1:4] == ['FLOCK', 'ADVISORY', 'WRITE']
                and int(fields[4]) == lease['wrapper_pid'], 'no actual exclusive wrapper FLOCK')
        major, minor, inode = fields[5].split(':')
        require((int(major, 16), int(minor, 16), int(inode)) == (row['device_major'], row['device_minor'], row['inode']), 'FLOCK inode differs')
    require([d['uuid'] for d in hardware['devices']] == lease['requested_uuids'], 'observed hardware changed')
    return value['started_ns'], value['finished_ns']


class ScopedEvidence:
    def __init__(self, parent, prefix):
        self.parent, self.prefix = parent, q.safe_path(prefix)
    def read(self, path):
        return self.parent.read(self.prefix + '/' + q.safe_path(path))
    def bound(self, ref):
        _keys(ref, {'path', 'sha256'}, 'scoped reference')
        raw = self.read(ref['path'])
        require(q.digest(raw) == ref['sha256'], 'scoped raw bytes changed')
        return raw
    def obj(self, ref):
        return json_object(self.bound(ref))


def validate_serving_run(run, evidence, required_manifest, expected_bindings):
    """Replay the complete source-owned model/route/cell denominator.

    expected_bindings is built by the release verifier from its validated v3
    build and immutable source, never supplied by captured JSON.
    """
    _keys(run, {'schema', 'source', 'build', 'policy', 'manifest', 'runs'}, 'serving release stage')
    require(run['schema'] == 'memra-serving-release-v1', 'unsupported serving release stage')
    b = expected_bindings
    require(run['source'] == b['source_reference'] and run['build'] == b['build_reference'], 'serving stage uses another source/build')
    policy, manifest = load_policy(b['repo'], b['head'], b['source'])
    require(manifest == required_manifest == evidence.bound(run['manifest'])
            and _same(json_object(evidence.bound(run['policy'])), policy), 'serving manifest/policy is stale or substituted')
    controllers = controller_identity(b['repo'], b['source']['commit'], b['source'])
    scopes = {s['scope']['id']: s for s in policy['scopes']}
    require(type(run['runs']) is list and len(run['runs']) == len(scopes), 'required model/route run denominator differs')
    seen = set(); verdicts = []; owners = set()
    for reference in run['runs']:
        prefix = str(PurePosixPath(reference['path']).parent)
        local = ScopedEvidence(evidence, prefix)
        captured = evidence.obj(reference)
        _keys(captured, {'schema', 'scope_id', 'state', 'errors', 'source_before', 'source_after', 'build_sha256',
            'binaries_before', 'binaries_after', 'models_before', 'models_after', 'controllers_before', 'controllers_after',
            'lease_owner', 'hardware', 'hardware_after', 'numeric_environment', 'started_unix', 'finished_unix',
            'started_ns', 'finished_ns', 'topology', 'lease', 'lease_before', 'lease_after', 'controller', 'cells'}, 'native serving scope run')
        name = captured['scope_id']
        require(name in scopes and name not in seen, 'duplicate/unknown serving route'); seen.add(name)
        scope = scopes[name]
        require(captured['schema'] == 'memra-native-serving-scope-v1' and captured['state'] == 'captured'
                and captured['errors'] == [], 'failed/skipped/incomplete serving scope')
        require(captured['source_before'] == captured['source_after'] == b['source']['inputs_sha256']
                and captured['build_sha256'] == b['build_reference']['sha256'], 'serving source/build changed')
        require(_same(captured['binaries_before'], b['build']['binaries']) and _same(captured['binaries_after'], b['build']['binaries']), 'serving ELF inventory changed')
        models = {v['path']: {k: v[k] for k in ('bytes', 'sha256')} for v in scope['artifacts'].values()}
        require(_same(captured['models_before'], models) and _same(captured['models_after'], models), 'serving model/draft/artifact changed')
        require(_same(captured['controllers_before'], controllers) and _same(captured['controllers_after'], controllers), 'serving controller source changed')
        require(_same(captured['hardware'], captured['hardware_after']), 'serving topology/hardware changed')
        devices = [{k: d[k] for k in ('name', 'compute_cap')} for d in captured['hardware']['devices']]
        require(_same(devices, scope['hardware']['devices']), 'hardware differs from reviewed model/route program')
        require(q.digest(local.bound(captured['topology'])) == captured['hardware']['topology_sha256'], 'topology raw bytes changed')
        lease = local.obj(captured['lease']); q.validate_physical_lease(lease, captured)
        before = local.obj(captured['lease_before']); after = local.obj(captured['lease_after'])
        first = lease_observation(before, lease, captured['controller'], captured['hardware'])
        last = lease_observation(after, lease, captured['controller'], captured['hardware'])
        require(before['boot_id'] == after['boot_id'] and
                [{k:v for k,v in p.items() if k != 'state'} for p in before['ancestors']] ==
                [{k:v for k,v in p.items() if k != 'state'} for p in after['ancestors']], 'lease process identity changed')
        require(type(captured['started_ns']) is int and type(captured['finished_ns']) is int
                and captured['started_ns'] <= first[0] <= first[1] < last[0] <= last[1] <= captured['finished_ns'], 'live lease interval differs')
        require(captured['started_unix'] <= before['unix_ns']/1e9 <= after['unix_ns']/1e9 <= captured['finished_unix'], 'lease clock observations disagree')
        cells = captured['cells']
        expected = validate_cell_coverage(manifest, [scope['scope']], cells)
        by_id = {x['id']: x for x in cells}
        source_cells = {c['scenario']: c for c in scope['cells']}
        previous = first[1]
        for required in expected:
            cell = by_id[required['id']]
            _keys(cell, {'id', 'scope', 'scenario', 'state', 'errors', 'runtime', 'capture', 'capture_root', 'log_identity',
                        'started_ns', 'finished_ns', 'startup'}, 'serving capture binding')
            require(cell['errors'] == [] and type(cell['started_ns']) is int and type(cell['finished_ns']) is int
                    and previous <= cell['started_ns'] < cell['finished_ns'] <= last[0], 'cell failed or escaped live lease interval')
            previous = cell['finished_ns']
            owner_key = q.object_digest(cell['runtime']['server_identity'])
            require(owner_key not in owners, 'closed server identity reused by another cell'); owners.add(owner_key)
            program, launch, plan, identities = materialize(scope, source_cells[required['scenario']], cell['runtime'], lease['requested_uuids'], b['build']['binaries'])
            root = q.safe_path(cell['capture_root'])
            require(cell['capture']['path'] == root + '/capture.json', 'capture path differs from raw evidence root')
            raw = local.bound(cell['capture']); index = json_object(raw)
            child = ScopedEvidence(local, root)
            result = replay_cell(required, program, raw, child.read, capture_sha256=cell['capture']['sha256'],
                launch=launch, plan=plan, identities=identities, log_identity=cell['log_identity'])
            # Captures use one host monotonic clock. Body reads after server exit
            # are already handled by the drain adapter; never add post-exit HTTP.
            if required['scenario'] in GROUP:
                life = child.obj(index['lifecycle'])
            elif required['scenario'] in PHASE:
                life = validate_startup(local, cell, program, launch)
                from serving_policy import _drain_lifecycle
                _drain_lifecycle({'server_identity': program['server_identity'], 'stop_reason': 'serving_scope_complete',
                    'drain_timeout_ns': int(launch['timeouts']['drain']*1e9)}, life)
            else:
                life = child.obj(index['lifecycle'])
            if required['scenario'] not in GROUP | PHASE:
                startup_life = validate_startup(local, cell, program, launch)
                require(_same(startup_life, life), 'startup/final lifecycle differs')
            from serving_drain_evidence import validate_launch
            validate_launch(life, launch)
            require(life['boot_id'] == before['boot_id'], 'server capture belongs to another host boot')
            require(life['output_path'] == launch['output_path'] and
                    {'pid':life['server']['pid'], 'start_identity':life['boot_id']+':'+life['server']['start_time']} == program['server_identity'], 'final owned identity differs')
            require(cell['started_ns'] <= int(life['started_monotonic']*1e9)
                    and int(life['cleanup']['finished_monotonic']*1e9) <= cell['finished_ns'], 'owned server escaped cell lifetime')
            require(life['supervisor']['ppid'] == captured['controller']['pid'], 'server was not owned by the leased controller')
            if required['scenario'] not in GROUP:
                require(cell['started_ns'] <= index['started_ns'] <= index['finished_ns'] <= cell['finished_ns'], 'capture escaped cell boundary')
            verdicts.append({'id': required['id'], 'scope': required['scope'], 'scenario': required['scenario'],
                             'capture_sha256': cell['capture']['sha256'], 'verified_payloads': result['verified_payloads'],
                             'requests': len(program['requests']), 'verdict': 'required raw evidence validated'})
    require(seen == scopes.keys(), 'missing required serving model/route')
    return {'schema': 'memra-serving-verdicts-v1', 'policy_sha256': run['policy']['sha256'],
            'manifest_sha256': run['manifest']['sha256'], 'cells': verdicts,
            'models': {v['path']: {k:v[k] for k in ('bytes','sha256')} for scope in policy['scopes'] for v in scope['artifacts'].values()},
            'scope': 'reviewed required serving programs; no universal model/hardware or latency claim'}


def validate_startup(local, cell, program, launch):
    from serving_cancel_evidence import _Bundle
    from serving_evidence import _listener
    from serving_drain_evidence import validate_launch
    reference = cell['startup']
    require(type(reference) is dict and reference['path'].endswith('/startup/capture.json'), 'missing owned startup evidence')
    root = str(PurePosixPath(reference['path']).parent)
    evidence = ScopedEvidence(local, root)
    startup = local.obj(reference)
    _keys(startup, {'schema','state','observations','listeners','errors','lifecycle','payloads'}, 'serving startup')
    require(startup['schema']=='memra-serving-startup-v1' and startup['state']=='ready' and startup['errors']==[], 'startup failed')
    bundle = _Bundle(startup['payloads'], evidence.read)
    rows = [bundle.observation(ref,program['http']['max_body_bytes']) for ref in startup['observations']]
    require(len(rows)>=2 and len(rows)%2==0, 'startup probe denominator differs')
    previous = cell['started_ns']
    for i,row in enumerate(rows):
        require(row['id']=='startup-'+str(i+1) and row['method']=='GET' and row['path']==('/health' if i%2==0 else '/readyz')
                and row['server_identity']==program['server_identity'] and row['transport_error'] is None
                and previous<=row['started_ns']<=row['finished_ns']<=cell['finished_ns'], 'startup probe ownership/framing/order differs')
        body=json_object(row['body']);require(row['status'] in (200,503), 'unexpected startup status')
        previous=row['finished_ns']
    for row,status in zip(rows[-2:],('ok','ready')):
        body=json_object(row['body'])
        require(row['status']==200 and body.get('status')==status and program['scope']['model'] in body.get('models',[]), 'startup did not observe ready model')
    proofs=[]
    for ref in startup['listeners']:
        item=bundle.obj(ref)
        if 'error' in item:
            require(set(item)=={'error','observed_ns'} and type(item['error']) is str and item['error'], 'invalid startup listener failure')
        else:proofs.append(_listener(item,program['server_identity'],program['endpoint']))
    require(len(proofs)>=2 and proofs[-2][1]<=rows[-2]['started_ns'] and rows[-1]['finished_ns']<=proofs[-1][0], 'startup listener brackets missing')
    life=bundle.obj(startup['lifecycle']);validate_launch(life,launch)
    return life


def validate_stage_directory(root, repo, head):
    """CPU/file replay of a serving-only stage. It is not a full release record."""
    from pathlib import Path
    root = Path(root)
    index = json_object((root / 'stage.json').read_bytes())
    _keys(index, {'schema', 'source', 'build', 'serving', 'payloads'}, 'serving stage index')
    require(index['schema'] == 'memra-serving-stage-v1' and type(index['payloads']) is dict and index['payloads'], 'missing sealed serving stage')
    evidence = q.Evidence(root); evidence.payloads = index['payloads']
    for name, sha in index['payloads'].items():
        require(name in ('source.json', 'build.json', 'build.log', 'fetch.log') or name.startswith('serving/'),
                'serving stage payload invades another release namespace')
        require(type(sha) is str and q.SHA.fullmatch(sha) and q.digest(evidence.read(name)) == sha, 'serving stage payload changed')
    source, build = evidence.obj(index['source']), evidence.obj(index['build'])
    proof = q.verify_source(source, repo, head)
    q.validate_build(build, source, index['source'], evidence)
    verdicts = validate_serving_run(evidence.obj(index['serving']), evidence, tracked(repo, head, MANIFEST),
        {'repo': repo, 'head': head, 'source': source, 'source_reference': index['source'],
         'build': build, 'build_reference': index['build']})
    return {'source': source, 'build': build, 'verdicts': verdicts, 'source_proof': proof,
            'index': index, 'scope': 'required-serving-only'}


def import_stage(root, destination):
    """Copy an already verified closure; caller verifies it again when sealing v2."""
    from pathlib import Path
    root, destination = Path(root), Path(destination)
    index = json_object((root / 'stage.json').read_bytes())
    evidence = q.Evidence(root)
    for name, sha in index['payloads'].items():
        require(name in ('source.json', 'build.json', 'build.log', 'fetch.log') or name.startswith('serving/'),
                'serving import invades another release namespace')
        raw = evidence.read(name)
        require(q.digest(raw) == sha, 'serving stage changed during import')
        path = destination / q.safe_path(name)
        require(path.resolve().is_relative_to(destination.resolve()) and not path.is_symlink(), 'serving import escapes output')
        if path.exists(): require(path.read_bytes() == raw, 'serving import collides with different evidence')
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as out: out.write(raw)
    return index['serving']
