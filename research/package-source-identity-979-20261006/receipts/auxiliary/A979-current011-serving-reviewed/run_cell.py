#!/usr/bin/python3 -I
"""Unexecuted current-source two-arm serving nomination. Never run to plan."""
import argparse, hashlib, importlib.util, json, os, queue, re, signal, stat, subprocess, threading, time
import urllib.request
from pathlib import Path
_HELPER=Path(__file__).absolute().with_name('runtime_helpers.py')
assert hashlib.sha256(_HELPER.read_bytes()).hexdigest()=='10b76ca09f4ec51caf20c21237eb8d506f86357fc173cf072ef2f9840cf00a8f'
_spec=importlib.util.spec_from_file_location('cell_runtime_helpers',_HELPER);h=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(h)

def digest_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    budget=h.Budget(total=600,arms=2,term=20,kill=10,join=5)
    with budget.operation():
        parser = argparse.ArgumentParser(); parser.add_argument('--inputs', type=Path, required=True)
        args = parser.parse_args(); inputs = json.loads(args.inputs.read_bytes())
        assert set(inputs) == {'source', 'workspace', 'binary', 'binary_sha256', 'system_fingerprint', 'model', 'draft',
                              'metadata', 'metadata_sha256', 'request_file', 'request_sha256',
                              'record_root', 'port', 'gate_files', 'library_path', 'admission'}
        assert inputs['admission'] == 'one-rig-package-serving-cell-600s'
        assert all(isinstance(inputs[key], str) and inputs[key]
                   for key in ('binary', 'binary_sha256', 'system_fingerprint')), 'Root binary binding missing'
        assert os.environ.get('MEMRA_RIG_LOCK_FD') == '9'
        assert os.environ.get('MEMRA_GPU_LOCK') == '/tmp/memra-5090.lock'
        workspace = Path(inputs['workspace']); binary = Path(inputs['binary'])
        assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace, text=True).strip() == inputs['source']
        assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=workspace)
        assert digest_file(binary) == inputs['binary_sha256']
        for name, sha in inputs['gate_files'].items():
            assert digest_file(workspace/name) == sha
        required = {'tools/port-guard.sh', 'tools/tier-lock-proof.py', 'tools/assert-drafter-attached.sh'}
        assert required <= set(inputs['gate_files'])
        lock = subprocess.run(['/usr/bin/python3', str(workspace/'tools/tier-lock-proof.py'),
                               '--fd', '9', '--lock', '/tmp/memra-5090.lock', '--owner', 'collector'],
                              capture_output=True, pass_fds=(9,), timeout=5)
        assert lock.returncode == 0, lock.stderr
        artifact_snapshots = {}
        for role in ('model', 'draft'):
            row = inputs[role]; path = Path(row['path'])
            before = path.lstat()
            assert stat.S_ISREG(before.st_mode) and before.st_size == row['bytes']
            assert digest_file(path) == row['sha256']
            after = path.lstat()
            identity = lambda s: (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            assert identity(before) == identity(after)
            artifact_snapshots[role] = identity(after)
        metadata = Path(inputs['metadata']); request_file = Path(inputs['request_file'])
        assert digest_file(metadata) == inputs['metadata_sha256']
        assert digest_file(request_file) == inputs['request_sha256']
        requests = json.loads(request_file.read_bytes())
        assert len(requests['turns']) == 8
        absent = {'temperature', 'top_p', 'top_k', 'min_p', 'seed', 'frequency_penalty',
                  'presence_penalty', 'repetition_penalty', 'reasoning_effort'}
        assert all(not absent & set(row) for row in requests['turns']+[requests['stream']])
        root = Path(inputs['record_root']); assert root.is_absolute() and not root.exists()
        root.mkdir()
        base = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
                'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                'CUDA_DEVICE_ORDER': 'PCI_BUS_ID',
                'MEMRA_GPU_LOCK': '/tmp/memra-5090.lock', 'MEMRA_RIG_LOCK_FD': '9'}
        if inputs['library_path'] is not None:
            base['LD_LIBRARY_PATH'] = inputs['library_path']
        version = subprocess.run([str(binary), '--version'], cwd=workspace, env=base,
                                 capture_output=True, timeout=10)
        (root/'version.stdout').write_bytes(version.stdout); (root/'version.stderr').write_bytes(version.stderr)
        text = version.stdout.decode(); assert version.returncode == 0 and 'build_id_src package-source-v1' in text
        fingerprint = next(line.removeprefix('system_fingerprint ') for line in text.splitlines()
                           if line.startswith('system_fingerprint '))
        assert fingerprint == inputs['system_fingerprint'], 'inspected binary fingerprint differs'
    def lock_proof():
        with budget.operation(cap=5,end=budget.end):
            checked=subprocess.run(['/usr/bin/python3','-I','-B',str(workspace/'tools/tier-lock-proof.py'),
                                    '--fd','9','--lock','/tmp/memra-5090.lock','--owner','collector'],
                                   env=base,capture_output=True,pass_fds=(9,),timeout=budget.remaining(budget.end,5))
            assert checked.returncode==0,checked.stderr
            return json.loads(checked.stdout)
    all_results = []
    for arm, cache in (('cache-off', '0'), ('cache-on', '1024')):
        budget.remaining()
        records = root/arm; records.mkdir(); port = inputs['port']
        guard = subprocess.run(['bash', str(workspace/'tools/port-guard.sh'), 'check', 'package-rig-cell', str(port)],
                               cwd=workspace, env=base, capture_output=True, timeout=budget.remaining(cap=5))
        assert guard.returncode == 0, guard.stderr
        env = dict(base, MEMRA_MODELS='package-probe='+inputs['model']['path'],
                   MEMRA_MTP_DRAFT=inputs['draft']['path'], MEMRA_MODEL_METADATA=str(metadata),
                   MEMRA_COMPAT='openai', MEMRA_ADDR='127.0.0.1:'+str(port), MEMRA_CTX='2048',
                   MEMRA_PREFIX_CACHE_MB=cache)
        log = records/'server.log'
        lock_proof()
        proc = subprocess.Popen([str(binary)], cwd=workspace, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, start_new_session=True, pass_fds=(9,))
        ownership=None
        reader=h.LogReader(proc,log);primary=None
        try:
            ownership=h.OwnedSession(proc)
            reader.start()
            with budget.operation(cap=120):
                assert reader.events.get(timeout=budget.remaining(cap=120)) == 'listening'
            # Verify actual listener ownership using the existing gate seam.
            owned = subprocess.run(['bash', '-c', 'source "$1"; memra_port_owned package-rig-cell "$2" "$3"',
                                    '_', str(workspace/'tools/port-guard.sh'), str(port), str(proc.pid)],
                                   env=base, capture_output=True, timeout=budget.remaining(cap=5))
            assert owned.returncode == 0, owned.stderr
            def http(label, path, payload=None):
                budget.remaining()
                request = urllib.request.Request('http://127.0.0.1:'+str(port)+path,
                                                data=json.dumps(payload).encode() if payload is not None else None,
                                                headers={'Content-Type': 'application/json'})
                # Absolute wall alarm covers complete bodies, including trickle reads.
                with budget.operation(cap=30):
                    with urllib.request.urlopen(request, timeout=budget.remaining(cap=30)) as response:
                        body = response.read(4*1024*1024+1); assert len(body) <= 4*1024*1024
                        assert response.status == 200
                (records/(label+'.response')).write_bytes(body)
                (records/(label+'.request.json')).write_text(json.dumps(payload)+'\n')
                return body
            http('ready', '/readyz'); models = json.loads(http('models', '/v1/models'))
            assert any(row['id'] == 'package-probe' for row in models['data'])
            turns = []
            for i, payload in enumerate(requests['turns']):
                value = json.loads(http('turn'+str(i+1), '/v1/chat/completions', payload))
                assert value['system_fingerprint'] == fingerprint
                assert isinstance(value['id'], str) and value['id'] and isinstance(value['created'], int)
                assert value['usage']['completion_tokens'] > 0
                assert value['choices'][0]['finish_reason'] in ('stop', 'length')
                message = value['choices'][0]['message']
                assert ((message.get('reasoning') or '')+(message.get('content') or '')).strip()
                assert value['usage']['spec']['rounds'] > 0 and value['usage']['spec']['drafted'] > 0
                turns.append({'turn': i+1, 'id': value['id'], 'prompt_tokens': value['usage']['prompt_tokens'],
                              'cached_tokens': value['usage']['prompt_tokens_details']['cached_tokens'],
                              'spec': value['usage']['spec'], 'fingerprint': value['system_fingerprint']})
            stream = http('stream', '/v1/chat/completions', requests['stream']).decode()
            frames = [line[6:] for line in stream.splitlines() if line.startswith('data: ')]
            assert frames and frames[-1] == '[DONE]'
            decoded = [json.loads(frame) for frame in frames[:-1]]
            assert decoded and all(frame['system_fingerprint'] == fingerprint for frame in decoded)
            assert len({frame['id'] for frame in decoded}) == 1
            assert all(isinstance(frame['created'], int) for frame in decoded)
            attach = subprocess.run(['bash', str(workspace/'tools/assert-drafter-attached.sh'), str(log),
                                     inputs['draft']['path']], capture_output=True, timeout=budget.remaining(cap=5))
            assert attach.returncode == 0, attach.stdout+attach.stderr
            current_log = log.read_text()
            # Cache-store evidence is distinct from spec-affinity reuse.
            cache_evidence = {}
            if arm == 'cache-off':
                refusals = re.findall(r'\[prefix-cache\] insert refused: entry \d+ exceeds budget (\d+)', current_log)
                assert refusals and set(refusals) == {'0'}
                prefix_lines = [line for line in current_log.splitlines() if '[prefix-cache]' in line]
                assert not any(re.search(r'\bhit\b|\brestored?\b', line) for line in prefix_lines)
                cache_evidence = {'prefix_insert_refusal_budgets': refusals,
                                  'prefix_store_hit_or_restore': False, 'affinity_reuse_allowed': True}
            else:
                prefix_store_spec_hits = []
                for row in turns[1:]:
                    if row['cached_tokens'] <= 0 or row['spec'].get('accepted', 0) <= 0:
                        continue
                    block = re.search(r'^\[meter\] admit id='+re.escape(row['id'])+
                                      r' [^\n]*\n(.*?)(?=^\[meter\] admit id=|\Z)',
                                      current_log, re.M | re.S)
                    assert block is not None, 'missing request admission log: '+row['id']
                    shape = str(row['cached_tokens'])+' of '+str(row['prompt_tokens'])+' prompt tokens'
                    hit = re.search('^'+re.escape('[prefix-cache] hit: '+shape+
                                    ' from cache (model package-probe)')+'$', block[1], re.M)
                    restore = re.search(r'^\[prefix-cache\] spec restore: '+re.escape(shape)+
                                        r' \+ draft plane from cache \[(?:continuation|suffix queued|suffix fed)\]'
                                        r' \(model package-probe\)$', block[1], re.M)
                    if hit is not None and restore is not None and hit.end() < restore.start():
                        prefix_store_spec_hits.append({'turn': row['turn'], 'id': row['id'],
                                                       'hit': hit[0], 'spec_restore': restore[0]})
                assert prefix_store_spec_hits, 'no prefix-store hit with same-request spec restore and acceptance'
                cache_evidence = {'prefix_store_spec_hits': prefix_store_spec_hits}
            assert re.search(r'\[spec-k\] model="package-probe".*K=[1-9][0-9]*', current_log)
            assert '[env-audit] REFUSED' not in current_log and 'registry absent' not in current_log
            all_results.append({'arm': arm, 'turns': turns, 'SSE_frames': len(decoded),
                                'cache_evidence': cache_evidence,
                                'fingerprint': fingerprint, 'external_drafter_engaged': True})
        except BaseException as error:
            primary=error
            raise
        finally:
            h.cleanup_owned(proc,reader,budget,lock_proof,records/'TEARDOWN.json',primary,ownership)
    with budget.operation(end=budget.end):
        assert digest_file(binary) == inputs['binary_sha256']
        for role in ('model', 'draft'):
            row = inputs[role]; path = Path(row['path']); current = path.lstat()
            assert (current.st_dev, current.st_ino, current.st_mode, current.st_size,
                    current.st_mtime_ns, current.st_ctime_ns) == artifact_snapshots[role]
            assert digest_file(path) == row['sha256']
    (root/'RESULT.json').write_text(json.dumps({'source': inputs['source'], 'binary_sha256': inputs['binary_sha256'],
                                               'request_sha256': inputs['request_sha256'], 'results': all_results,
                                               'cached_model_draft_before_after_identity_and_bytes': True,
                                               'scope': 'Package-source identity and vendor-default local serving engagement only.',
                                               'model_or_serving_qualification': False,'whole_cell_budget':budget.report()}, indent=2)+'\n')
    print(json.dumps({'arms': 2, 'turns_per_arm': 8, 'stream_identity_checked': True, 'qualified': False}))

if __name__ == '__main__':
    with h.cancellation():
        main()
