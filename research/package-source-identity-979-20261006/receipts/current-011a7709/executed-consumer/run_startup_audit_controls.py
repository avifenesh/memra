#!/usr/bin/python3 -I
"""Nine bounded stock-server audit controls, stopping at an invalid address."""
import argparse, hashlib, importlib.util, json, os, re, subprocess
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def current_primary_source(record,entry,cap,receiver,m):
    owner=record['package'];manifest=entry/cap['roots'][owner];row=cap['snapshot']['payload']['packages'][owner]
    m.require(record['manifest_root']==cap['roots'][owner], 'issued compiler manifest differs')
    return receiver.owned_primary_source(record['argv'][1:],row,manifest)

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args(); inputs = json.loads(args.inputs.read_bytes())
    assert set(inputs) == {'binary', 'binary_file', 'receiver', 'receiver_sha256',
                          'entry', 'registry', 'registry_file', 'record_root', 'runtime_library_path'}
    helper = Path(inputs['receiver']).with_name('package_source_identity.py')
    assert sha(helper) == 'bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39'
    assert sha(Path(inputs['receiver'])) == inputs['receiver_sha256']
    spec = importlib.util.spec_from_file_location('receiver', inputs['receiver'])
    receiver = importlib.util.module_from_spec(spec); spec.loader.exec_module(receiver); m = receiver.m
    binary = Path(inputs['binary']); registry = Path(inputs['registry'])
    assert binary.is_absolute() and registry.is_absolute()
    m.require(m.regular(binary.parent, binary.name) == inputs['binary_file'], 'startup binary differs')
    m.require(m.regular(registry.parent, registry.name) == inputs['registry_file'], 'startup registry differs')
    cap = m.package_capsule(Path(inputs['entry']))
    record = receiver.completed_artifact_record(binary, cap)
    m.require(cap['snapshot']['payload']['packages'][record['package']]['name'] == 'memra-server',
              'startup artifact is not the stock server package')
    m.require(receiver.val(record['argv'][1:], '--crate-name') == 'memra_server'
              and receiver.crate_types(record['argv'][1:]) == ['bin']
              and current_primary_source(record,Path(inputs['entry']),cap,receiver,m)==Path(inputs['entry'])/'src/main.rs',
              'startup artifact is not the stock server target')
    table = m.regular(registry.parent, registry.name, contents=True).decode()
    def names(role):
        match = re.search(r'pub const '+role+r':[^=]+ = &\[(.*?)\];', table, re.S)
        m.require(match is not None, 'registry role missing: '+role)
        return re.findall(r'"(MEMRA_[A-Z0-9_]+)"', match.group(1))
    legal = set(names('LEGAL_NAMES')); prefixes = names('LEGAL_PREFIXES')
    families = names('OWNED_FAMILIES'); retired = names('RETIRED')
    m.require(retired and families and 'MEMRA_ADDR' in legal, 'startup registry nonvacuity failed')
    retired_key = retired[0]
    unknown = next((family+'DEFINITELY_NOT_A_DOOR_ZZ' for family in families
                    if family+'DEFINITELY_NOT_A_DOOR_ZZ' not in legal
                    and not any((family+'DEFINITELY_NOT_A_DOOR_ZZ').startswith(prefix) for prefix in prefixes)), None)
    m.require(unknown is not None, 'owned unknown startup key unavailable')
    root = Path(inputs['record_root'])
    m.require(root.is_absolute() and not os.path.lexists(root)
              and not any(root.is_relative_to(p) or p.is_relative_to(root)
                          for p in (Path(cap['output']), Path(inputs['entry']), Path(cap['expectations']))),
              'startup records overlap custody')
    root.mkdir()
    base = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8', 'MEMRA_ADDR': '127.0.0.1:invalid'}
    # Optional loader path is a separately pinned public runtime profile, never
    # copied from owner ambient values. No other inherited settings or secrets.
    if inputs['runtime_library_path'] is not None:
        assert isinstance(inputs['runtime_library_path'], str)
        base['LD_LIBRARY_PATH'] = inputs['runtime_library_path']
    malformed = families[0].encode()+b'\xff'
    cases = [
        ('clean', {}, 'address', []),
        ('retired-default', {retired_key: '1'}, 'audit', [retired_key, 'retired']),
        ('unknown-owned-default', {unknown: '1'}, 'audit', [unknown, 'unknown']),
        ('retired-warn', {retired_key: '1', 'MEMRA_ENV_AUDIT': 'warn'}, 'address', ['warning', retired_key]),
        ('retired-off-zero', {retired_key: '1', 'MEMRA_ENV_AUDIT': '0'}, 'address', ['audit is OFF']),
        ('owned-warn', {unknown: '1', 'MEMRA_ENV_AUDIT': 'warn'}, 'address', ['warning', unknown]),
        ('unowned-warning', {'MEMRA_ZZ_ORPHAN_LAUNCHER_ONLY': '1'}, 'address', ['warning', 'MEMRA_ZZ_ORPHAN_LAUNCHER_ONLY']),
        ('retired-off-word', {retired_key: '1', 'MEMRA_ENV_AUDIT': 'off'}, 'address', ['audit is OFF']),
        ('malformed-owned-default', {malformed: b'1'}, 'audit', ['\\xFF', 'unknown']),
    ]
    results = []
    for label, extra, stop, needles in cases:
        env = {os.fsencode(k): os.fsencode(v) for k, v in base.items()}
        env.update({os.fsencode(k): os.fsencode(v) for k, v in extra.items()})
        result = subprocess.run([str(binary)], cwd=inputs['entry'], env=env,
                                capture_output=True, timeout=10)
        (root/(label+'.stdout')).write_bytes(result.stdout); (root/(label+'.stderr')).write_bytes(result.stderr)
        stderr = result.stderr.decode('utf-8', errors='backslashreplace')
        m.require(result.returncode != 0 and all(needle in stderr for needle in needles),
                  'startup control outcome differs: '+label)
        if stop == 'audit':
            m.require('[env-audit] REFUSED' in stderr and 'registry absent' not in stderr,
                      'startup audit refusal did not engage: '+label)
            m.require('MEMRA_ADDR="127.0.0.1:invalid" cannot be resolved' not in stderr,
                      'startup passed the expected audit refusal: '+label)
        else:
            m.require('[env-audit] REFUSED' not in stderr
                      and 'MEMRA_ADDR="127.0.0.1:invalid" cannot be resolved' in stderr,
                      'invalid-address stop missing: '+label)
        results.append({'case': label, 'exit': result.returncode, 'stop': stop,
                        'synthetic_env_names_hex': [name.hex() for name in sorted(env)]})
    m.require(m.regular(binary.parent, binary.name) == inputs['binary_file']
              and m.regular(registry.parent, registry.name) == inputs['registry_file'],
              'startup binary/registry changed during controls')
    m.immutable_json(root/'RESULT.json', {'cases': results, 'binary': inputs['binary_file'],
                                         'registry': inputs['registry_file'], 'worker_startup_expected': False,
                                         'model_or_GPU_qualification': False})
    print(json.dumps({'cases': 9, 'passed': True, 'qualified': False}))

if __name__ == '__main__':
    main()
