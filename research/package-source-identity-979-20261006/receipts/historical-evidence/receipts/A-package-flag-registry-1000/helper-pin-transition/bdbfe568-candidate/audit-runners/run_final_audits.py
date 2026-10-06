#!/usr/bin/python3 -I
"""Existing eight package audits plus real-workspace retirement census.

Uses the prepared package's shared receiver and corpus authority. No Cargo
invocation, native loop, artifact materialization or corpus generation.
"""
import argparse, base64, hashlib, importlib.util, json, os, re, subprocess, tomllib
from pathlib import Path

TESTS = [
    'os_audit_preserves_utf8_retired_wildcard_and_value_ignorance',
    'os_audit_refuses_owned_bytes_and_summarizes_deduplicated_unowned_names',
    'os_audit_classifies_raw_prefixes_before_rendering_and_keeps_utf8_distinct',
    'os_audit_all_legal_names_accept_uninterpretable_values',
    'registry_is_populated_sorted_and_disjoint',
    'every_legal_name_passes',
    'a_retired_door_refuses_with_its_ledger',
    'unknown_names_refuse_inside_owned_families_and_warn_outside',
]
CENSUS = 'retired_names_are_not_read_at_runtime'
HELPER = 'bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', type=Path, required=True)
    args = parser.parse_args()
    raw = args.inputs.read_bytes()
    assert len(raw) <= 256 * 1024
    inputs = json.loads(raw)
    assert set(inputs) == {'source', 'workspace', 'entry', 'engine_manifest',
                          'engine_artifact', 'receiver_sha256', 'rustc',
                          'audit_source_sha256', 'controls_root'}
    workspace = Path(inputs['workspace']); entry = Path(inputs['entry'])
    manifest = Path(inputs['engine_manifest'])
    for path in (workspace, entry, manifest):
        assert path.is_absolute() and '..' not in path.parts
    support = entry/'build-support'
    assert sha(support/'package_source_identity.py') == HELPER
    assert sha(support/'package_source_rustc.py') == inputs['receiver_sha256']
    spec = importlib.util.spec_from_file_location('receiver', support/'package_source_rustc.py')
    receiver = importlib.util.module_from_spec(spec); spec.loader.exec_module(receiver)
    m = receiver.m
    cap = m.package_capsule(entry)
    owners = [key for key, name in cap['roots'].items()
              if entry/name/'Cargo.toml' == manifest
              and cap['snapshot']['payload']['packages'][key]['name'] == 'memra-engine']
    m.require(len(owners) == 1, 'audit engine owner differs')
    owner = owners[0]
    record = receiver.artifact_record(Path(inputs['engine_artifact']), cap)
    m.require(record['package'] == owner, 'audit artifact owner differs')
    m.require('memra_env_registry_present' in receiver.cfg_options(record['argv'][1:]),
              'actual engine compiler cfg missing')
    out = Path(record['declared_output_root'])
    table = out/'memra_env_registry.rs'; provenance = out/'memra_env_registry.provenance.json'
    m.require(str(table) in record['generated'], 'actual engine registry include missing')
    m.require({str(table), str(provenance)} <= set(record['declared_outputs']),
              'actual registry/provenance declaration missing')
    transport = m.registry_inputs(manifest)
    evidence = m.owned_json(provenance)
    m.require(evidence['source_seal'] == transport['source_seal']
              and evidence['corpus_sha256'] == transport['corpus_sha256']
              and evidence['owner'] == owner and evidence['qualification'] is False,
              'registry output/corpus binding differs')
    m.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=workspace,
                                     text=True).strip() == inputs['source']
              and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=workspace),
              'joint workspace source is not frozen')
    members = transport['workspace_members']
    root_manifest = tomllib.loads(m.regular(workspace, 'Cargo.toml', contents=True).decode())
    m.require(set(root_manifest['workspace']['members']) == set(members.values()),
              'workspace member map differs from independent corpus')
    for name, relative in members.items():
        config = tomllib.loads(m.regular(workspace, relative+'/Cargo.toml', contents=True).decode())
        m.require(config['package']['name'] == name, 'workspace member name differs')

    # The shared interface already validates package/archive/corpus ownership.
    # Here we bind its returned original source bytes to the actual workspace
    # the unchanged ninth test will read at runtime.
    workspace_rows = {}
    for name, encoded in transport['inputs'].items():
        source_bytes = base64.b64decode(encoded, validate=True)
        m.require(m.regular(workspace, name, contents=True) == source_bytes,
                  'joint workspace differs from registry corpus: '+name)
        workspace_rows[name] = m.regular(workspace, name)
    expected_rust = {name for name in transport['inputs']
                     if '/src/' in name and name.endswith('.rs')}
    def runtime_rust_owners():
        actual = set()
        for child in (workspace/'crates').iterdir():
            if not child.is_dir() or not (child/'src').is_dir():
                continue
            descriptor, _ = m.directory(child/'src'); os.close(descriptor)
            files = m.inventory(child/'src', package=False)
            actual.update('crates/'+child.name+'/src/'+name
                          for name in files if name.endswith('.rs'))
        return actual
    m.require(runtime_rust_owners() == expected_rust, 'ninth runtime source walk differs from corpus')
    package_source = manifest.parent/'src/env_audit.rs'
    workspace_manifest = workspace/members['memra-engine']
    workspace_source = workspace_manifest/'src/env_audit.rs'
    m.require(sha(package_source) == sha(workspace_source) == inputs['audit_source_sha256'],
              'package/workspace audit sources differ')
    m.require(tomllib.loads(m.regular(manifest.parent, 'Cargo.toml', contents=True).decode())['package']['edition'] == '2024'
              and tomllib.loads(m.regular(workspace_manifest, 'Cargo.toml', contents=True).decode())['package']['edition'] == '2024',
              'audit source edition differs')
    rustc = Path(inputs['rustc']['path'])
    m.require(inputs['rustc'] == cap['compiler']
              and m.regular(rustc.parent, rustc.name) == inputs['rustc']['file'],
              'audit compiler differs from final native receipt')
    controls = Path(inputs['controls_root'])
    m.require(controls.is_absolute() and not os.path.lexists(controls)
              and not any(controls.is_relative_to(root) or root.is_relative_to(controls)
                          for root in (workspace, entry, out, Path(cap['output']), Path(cap['expectations']))),
              'controls destination exists or overlaps source/custody')
    controls.mkdir()
    before = {str(p): m.regular(p.parent, p.name) for p in (table, provenance, package_source, workspace_source)}
    results = []
    def command(label, argv, cwd, env):
        result = subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=15)
        (controls/(label+'.stdout')).write_bytes(result.stdout)
        (controls/(label+'.stderr')).write_bytes(result.stderr)
        m.immutable_json(controls/(label+'.command.json'),
                         {'argv': argv, 'cwd': str(cwd), 'exit': result.returncode})
        m.require(result.returncode == 0, 'audit control failed: '+label)
        return result.stdout.decode()
    for role, source, actual_manifest, tests in (
        ('package', package_source, manifest.parent, TESTS),
        ('workspace', workspace_source, workspace_manifest, [CENSUS]),
    ):
        m.require(source == actual_manifest/'src/env_audit.rs', 'audit manifest/source ownership differs')
        target = controls/role; target.mkdir()
        name = role+'_env_audit_controls'
        # The source and manifest context identify the same real owner. This
        # direct std-only compile never calls Cargo or the engine build script.
        env = {'PATH': '/usr/bin:/bin', 'LANG': 'C.UTF-8',
               'OUT_DIR': str(out), 'CARGO_MANIFEST_DIR': str(actual_manifest)}
        argv = [str(rustc), '--edition=2024', '--crate-name', name,
                '--cfg', 'memra_env_registry_present', '--check-cfg', 'cfg(memra_env_registry_present)',
                '--test', '--emit=link,dep-info', '--out-dir', str(target), str(source)]
        command(role+'-compile', argv, actual_manifest, env)
        binary = target/name
        listing = command(role+'-listing', [str(binary), '--list', '--format=terse'], actual_manifest, env)
        listed = {line.removesuffix(': test') for line in listing.splitlines() if line.endswith(': test')}
        m.require(listed == {'tests::'+test for test in TESTS+[CENSUS]}, 'compiled audit inventory differs')
        depinfo = m.regular(target, name+'.d', contents=True).decode()
        m.require(str(source) in depinfo and str(table) in depinfo, 'audit compiler read binding missing')
        for test in tests:
            text = command(role+'-'+test, [str(binary), 'tests::'+test, '--exact', '--test-threads=1'],
                           actual_manifest, env)
            m.require(re.search(r'1 passed; 0 failed; 0 ignored;', text) is not None,
                      'audit test did not execute exactly one passing control')
            results.append({'role': role, 'name': test, 'passed': True, 'binary': m.regular(target, name)})
    for name, row in before.items():
        path = Path(name); m.require(m.regular(path.parent, path.name) == row, 'audit source/output drift')
    for name, row in workspace_rows.items():
        m.require(m.regular(workspace, name) == row, 'workspace census input drift')
    m.require(runtime_rust_owners() == expected_rust
              and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=workspace),
              'workspace census membership changed during controls')
    m.require(m.package_capsule(entry) == cap, 'prepared source changed during audit controls')
    m.immutable_json(controls/'RESULT.json',
                     {'source': inputs['source'], 'engine_owner': owner, 'engine_out_dir': str(out),
                      'registry_outputs': before, 'corpus_sha256': transport['corpus_sha256'],
                      'runtime_workspace_census': {'members': members, 'source_rows': workspace_rows},
                      'results': results, 'package_tests': 8, 'workspace_census_tests': 1,
                      'native_build_executed_by_runner': False, 'qualification': False})
    print(json.dumps({'package_tests': 8, 'workspace_census': 1, 'passed': True, 'qualified': False}))

if __name__ == '__main__':
    main()
