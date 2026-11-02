#!/usr/bin/python3 -I
"""Bind audit runners to Root's existing reviewed issued/alias selection.

No candidate receipt discovery, output transport, compiler invocation or GPU use.
"""
import argparse, hashlib, importlib.util, json, os, subprocess
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def current_primary_source(record,entry,cap,receiver,m):
    owner=record['package'];manifest=entry/cap['roots'][owner];row=cap['snapshot']['payload']['packages'][owner]
    m.require(record['manifest_root']==cap['roots'][owner], 'issued compiler manifest differs')
    return receiver.owned_primary_source(record['argv'][1:],row,manifest)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--selection', type=Path, required=True,
                        help='Existing Root APPROVED-RESTORE.json issued-path/alias selection')
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    approved = json.loads((args.preparation/'APPROVED-NATIVE.json').read_bytes())
    exit_record = json.loads((Path(approved['records'])/'BUILD-EXIT.json').read_bytes())
    assert exit_record['exit'] == 0 and exit_record['source'] == approved['source']
    entry = Path(approved['entry']); support = entry/'build-support'
    assert sha(support/'package_source_identity.py') == 'bdbfe568d72654c220a7b20df0c678568c7395ed192df329fb8ba61980666a39'
    assert sha(support/'package_source_rustc.py') == approved['source_pins']['tools/package_source_rustc.py']['sha256']
    spec = importlib.util.spec_from_file_location('receiver', support/'package_source_rustc.py')
    receiver = importlib.util.module_from_spec(spec); spec.loader.exec_module(receiver); m = receiver.m
    selected = m.owned_json(args.selection)
    m.require(selected['head'] == approved['source'] and selected['entry'] == str(entry)
              and selected['output'] == approved['output']
              and selected['expectations'] == approved['expectations'], 'issued selection/native roles differ')
    m.require(subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=args.workspace,
                                     text=True).strip() == approved['source']
              and not subprocess.check_output(['git', 'status', '--porcelain'], cwd=args.workspace),
              'audit workspace source differs')
    cap = m.package_capsule(entry)
    engine = []; server = []
    for name in selected['issued']:
        path = Path(name); record = receiver.completed_artifact_record(path, cap)
        package = cap['snapshot']['payload']['packages'][record['package']]['name']
        if package == 'memra-engine' and record['artifact_role'] == 'rlib':
            engine.append((path, record))
        if package == 'memra-server' and record['artifact_role'] == 'binary' \
                and receiver.val(record['argv'][1:], '--crate-name') == 'memra_server' \
                and current_primary_source(record,entry,cap,receiver,m)==entry/'src/main.rs':
            server.append((path, record))
    m.require(len(engine) == len(server) == 1, 'actual audit engine/stock-server producer is not unique')
    engine_path, engine_record = engine[0]; server_path, server_record = server[0]
    manifest = entry/cap['roots'][engine_record['package']]/'Cargo.toml'
    out = Path(engine_record['declared_output_root'])
    table = out/'memra_env_registry.rs'
    m.require(str(table) in engine_record['generated'], 'actual engine did not include the registry')
    m.require(str(out/'memra_env_registry.provenance.json') in engine_record['declared_outputs'],
              'actual engine provenance sidecar is not bound')
    # Use the actual issued hashed BIN producer. Cargo's release/memra-server
    # alias can share bytes without having a separately issued receipt.
    for alias, producer in selected['aliases'].items():
        if producer == str(server_path):
            alias = Path(alias)
            m.require(m.regular(alias.parent, alias.name) == server_record['artifact'],
                      'stock-server Cargo alias differs')
    root = args.destination
    m.require(root.is_absolute() and not os.path.lexists(root), 'audit binding destination exists')
    m.require(not any(root.is_relative_to(p) or p.is_relative_to(root)
                      for p in (args.workspace, entry, Path(cap['output']), Path(cap['expectations']))),
              'audit binding destination overlaps custody')
    root.mkdir()
    audit = {'source': approved['source'], 'workspace': str(args.workspace), 'entry': str(entry),
             'engine_manifest': str(manifest), 'engine_artifact': str(engine_path),
             'receiver_sha256': sha(support/'package_source_rustc.py'), 'rustc': cap['compiler'],
             'audit_source_sha256': sha(manifest.parent/'src/env_audit.rs'),
             'controls_root': str(root/'audit-results')}
    startup = {'binary': str(server_path), 'binary_file': server_record['artifact'],
               'receiver': str(support/'package_source_rustc.py'),
               'receiver_sha256': audit['receiver_sha256'], 'entry': str(entry),
               'registry': str(table), 'registry_file': m.regular(table.parent, table.name),
               'record_root': str(root/'startup-results'), 'runtime_library_path': None}
    m.immutable_json(root/'AUDIT-INPUTS.json', audit)
    m.immutable_json(root/'STARTUP-INPUTS.json', startup)
    m.immutable_json(root/'BINDING-RESULT.json',
                     {'source': approved['source'], 'selected_issued_paths': selected['issued'],
                      'selection_file': m.regular(args.selection.parent, args.selection.name),
                      'native_exit_file': m.regular(Path(approved['records']), 'BUILD-EXIT.json'),
                      'actual_server_producer': str(server_path), 'registry': str(table),
                      'audit_runner_executed': False, 'startup_runner_executed': False,
                      'qualified': False})
    print(json.dumps({'actual_bindings_created': True, 'audit_executed': False, 'qualified': False}))

if __name__ == '__main__':
    main()
