#!/usr/bin/env python3
"""Prepare and verify finite package source custody; receivers never invoke Cargo."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import re
import sys
sys.dont_write_bytecode = True

SCHEMA = 'memra-package-input-prototype-v1'
RESERVED = '.memra-package-source.json'
MAX_JSON = 64 * 1024 * 1024
SHA256 = re.compile('[0-9a-f]{64}')


def file_entry_shape(row):
    require(type(row) is dict and set(row) == {'sha256', 'bytes', 'mode'}
            and type(row['bytes']) is int and row['bytes'] >= 0
            and type(row['sha256']) is str and SHA256.fullmatch(row['sha256'])
            and row['mode'] in ('100644', '100755'), 'invalid source file identity')


class Refused(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Refused(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def json_bytes(raw):
    require(type(raw) is bytes and len(raw) <= MAX_JSON, 'JSON input exceeds bound')
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def bad_constant(value):
        raise Refused('nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=object_pairs, parse_constant=bad_constant)


def directory(root):
    """Pin every absolute ancestor without following a path alias."""
    path = Path(root).absolute()
    require('..' not in path.parts, 'noncanonical source root')
    descriptor = os.open('/', os.O_RDONLY | os.O_DIRECTORY)
    identities = []
    try:
        for part in path.parts[1:]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor); descriptor = child
            info = os.fstat(descriptor)
            identities.append((info.st_dev, info.st_ino, info.st_mode))
        return descriptor, identities
    except BaseException:
        os.close(descriptor)
        raise


def regular(root, name, *, contents=False):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == name and not any(c in name for c in '\0\n\r\\'),
            'noncanonical source path')
    parent, root_identity = directory(root)
    ancestor_identity = []
    descriptor = None
    try:
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = child
            info = os.fstat(parent)
            ancestor_identity.append((info.st_dev, info.st_ino, info.st_mode))
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), 'source is not a regular file: ' + name)
        hasher = hashlib.sha256()
        length = 0
        blocks = []
        while True:
            data = os.read(descriptor, 1024 * 1024)
            if not data:
                break
            hasher.update(data); length += len(data)
            if contents:
                require(length <= MAX_JSON, 'owned JSON input exceeds bound')
                blocks.append(data)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                'source changed while read: ' + name)
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        require((current.st_dev, current.st_ino, current.st_size, current.st_mode,
                 current.st_mtime_ns, current.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mode,
                    after.st_mtime_ns, after.st_ctime_ns), 'source pathname replaced: ' + name)
        check, fresh_root = directory(root)
        fresh_ancestors = []
        try:
            for part in path.parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=check)
                os.close(check); check = child
                info = os.fstat(check)
                fresh_ancestors.append((info.st_dev, info.st_ino, info.st_mode))
        finally:
            os.close(check)
        require(fresh_root == root_identity and fresh_ancestors == ancestor_identity,
                'source ancestor replaced: ' + name)
        if contents:
            return b''.join(blocks)
        return {'sha256': hasher.hexdigest(), 'bytes': length,
                'mode': '100755' if before.st_mode & stat.S_IXUSR else '100644'}
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def inventory(root, *, package=True):
    """All physical package resources, except named output/history/derived metadata."""
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), 'source root is not a contained directory')
    def membership():
        paths, directories, leaves = [], {}, {}
        pinned, identity = directory(root); os.close(pinned)
        for parent, dirs, files in os.walk(root, followlinks=False):
            info = os.lstat(parent)
            require(stat.S_ISDIR(info.st_mode), 'source directory type changed')
            directories[Path(parent).relative_to(root).as_posix()] = (info.st_dev, info.st_ino, info.st_mode)
            for name in list(dirs):
                path = Path(parent) / name
                require(not path.is_symlink(), 'source directory link is not admitted')
                # Only an actual package-root Git history directory is a role;
                # src/target and nested .git are source, not blanket exclusions.
                if package and path == root / '.git' and (path / 'HEAD').is_file() and (path / 'objects').is_dir():
                    dirs.remove(name)
            for name in files:
                relative = (Path(parent) / name).relative_to(root).as_posix()
                if package and relative in (RESERVED, '.cargo_vcs_info.json', '.cargo-ok'):
                    continue
                paths.append(relative)
                info = os.lstat(Path(parent) / name)
                leaves[relative] = (info.st_dev, info.st_ino, info.st_size, info.st_mode,
                                    info.st_mtime_ns, info.st_ctime_ns)
        return sorted(paths), directories, identity, leaves
    paths, dirs, identity, leaves = membership()
    require(paths and (not package or 'Cargo.toml' in paths), 'source inventory is empty')
    result = {name: regular(root, name) for name in paths}
    require(membership() == (paths, dirs, identity, leaves), 'source membership or directory binding changed')
    return result


def package_key(package):
    source = package.get('source') or 'path-source'
    # Git revision remains provenance. Source bytes, not rewritten history,
    # distinguish the package-source prototype's logical content keys.
    if source.startswith('git+'):
        source = source.split('#', 1)[0]
    return package['name'] + '@' + package['version'] + ':' + source


def snapshot_shape(snapshot):
    require(type(snapshot) is dict and set(snapshot) == {'payload', 'sha256', 'bindings', 'resolver', 'workspace_root_observed'},
            'unknown source prototype schema')
    payload = snapshot['payload']
    require(type(payload) is dict and set(payload) == {'schema', 'domain', 'entry', 'target', 'features', 'packages', 'qualification', 'closure_limit'}
            and payload['schema'] == SCHEMA and payload['qualification'] is False,
            'unknown source payload schema')
    require(type(snapshot['sha256']) is str and SHA256.fullmatch(snapshot['sha256']), 'invalid source seal')
    require(all(type(payload[k]) is str and payload[k] for k in ('domain', 'entry', 'target', 'closure_limit'))
            and payload['domain'] == 'unqualified-package-source-input-prototype'
            and type(payload['features']) is list and all(type(x) is str for x in payload['features']),
            'invalid source domain/recipe shape')
    require(type(payload['packages']) is dict and 0 < len(payload['packages']) <= 10000,
            'invalid package inventory')
    require(type(snapshot['bindings']) is dict and set(snapshot['bindings']) == set(payload['packages']),
            'source binding inventory differs')
    require(type(snapshot['resolver']) is list and snapshot['resolver']
            and all(type(x) is str for x in snapshot['resolver'])
            and type(snapshot['workspace_root_observed']) is str
            and snapshot['workspace_root_observed'], 'invalid resolver evidence')
    require(payload['entry'] in payload['packages'], 'entry package absent')
    for name, row in payload['packages'].items():
        require(type(name) is str and type(row) is dict and set(row) == {'name', 'version', 'source', 'features', 'dependencies', 'files'},
                'unknown package source schema')
        require(type(row['files']) is dict and 0 < len(row['files']) <= 100000, 'invalid file inventory')
        require(all(type(row[k]) is str and row[k] for k in ('name', 'version'))
                and (row['source'] is None or type(row['source']) is str)
                and all(type(row[k]) is list and all(type(x) is str for x in row[k])
                        for k in ('features', 'dependencies')), 'invalid package source values')
        for path, entry in row['files'].items():
            require(type(path) is str, 'invalid source file path')
            file_entry_shape(entry)
            parsed = PurePosixPath(path)
            require(path and not parsed.is_absolute() and '..' not in parsed.parts
                    and parsed.as_posix() == path, 'invalid source inventory path')
        binding = snapshot['bindings'][name]
        require(type(binding) is dict and set(binding) == {'cargo_package_id', 'manifest_path'}
                and all(type(x) is str and x for x in binding.values()), 'invalid actual source binding')


def owned_json(path):
    path = Path(path).absolute()
    return json_bytes(regular(path.parent, path.name, contents=True))


def write_owned(path, result, inputs):
    path = Path(path).absolute()
    require(not any(path == root or root in path.parents for root in inputs), 'output would overwrite admitted source inputs')
    parent, identity = directory(path.parent)
    descriptor = None
    try:
        raw = canonical(result) + b'\n'
        require(len(raw) <= MAX_JSON, 'output descriptor exceeds bound')
        descriptor = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                             0o600, dir_fd=parent)
        with os.fdopen(descriptor, 'wb') as stream:
            descriptor = None
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            opened = os.fstat(stream.fileno())
            current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            require((opened.st_dev, opened.st_ino) == (current.st_dev, current.st_ino), 'output pathname replaced')
        check, current_identity = directory(path.parent); os.close(check)
        require(identity == current_identity, 'output ancestor replaced')
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def cargo_graph(manifest, target, features):
    """Nonreentrant producer only; do not invoke from a Cargo build script."""
    require('OUT_DIR' not in os.environ and 'CARGO_MANIFEST_DIR' not in os.environ,
            'source resolver cannot run recursively under Cargo')
    manifest = Path(manifest).absolute()
    regular(manifest.parent, manifest.name)
    command = ['cargo', 'metadata', '--offline', '--locked', '--format-version', '1',
               '--filter-platform', target, '--manifest-path', str(manifest)]
    if features:
        command.extend(['--features', ','.join(features)])
    result = subprocess.run(command, cwd=manifest.parent, capture_output=True, check=False)
    require(result.returncode == 0, 'actual locked Cargo resolution failed; closure unavailable')
    require(len(result.stdout) <= 64 * 1024 * 1024, 'Cargo graph exceeds prototype bound')
    graph = json_bytes(result.stdout)
    root_fields = {'build_directory', 'metadata', 'packages', 'resolve', 'target_directory', 'version',
                   'workspace_default_members', 'workspace_members', 'workspace_root'}
    require(type(graph) is dict and set(graph) == root_fields
            and type(graph['version']) is int and graph['version'] == 1
            and type(graph['packages']) is list and 0 < len(graph['packages']) <= 10000,
            'unknown Cargo metadata schema')
    require(graph.get('resolve') is not None, 'Cargo graph has no resolved dependency bindings')
    require(type(graph['resolve']) is dict and set(graph['resolve']) == {'nodes', 'root'}
            and type(graph['resolve']['nodes']) is list, 'unknown resolved Cargo graph schema')
    for row in graph['packages']:
        package_fields = {'authors', 'categories', 'default_run', 'dependencies', 'description',
                          'documentation', 'edition', 'features', 'homepage', 'id', 'keywords',
                          'license', 'license_file', 'links', 'manifest_path', 'metadata', 'name',
                          'publish', 'readme', 'repository', 'rust_version', 'source', 'targets', 'version'}
        require(type(row) is dict and set(row) == package_fields
                and all(type(row[k]) is str and row[k] for k in ('id', 'name', 'version', 'manifest_path')),
                'invalid Cargo package binding')
    packages = {row['id']: row for row in graph['packages']}
    require(len(packages) == len(graph['packages']), 'duplicate Cargo package identity')
    entry = [row for row in packages.values() if Path(row['manifest_path']).absolute() == manifest]
    require(len(entry) == 1, 'canonical entry does not have exactly one Cargo binding')
    nodes = {row['id']: row for row in graph['resolve']['nodes']}
    require(len(nodes) == len(graph['resolve']['nodes']), 'duplicate resolved package node')
    require(set(nodes) == set(packages), 'resolved nodes do not bind all Cargo packages')
    for row in nodes.values():
        require(set(row) == {'dependencies', 'deps', 'features', 'id'} and type(row['deps']) is list
                and type(row['features']) is list and all(type(x) is str for x in row['features']),
                'unknown resolved package node schema')
        for dep in row['deps']:
            require(type(dep) is dict and set(dep) == {'dep_kinds', 'name', 'pkg'}
                    and dep['pkg'] in packages and type(dep['dep_kinds']) is list,
                    'invalid resolved dependency binding')
            for kind in dep['dep_kinds']:
                require(type(kind) is dict and set(kind) == {'kind', 'target'}
                        and kind['kind'] in (None, 'build', 'dev'), 'unknown dependency kind')
    reached = {entry[0]['id']}
    while True:
        following = reached | {dep['pkg'] for key in reached for dep in nodes[key]['deps']
                               if any(kind.get('kind') in (None, 'build') for kind in dep['dep_kinds'])}
        if following == reached:
            break
        reached = following
    logical, bindings = {}, {}
    for key in sorted(reached):
        package, node = packages[key], nodes[key]
        identity = package_key(package)
        require(identity not in logical, 'ambiguous same-name/version source package')
        source_root = Path(package['manifest_path']).parent
        files = inventory(source_root)
        deps = sorted(package_key(packages[dep['pkg']]) for dep in node['deps']
                      if dep['pkg'] in reached and any(k.get('kind') in (None, 'build') for k in dep['dep_kinds']))
        source = package.get('source')
        if source and source.startswith('git+'):
            source = source.split('#', 1)[0]
        logical[identity] = {'name': package['name'], 'version': package['version'],
                             'source': source, 'features': sorted(node['features']),
                             'dependencies': deps, 'files': files}
        bindings[identity] = {'cargo_package_id': key, 'manifest_path': str(package['manifest_path'])}
    payload = {'schema': SCHEMA, 'domain': 'unqualified-package-source-input-prototype',
               'entry': package_key(entry[0]), 'target': target, 'features': sorted(features),
               'packages': logical, 'qualification': False,
               'closure_limit': 'Resolved source roots only; compiler binding, native/generated/external inputs not admitted yet'}
    return {'payload': payload, 'sha256': digest(payload), 'bindings': bindings,
            'resolver': command, 'workspace_root_observed': graph['workspace_root']}


def verify(snapshot, current):
    snapshot_shape(snapshot); snapshot_shape(current)
    require(digest(snapshot['payload']) == snapshot['sha256'], 'source prototype seal differs')
    require(snapshot['payload'] == current['payload'] and snapshot['sha256'] == current['sha256'],
            'actual resolved source bytes/modes/membership/features differ')
    # Paths are physical bindings, not logical identity. Relocation must be
    # explicitly reobserved; a descriptor is not proof of compiler consumption.
    return {'schema': SCHEMA, 'inputs_verified': True, 'qualification': False,
            'sha256': current['sha256'], 'bindings': current['bindings'],
            'compiler_binding_verified': False, 'package_source_marker_admitted': False}


def compiler_bindings(snapshot, messages):
    """Producer-side actual Cargo artifacts; still not native/generated closure."""
    observed = {}
    for line in messages.splitlines():
        row = json_bytes(line.encode())
        if row.get('reason') != 'compiler-artifact':
            continue
        key = row.get('package_id')
        require(type(key) is str and type(row.get('manifest_path')) is str,
                'Cargo artifact lacks an actual source binding')
        binding = {'manifest_path': row['manifest_path'], 'features': sorted(row.get('features', []))}
        if key in observed:
            require(observed[key] == binding, 'ambiguous compiled source binding')
        observed[key] = binding
    require(observed, 'no actual compiled package binding')
    for name, expected in snapshot['bindings'].items():
        actual = observed.get(expected['cargo_package_id'])
        require(actual is not None and actual['manifest_path'] == expected['manifest_path'],
                'resolved source is not the compiler-reported source: ' + name)
        require(actual['features'] == snapshot['payload']['packages'][name]['features'],
                'compiled features differ from resolved source: ' + name)
    return {'source_bindings_verified': True, 'qualification': False,
            'package_source_marker_admitted': False,
            'closure_limit': 'Cargo source binding only; native/generated/external operative closure remains pending'}


# Prepared package contract. Cargo runs only in prepare_package, never receive.
PACKAGE_SCHEMA = 'memra-package-source-v1'
PACKAGE_FIELDS = {'schema', 'snapshot', 'roots', 'recipe', 'supplementary', 'output',
                  'expectations', 'compiler', 'ambient', 'source_seal', 'seal'}
METADATA_ENV = {'MEMRA_BUILD_ID', 'MEMRA_BUILD_ID_SRC', 'MEMRA_BUILD_ID_NOTE', 'MEMRA_BUILD_SHA'}


def identity(domain, value):
    """FNV metadata label, not an integrity or qualification seal."""
    h = 0x6c62272e07bb014262b821756295c58d
    for byte in domain.encode() + b'\0' + canonical(value):
        h = ((h ^ byte) * 0x1000000000000000000013b) & ((1 << 128) - 1)
    return f'{h >> 80:012x}'


def immutable_json(path, value):
    """Publish a complete no-follow owned leaf atomically; never replace history."""
    import secrets
    path = Path(path).absolute()
    parent, before = directory(path.parent)
    temporary = '.memra-write-' + secrets.token_hex(16)
    fd = None
    try:
        raw = canonical(value) + b'\n'
        require(len(raw) <= MAX_JSON, 'derived JSON exceeds bound')
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        with os.fdopen(fd, 'wb') as stream:
            fd = None
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        check, after = directory(path.parent); os.close(check)
        require(before == after, 'derived output ancestor replaced')
        try:
            os.link(temporary, path.name, src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
        except FileExistsError:
            require(regular(path.parent, path.name, contents=True) == raw, 'immutable custody collision: ' + str(path))
        os.fsync(parent)
        check, after = directory(path.parent); os.close(check)
        require(before == after, 'derived output ancestor replaced')
    finally:
        if fd is not None: os.close(fd)
        try: os.unlink(temporary, dir_fd=parent)
        except FileNotFoundError: pass
        os.close(parent)


def relative_name(value):
    require(type(value) is str, 'relative source role is not text')
    p = PurePosixPath(value)
    require(value and not p.is_absolute() and '..' not in p.parts and p.as_posix() == value
            and not any(c in value for c in '\0\n\r\\'), 'invalid source role path')
    return value


def recipe_shape(recipe, packages):
    fields = {'config', 'wrapper', 'target', 'env', 'generated', 'cfgs', 'codegen', 'targets'}
    require(type(recipe) is dict and set(recipe) == fields, 'unknown supported recipe')
    require(recipe['wrapper'] == 'build-support/package_source_rustc.py'
            and type(recipe['target']) is str and recipe['target'].endswith('-unknown-linux-gnu'), 'unsupported recipe platform')
    for name in ('env', 'generated', 'cfgs'):
        rows = recipe[name]
        require(type(rows) is dict and set(rows) == set(packages), 'missing package ' + name + ' declarations')
        for items in rows.values():
            require(type(items) is list and len(items) <= 10000 and len(set(items)) == len(items)
                    and all(type(x) is str for x in items), 'invalid recipe ' + name)
            if name == 'generated':
                for item in items: relative_name(item)
            if name == 'env':
                require(all(re.fullmatch('[A-Z][A-Z0-9_]{0,100}', item) and item not in METADATA_ENV
                            for item in items), 'invalid nonidentity env declaration')
    require(type(recipe['codegen']) is dict and len(recipe['codegen']) <= 32
            and all(type(k) is str and type(v) is list and len(v) <= 32 and all(type(x) is str for x in v) for k, v in recipe['codegen'].items())
            and type(recipe['targets']) is list
            and recipe['targets'] and all(type(x) is str for x in recipe['targets']), 'invalid compiler context')


def ambient_config(root):
    import tomllib
    rows = {}
    candidates = [(f'ancestor:{index}:{name}', parent / '.cargo' / name)
                  for index, parent in enumerate(root.parents)
                  for name in ('config', 'config.toml')]
    cargo_dir = Path(os.environ.get('CARGO_HOME', str(Path.home() / '.cargo')))
    candidates += [('cargo-home:' + name, cargo_dir / name) for name in ('config', 'config.toml')]
    require(not os.path.lexists(root / '.cargo/config'), 'ambiguous local Cargo config')
    for role, path in candidates:
        if not os.path.lexists(path): continue
        raw = regular(path.parent, path.name, contents=True)
        cfg = tomllib.loads(raw.decode())
        # The package's nearer exact wrapper config overrides this one known
        # build setting. Other source/profile/default controls are unsupported.
        require(set(cfg) == {'build'} and set(cfg['build']) == {'rustc-wrapper'}
                and type(cfg['build']['rustc-wrapper']) is str, 'unknown ambient Cargo controls')
        rows[role] = regular(path.parent, path.name)
    return rows

def package_capsule(root):
    root = Path(root).absolute()
    require(not os.path.lexists(root / '.memra-source-freshness-required'), 'freshness sentinel must remain absent')
    cap = owned_json(root / RESERVED)
    require(type(cap) is dict and set(cap) == PACKAGE_FIELDS and cap['schema'] == PACKAGE_SCHEMA,
            'unknown package source capsule')
    require(cap['seal'] == digest({k: v for k, v in cap.items() if k != 'seal'}), 'capsule seal differs')
    snapshot_shape(cap['snapshot'])
    snap = cap['snapshot']
    require(digest(snap['payload']) == snap['sha256'], 'source snapshot seal differs')
    packages = snap['payload']['packages']
    require(type(cap['roots']) is dict and set(cap['roots']) == set(packages), 'source roots differ')
    for key, name in cap['roots'].items():
        relative_name(name)
        require(inventory(root / name) == packages[key]['files'], 'fresh package source/mode/membership differs: ' + key)
    recipe_shape(cap['recipe'], packages)
    require(ambient_config(root) == cap['ambient'], 'ambient Cargo config differs')
    require(not any(key.startswith('CARGO_SOURCE_') or key in ('CARGO_BUILD_RUSTC','CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER') for key in os.environ), 'unknown Cargo source/builder environment')
    import tomllib
    config = regular(root, '.cargo/config.toml', contents=True)
    require(tomllib.loads(config.decode()) == cap['recipe']['config'], 'source config differs')
    cfg = cap['recipe']['config']
    require(set(cfg) == {'source', 'build'} and set(cfg['build']) == {'rustc-wrapper'}
            and cfg['build']['rustc-wrapper'] == cap['recipe']['wrapper'], 'unknown Cargo build config')
    require(cfg['source'] == {'crates-io': {'replace-with': 'memra-package-source'},
                             'memra-package-source': {'directory': 'vendor'}}, 'unknown Cargo source replacement')
    for name in ('output', 'expectations'):
        require(type(cap[name]) is str and Path(cap[name]).is_absolute()
                and not Path(cap[name]).is_relative_to(root) and not root.is_relative_to(Path(cap[name])), 'derived role overlaps source')
        fd, _ = directory(cap[name]); os.close(fd)
    require(not Path(cap['expectations']).is_relative_to(Path(cap['output']))
            and not Path(cap['output']).is_relative_to(Path(cap['expectations'])), 'custody roles overlap')
    require(type(cap['compiler']) is dict and set(cap['compiler']) == {'path', 'file'}, 'unknown builder')
    compiler = Path(cap['compiler']['path'])
    require(compiler.is_absolute() and regular(compiler.parent, compiler.name) == cap['compiler']['file'], 'compiler bytes/mode differ')
    supplemental = cap['supplementary']
    require(type(supplemental) is dict, 'supplementary input shape differs')
    for key, row in supplemental.items():
        require(key.startswith('supplementary:') and type(row) is dict
                and set(row) == {'owner', 'role', 'root', 'files'} and row['owner'] in packages,
                'supplementary owner is not an actual Cargo package')
        relative_name(row['root'])
        require(type(row['files']) is dict and row['files'], 'empty supplementary custody')
        for path, expected in row['files'].items():
            file_entry_shape(expected)
            require(regular(root / row['root'], relative_name(path)) == expected, 'supplementary source differs')
        require(inventory(root / row['root'], package=False) == row['files'], 'supplementary membership differs')
    logical = {'source': snap['payload'], 'recipe': cap['recipe'], 'supplementary': supplemental, 'ambient': cap['ambient']}
    require(cap['source_seal'] == digest(logical), 'logical package source seal differs')
    return cap


def receive(root):
    cap = package_capsule(root)
    wrapper = os.environ.get('RUSTC_WRAPPER', '')
    if not wrapper: return 'degraded'
    require(Path(wrapper).absolute() == Path(root).absolute() / cap['recipe']['wrapper'], 'unmatched effective wrapper')
    require(not os.environ.get('RUSTC_WORKSPACE_WRAPPER'), 'nested compiler receiver unsupported')
    require(os.environ.get('TARGET', cap['recipe']['target']) == cap['recipe']['target'], 'package target differs')
    return 'prepared'


def prepare_package(manifest, target, features, declarations, output, expectations):
    root = Path(manifest).absolute().parent
    require(root / RESERVED != declarations.absolute(), 'declarations cannot be own capsule')
    plan = owned_json(declarations)
    require(type(plan) is dict and set(plan) == {'env', 'generated', 'cfgs', 'codegen', 'targets', 'supplementary'}, 'unknown source declaration plan')
    # Source capture runs before Cargo acquires build-script locks.
    snapshot = cargo_graph(manifest, target, features)
    require(Path(snapshot['workspace_root_observed']) == root, 'normal package borrowed a parent workspace')
    roots = {}
    for key, binding in snapshot['bindings'].items():
        path = Path(binding['manifest_path']).parent
        require(path.is_relative_to(root), 'resolved package outside owned recipe')
        roots[key] = path.relative_to(root).as_posix()
    import tomllib
    recipe = {'config': tomllib.loads(regular(root, '.cargo/config.toml', contents=True).decode()),
              'wrapper': 'build-support/package_source_rustc.py', 'target': target,
              **{key: plan[key] for key in ('env', 'generated', 'cfgs', 'codegen', 'targets')}}
    sysroot = subprocess.run(['rustc', '--print', 'sysroot'], capture_output=True, check=True).stdout.decode().strip()
    compiler = Path(sysroot) / 'bin/rustc'
    cap = {'schema': PACKAGE_SCHEMA, 'snapshot': snapshot, 'roots': roots, 'recipe': recipe,
           'supplementary': plan['supplementary'], 'output': str(output.absolute()),
           'expectations': str(expectations.absolute()),
           'compiler': {'path': str(compiler), 'file': regular(compiler.parent, compiler.name)}, 'ambient': ambient_config(root)}
    cap['source_seal'] = digest({'source': snapshot['payload'], 'recipe': recipe, 'supplementary': cap['supplementary'], 'ambient': cap['ambient']})
    cap['seal'] = digest(cap)
    immutable_json(root / RESERVED, cap)
    package_capsule(root)
    return cap


MAX_REGISTRY_RESPONSE = MAX_JSON
# Source-and-byte transport only. A derives semantic parser/member roles.
def registry_inputs(engine_manifest):
    import base64
    manifest = Path(engine_manifest).absolute()
    require(manifest.name == 'Cargo.toml', 'registry caller requires Cargo manifest')
    candidates = []
    if os.path.lexists(manifest.parent / RESERVED):
        candidates.append(manifest.parent)  # Actual engine ENTRY role.
    if manifest.parent.parent.name == 'vendor':
        entry = manifest.parent.parent.parent
        if os.path.lexists(entry / RESERVED):
            candidates.append(entry)  # Exact engine dependency role.
    require(len(candidates) == 1, 'missing or ambiguous finite registry entry role')
    entry = candidates[0]
    cap = package_capsule(entry)
    owners = [key for key, path in cap['roots'].items()
              if entry / path / 'Cargo.toml' == manifest
              and cap['snapshot']['payload']['packages'][key]['name'] == 'memra-engine']
    require(len(owners) == 1, 'registry caller is not the actual owned engine package')
    owner = owners[0]
    attachments = [row for row in cap['supplementary'].values()
                   if row['owner'] == owner and row['role'] == 'env-registry-source-v1']
    require(len(attachments) == 1, 'missing or ambiguous registry source attachment')
    attachment = attachments[0]
    roots = cap['snapshot']['payload']['packages'][owner]['files']
    planned = [(('crates/memra-engine/' + path), expected) for path, expected in roots.items()
               if path.startswith('src/') and path.endswith('.rs')]
    planned += list(attachment['files'].items())
    require(len(planned) <= 10000 and len({name for name, _ in planned}) == len(planned),
            'registry response input count/owner collision')
    # Prepared BEFORE transport from independently pinned actual archives and
    # original workspace/FLAGS/probe inputs. Neither reference nor corpus lives
    # in candidate source/output; capsule declarations cannot replace them.
    reference = owned_json(Path(cap['expectations']) / ('registry-corpus-' + cap['source_seal'] + '.json'))
    require(type(reference) is dict and set(reference) == {'schema', 'owner', 'corpus_sha256'}
            and reference['schema'] == 'memra-registry-corpus-reference-v1'
            and reference['owner'] == owner and type(reference['corpus_sha256']) is str
            and SHA256.fullmatch(reference['corpus_sha256']), 'registry producer corpus reference differs')
    corpus = owned_json(Path(cap['expectations']) / (reference['corpus_sha256'] + '.registry-corpus.json'))
    require(type(corpus) is dict and set(corpus) == {'schema', 'workspace_members', 'inputs', 'engine_manifest'}
            and corpus['schema'] == 'memra-registry-original-corpus-v1'
            and digest(corpus) == reference['corpus_sha256'], 'registry independent corpus commitment differs')
    require(type(corpus['workspace_members']) is dict and corpus['workspace_members']
            and all(type(name) is str and type(path) is str for name, path in corpus['workspace_members'].items())
            and type(corpus['inputs']) is dict and corpus['inputs'], 'registry original corpus shape differs')
    file_entry_shape(corpus['engine_manifest'])
    require(roots['Cargo.toml'] == corpus['engine_manifest'], 'registry normalized engine manifest differs from original corpus')
    for path, row in corpus['inputs'].items():
        relative_name(path); file_entry_shape(row)
    for path in corpus['workspace_members'].values():
        relative_name(path)
    require(dict(planned) == corpus['inputs'], 'registry input set/bytes/modes differs from independent original corpus')
    response = {'schema': 'memra-registry-source-inputs-v2', 'owner': owner,
                'source_seal': cap['source_seal'], 'corpus_sha256': reference['corpus_sha256'],
                'workspace_members': corpus['workspace_members'], 'inputs': {},
                'engine_manifest': '', 'qualification': False}
    encoded_budget = sum(4 * ((row['bytes'] + 2) // 3) + len(canonical(name)) + 3 for name, row in planned)
    encoded_budget += max(0, len(planned) - 1)  # Entry commas inside existing {}.
    encoded_budget += 4 * ((corpus['engine_manifest']['bytes'] + 2) // 3)
    require(encoded_budget + len(canonical(response)) <= MAX_REGISTRY_RESPONSE,
            'registry response aggregate exceeds bound')
    bodies = {}
    def admit(name, root, path, expected):
        require(name not in bodies, 'duplicate canonical registry input owner')
        raw = regular(root, path, contents=True)
        require(hashlib.sha256(raw).hexdigest() == expected['sha256'] and len(raw) == expected['bytes']
                and regular(root, path) == expected, 'registry input changed during return')
        bodies[name] = base64.b64encode(raw).decode('ascii')
    for path, expected in roots.items():
        if path.startswith('src/') and path.endswith('.rs'):
            admit('crates/memra-engine/' + path, manifest.parent, path, expected)
    for path, expected in attachment['files'].items():
        admit(path, entry / attachment['root'], path, expected)
    require('docs/FLAGS.md' in bodies, 'registry FLAGS ownership missing')
    engine_manifest = regular(manifest.parent, 'Cargo.toml', contents=True)
    require(hashlib.sha256(engine_manifest).hexdigest() == corpus['engine_manifest']['sha256']
            and len(engine_manifest) == corpus['engine_manifest']['bytes'], 'engine manifest changed during return')
    response['inputs'] = bodies
    response['engine_manifest'] = base64.b64encode(engine_manifest).decode('ascii')
    require(package_capsule(entry) == cap, 'registry capsule/source identity changed during return')
    require(len(canonical(response)) <= MAX_REGISTRY_RESPONSE, 'registry final response exceeds bound')
    return response


def production_main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare-package', 'receive', 'rederive', 'registry-inputs'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--target')
    parser.add_argument('--features', default='')
    parser.add_argument('--declarations', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--expectations', type=Path)
    parser.add_argument('--identity')
    a = parser.parse_args(args)
    if a.command == 'prepare-package':
        require(all((a.target, a.declarations, a.output, a.expectations)), 'missing source recipe inputs')
        cap = prepare_package(a.manifest, a.target, a.features.split(',') if a.features else [], a.declarations, a.output, a.expectations)
        print(cap['source_seal'])
    elif a.command == 'receive': print(receive(a.manifest.absolute().parent))
    elif a.command == 'registry-inputs':
        require(not any((a.target, a.features, a.declarations, a.output, a.expectations, a.identity)),
                'registry transport accepts only the actual engine manifest')
        print(canonical(registry_inputs(a.manifest)).decode())
    else:
        cap = package_capsule(a.manifest.absolute().parent)
        require(type(a.identity) is str and re.fullmatch('[0-9a-f]{12}', a.identity), 'rederivation requires expected identity')
        value = owned_json(Path(cap['output']) / ('identity-' + a.identity + '.json'))
        require(set(value) == {'source_seal', 'tuple', 'bindings'} and value['source_seal'] == cap['source_seal']
                and identity('memra-package-compiled-input-v1', value['tuple']) == a.identity, 'package identity differs')
        expected = owned_json(Path(cap['expectations']) / ('identity-' + a.identity + '.json'))
        require(expected == {'sha256': digest(value)}, 'identity producer custody differs')
        for binding in value['bindings']:
            require(type(binding) is dict and set(binding) == {'path', 'file'}, 'unknown identity input binding')
            path = Path(binding['path'])
            require(path.is_absolute() and path.is_relative_to(Path(cap['output']))
                    and regular(path.parent, path.name) == binding['file'], 'identity compiled input differs')
        print(a.identity)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'verify'))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--features', default='')
    parser.add_argument('--snapshot', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    features = [name for name in args.features.split(',') if name]
    current = cargo_graph(args.manifest, args.target, features)
    if args.command == 'prepare':
        result = current
    else:
        require(args.snapshot is not None, 'verification requires the retained source expectation')
        result = verify(owned_json(args.snapshot), current)
    inputs = [Path(row['manifest_path']).absolute().parent for row in current['bindings'].values()]
    if args.snapshot:
        require(args.out.absolute() != args.snapshot.absolute(), 'output would overwrite source expectation')
    write_owned(args.out, result, inputs)
    print(json.dumps({'schema': SCHEMA, 'packages': len(current['payload']['packages']),
                      'sha256': current['sha256'], 'qualification': False,
                      'package_source_marker_admitted': False}))


if __name__ == '__main__':
    import sys
    try:
        if len(sys.argv)>1 and sys.argv[1] in ('prepare-package', 'receive', 'rederive', 'registry-inputs'): production_main(sys.argv[1:])
        else: main()
    except (Refused, OSError, ValueError, KeyError) as error:
        print('package source refused: ' + str(error), file=sys.stderr);sys.exit(86)
