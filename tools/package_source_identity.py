#!/usr/bin/env python3
"""Source-closure prototype. Run outside Cargo; never grants a build identity marker."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess
import re

SCHEMA = 'memra-package-input-prototype-v1'
RESERVED = '.memra-package-source.json'
MAX_JSON = 64 * 1024 * 1024
SHA256 = re.compile('[0-9a-f]{64}')


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


def inventory(root):
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
                if path == root / '.git' and (path / 'HEAD').is_file() and (path / 'objects').is_dir():
                    dirs.remove(name)
            for name in files:
                relative = (Path(parent) / name).relative_to(root).as_posix()
                if relative in (RESERVED, '.cargo_vcs_info.json', '.cargo-ok'):
                    continue
                paths.append(relative)
                info = os.lstat(Path(parent) / name)
                leaves[relative] = (info.st_dev, info.st_ino, info.st_size, info.st_mode,
                                    info.st_mtime_ns, info.st_ctime_ns)
        return sorted(paths), directories, identity, leaves
    paths, dirs, identity, leaves = membership()
    require('Cargo.toml' in paths and paths, 'package source inventory is empty')
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
            require(type(path) is str and type(entry) is dict and set(entry) == {'sha256', 'bytes', 'mode'}
                    and type(entry['bytes']) is int and entry['bytes'] >= 0
                    and type(entry['sha256']) is str and SHA256.fullmatch(entry['sha256'])
                    and entry['mode'] in ('100644', '100755'), 'invalid source file identity')
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
    main()
