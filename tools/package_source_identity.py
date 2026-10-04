#!/usr/bin/env python3
"""Source-closure prototype. Run outside Cargo; never grants a build identity marker."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import subprocess

SCHEMA = 'memra-package-input-prototype-v1'
RESERVED = '.memra-package-source.json'


class Refused(ValueError):
    pass


def require(value, reason):
    if not value:
        raise Refused(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(root, name):
    path = PurePosixPath(name)
    require(name and not path.is_absolute() and '..' not in path.parts
            and path.as_posix() == name and not any(c in name for c in '\0\n\r\\'),
            'noncanonical source path')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    descriptor = None
    try:
        for part in path.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), 'source is not a regular file: ' + name)
        hasher = hashlib.sha256()
        length = 0
        while True:
            data = os.read(descriptor, 1024 * 1024)
            if not data:
                break
            hasher.update(data); length += len(data)
        after = os.fstat(descriptor)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns),
                'source changed while read: ' + name)
        return {'sha256': hasher.hexdigest(), 'bytes': length,
                'mode': '100755' if before.st_mode & stat.S_IXUSR else '100644'}
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory)


def inventory(root):
    """All physical package resources, except named output/history/derived metadata."""
    root = Path(root)
    require(not root.is_symlink() and root.is_dir(), 'source root is not a contained directory')
    paths = []
    for parent, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(parent) / name
            require(not path.is_symlink(), 'source directory link is not admitted')
            if name in ('target', '.git'):
                dirs.remove(name)
        for name in files:
            relative = (Path(parent) / name).relative_to(root).as_posix()
            if relative in (RESERVED, '.cargo_vcs_info.json', '.cargo-ok'):
                continue
            paths.append(relative)
    require('Cargo.toml' in paths and paths, 'package source inventory is empty')
    return {name: regular(root, name) for name in sorted(paths)}


def package_key(package):
    source = package.get('source') or 'path-source'
    # Git revision remains provenance. Source bytes, not rewritten history,
    # distinguish the package-source prototype's logical content keys.
    if source.startswith('git+'):
        source = source.split('#', 1)[0]
    return package['name'] + '@' + package['version'] + ':' + source


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
    graph = json.loads(result.stdout)
    require(graph.get('resolve') is not None, 'Cargo graph has no resolved dependency bindings')
    packages = {row['id']: row for row in graph['packages']}
    entry = [row for row in packages.values() if Path(row['manifest_path']).absolute() == manifest]
    require(len(entry) == 1, 'canonical entry does not have exactly one Cargo binding')
    nodes = {row['id']: row for row in graph['resolve']['nodes']}
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
    require(set(snapshot) == set(current) and snapshot['payload']['schema'] == SCHEMA,
            'unknown source prototype schema')
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
        row = json.loads(line)
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
        result = verify(json.loads(args.snapshot.read_text()), current)
    args.out.write_bytes(canonical(result) + b'\n')
    print(json.dumps({'schema': SCHEMA, 'packages': len(current['payload']['packages']),
                      'sha256': current['sha256'], 'qualification': False,
                      'package_source_marker_admitted': False}))


if __name__ == '__main__':
    main()
