#!/usr/bin/env python3
"""Explain changed-input validation and select CI execution components.

Cargo dependency closure selects CPU jobs. Native requirements are reported separately;
this tool never turns a CPU result or a missing model into GPU qualification.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import fnmatch
import glob
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import posixpath
import re
import subprocess
import sys
import tempfile
import tomllib

ROOT = Path(__file__).resolve().parents[1]
CORE = {'memra-gguf', 'memra-reference', 'memra-tokenizer', 'memra-validate', 'memra-sampling', 'memra-net-guard'}
PORTABLE = {'memra-tier', 'memra-kv', 'memra-cli'}
NATIVE = {'memra-engine', 'memra-server', 'memra-probe'}
JOBS = ('build', 'clippy', 'server', 'engine', 'portable', 'core', 'lanes', 'arch', 'publish')
# These are executable CPU/harness contracts, not a blanket tools/** exemption.
# Their tests remain in the always-run gates job. Native reruns are named separately.
TOOL_CONTRACTS = {
    'tool-choice': {
        'presence': ['tools/tool-choice-gate.py', 'tools/test_tool_choice_gate.py'],
        'inputs': ['tools/tool-choice-gate.py', 'tools/test_tool_choice_gate.py',
                   'tools/validation_coverage.py'] + [
            f'research/tool-choice-20261003/receipts/{run}/{name}'
            for run in ('native-v3', 'native-v4')
            for name in (
                'candidate.sha256',
                'coverage-plan.json',
                'coverage-results.json',
                'expected-context.json',
                'file-sha256.json',
                'frozen-inputs.sha256',
                'gemma/baseline/environment.json',
                'gemma/baseline/server.log',
                'gemma/candidate/environment.json',
                'gemma/candidate/server.log',
                'gpu-250ms.csv',
                'gpu-after.txt',
                'gpu-before.txt',
                'input-snapshot/test_tool_choice_gate.py',
                'input-snapshot/tool-choice-gate.py',
                'input-snapshot/validation_coverage.py',
                'qwen/baseline/environment.json',
                'qwen/baseline/server.log',
                'qwen/candidate/environment.json',
                'qwen/candidate/server.log',
                'report.json',
            )
        ] + ['research/tool-choice-20261003/receipts/native-v4/source-binding.json'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_tool_choice_gate.py', '16'],
        'native': ['Pinned Qwen/Gemma tool-choice streams: required/named schema and grammar engagement, single-call policy, explicit refusals, and unchanged-auto/tool-none identity with coherent receipt red controls'],
    },
    'q35-cache': {
        'required': True,
        'inputs': ['tools/q35-cold-mixed-gate.py', 'tools/test_q35_cold_mixed_gate.py',
                   'research/sellgate-20260812/sellgate_replay.py', 'tools/cache_qualification.py',
                   'research/sellgate-20260812/workload.lock.json',
                   'research/spill-lead-20260919/integration-day12/integ68-q35ab/main-q35-cold-mixed.log',
                   'research/spill-lead-20260919/integration-day12/integ68-q35ab/integ68-q35-cold-mixed.log'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_q35_cold_mixed_gate.py', '13'],
        'native': ['Qwen3.6 MoE mixed c=4 cache/usage/golden gate on the pinned artifact'],
    },
    'cache-meter': {
        'presence': ['tools/cache-meter-gate.py', 'tools/test_cache_meter_gate.py'],
        'inputs': ['tools/cache-meter-gate.py', 'tools/prometheus_metrics.py',
                   'tools/test_cache_meter_gate.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_cache_meter_gate.py', '3'],
        'native': ['Pinned local model cache-meter closed form, native/OpenAI format parity, authenticated Prometheus and exact usage/token distributions'],
    },
    'metrics-live': {
        'presence': ['tools/metrics-live-gate.py', 'tools/test_prometheus_metrics.py'],
        'inputs': ['tools/metrics-live-gate.py', 'tools/prometheus_metrics.py',
                   'tools/test_prometheus_metrics.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_prometheus_metrics.py', '7'],
        'native': ['Pinned local plain/MTP lifecycle, real queue/cancellation, live capacity and owned-worker-fault assertions; CPU parser success does not qualify serving'],
    },
    'physical-gpu': {
        'required': True,
        'inputs': ['tools/resolve-physical-gpu.py', 'tools/test_resolve_physical_gpu.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_resolve_physical_gpu.py', '21'],
        'native': ['Selected physical GPU refusal/ready test when resolver behavior changes'],
    },
    'support-records': {
        'required': True,
        'inputs': ['tools/check-support-states.py', 'tools/test_check_support_states.py',
                   'docs/support-records.toml', 'tools/support_record_inputs.py',
                   'tools/test_validation_support_record_inputs.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_check_support_states.py', '21'],
        'native': [],
    },
    'serving-qualification': {
        'presence': ['tools/collect-serving-qualification.py', 'tools/test_collect_serving_qualification.py'],
        'inputs': ['tools/collect-serving-qualification.py', 'tools/test_collect_serving_qualification.py',
                   'tools/cache_qualification.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_collect_serving_qualification.py', '8'],
        'native': ['Source-bound composite streaming/cache/offered-concurrency/cancellation/context collector, with the separate cache-disabled red boot'],
    },
    'background-chat-text': {
        'presence': ['tools/background-chat-text-gate.py', 'tools/test_background_chat_text_gate.py'],
        'inputs': ['tools/background-chat-text-gate.py', 'tools/test_background_chat_text_gate.py', 'tools/cache_qualification.py',
                   'research/background-chat-text-20261002/tools/vendor-profile.toml'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_background_chat_text_gate.py', '9'],
        'native': ['Pinned-artifact fresh OFF/ON identity and bare-default probes; paired real chat/text delivery beyond 90 seconds, tenant isolation, native-progress cancellation and exactly-one final callback'],
    },
    'sampled-mtp': {
        'presence': ['tools/collect-sampled-mtp.py', 'tools/test_collect_sampled_mtp.py',
                     'tools/sampled-mtp-requirements.txt'],
        'inputs': ['tools/collect-sampled-mtp.py', 'tools/test_collect_sampled_mtp.py',
                   'tools/sampled-mtp-requirements.txt', 'tools/collect-serving-qualification.py',
                   'tools/cache_qualification.py'],
        'cpu': ['tools/unittest-floor.sh', 'tools', 'test_collect_sampled_mtp.py', '7'],
        'python_requirements': 'tools/sampled-mtp-requirements.txt',
        'native': ['Pinned positive-PMIN eager/graph/residual/bonus/zero-draft probes, distribution controls and vendor-default serving checks'],
    },
}


class Refused(ValueError):
    pass


_RUST_SCANNER = None
_SUPPORT_DATA = None


def support_record_data_inputs(tree, *, directory=False, allow_unknown_reader=False):
    global _SUPPORT_DATA
    if _SUPPORT_DATA is None:
        spec = importlib.util.spec_from_file_location(
            'support_record_data_inputs', Path(__file__).with_name('support_record_inputs.py'))
        _SUPPORT_DATA = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_SUPPORT_DATA)
    try:
        if directory:
            tree = _SUPPORT_DATA.DirectoryTree(tree)
        return _SUPPORT_DATA.resolve(tree)
    except _SUPPORT_DATA.UnmodelledReader as error:
        if allow_unknown_reader:
            return None
        raise Refused(str(error)) from error
    except _SUPPORT_DATA.InputContractError as error:
        raise Refused(str(error)) from error


def rust_code_view(text, string_spans=None):
    global _RUST_SCANNER
    if _RUST_SCANNER is None:
        spec = importlib.util.spec_from_file_location('validation_rust_scanner', Path(__file__).with_name('skip-census.py'))
        _RUST_SCANNER = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_RUST_SCANNER)
    try:
        return _RUST_SCANNER.rust_code_view(text, string_spans)
    except _RUST_SCANNER.CensusError as error:
        raise Refused(str(error)) from error


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args], stderr=subprocess.PIPE)


class Tree:
    def __init__(self, repo, ref):
        self.repo, self.ref = Path(repo), ref
        self.cache = {}

    def read(self, path):
        if path not in self.cache:
            self.cache[path] = git(self.repo, 'show', f'{self.ref}:{path}').decode()
        return self.cache[path]

    def read_bytes(self, path):
        return git(self.repo, 'show', f'{self.ref}:{path}')

    def paths(self, *prefixes):
        return [x.decode() for x in git(self.repo, 'ls-tree', '-r', '--name-only', '-z',
                                       self.ref, '--', *prefixes).split(b'\0') if x]

    def symlinks(self, *prefixes):
        result = {}
        for row in git(self.repo, 'ls-tree', '-r', '-z', self.ref, '--', *prefixes).split(b'\0'):
            if row:
                metadata, path = row.split(b'\t', 1)
                if metadata.startswith(b'120000 '):
                    name = path.decode(); result[name] = self.read(name)
        return result

    def symlinks_exact(self, paths):
        result = {}
        for row in git(self.repo, '--literal-pathspecs', 'ls-tree', '-z',
                       self.ref, '--', *sorted(paths)).split(b'\0'):
            if row:
                metadata, path = row.split(b'\t', 1)
                if metadata.startswith(b'120000 '):
                    name = path.decode(); result[name] = self.read(name)
        return result


class LocalTree(Tree):
    def __init__(self, repo):
        super().__init__(repo, 'WORKTREE')

    def read(self, path):
        target = (self.repo / path).resolve()
        if not target.is_relative_to(self.repo.resolve()):
            raise Refused('source symlink escapes checkout')
        return target.read_text()

    def read_bytes(self, path):
        target = (self.repo / path).resolve()
        if not target.is_relative_to(self.repo.resolve()):
            raise Refused('source symlink escapes checkout')
        return target.read_bytes()

    def paths(self, *prefixes):
        return [x.decode() for x in git(self.repo, 'ls-files', '--cached', '--others',
                                       '--exclude-standard', '-z', '--', *prefixes).split(b'\0') if x]

    def symlinks(self, *prefixes):
        return {p: os.readlink(self.repo / p) for p in self.paths(*prefixes) if (self.repo / p).is_symlink()}

    def symlinks_exact(self, paths):
        return {p: os.readlink(self.repo / p) for p in paths if (self.repo / p).is_symlink()}


def workspace(tree):
    manifest = tomllib.loads(tree.read('Cargo.toml'))
    members = manifest.get('workspace', {}).get('members')
    if not isinstance(members, list) or not members:
        raise Refused('workspace member list unavailable')
    packages, owners, manifests = {}, {}, {}
    workspace_deps = manifest.get('workspace', {}).get('dependencies', {})
    for member in members:
        if not isinstance(member, str) or any(c in member for c in '*?[\n') or '..' in PurePosixPath(member).parts:
            raise Refused('unmodelled workspace member')
        data = tomllib.loads(tree.read(member + '/Cargo.toml'))
        name = data['package']['name']
        if not re.fullmatch(r'memra-[a-z0-9-]+', name) or name in packages:
            raise Refused('unknown or duplicate workspace package')
        packages[name], owners[member], manifests[name] = set(), name, data

    def visit(data, result):
        for key, value in data.items():
            if key in ('dependencies', 'dev-dependencies', 'build-dependencies'):
                for alias, dependency in value.items():
                    if isinstance(dependency, dict) and dependency.get('workspace'):
                        dependency = workspace_deps[alias]
                    actual = dependency.get('package', alias) if isinstance(dependency, dict) else alias
                    if actual in packages:
                        result.add(actual)
                    elif isinstance(dependency, dict) and 'path' in dependency:
                        raise Refused('unmodelled local dependency: ' + alias)
            elif isinstance(value, dict):
                visit(value, result)

    for name, data in manifests.items():
        visit(data, packages[name])
    return packages, owners


def owner(path, owners):
    matches = [name for prefix, name in owners.items() if path.startswith(prefix + '/')]
    if len(matches) > 1:
        raise Refused('ambiguous package ownership')
    return matches[0] if matches else None


def include_argument(argument, package_root, generated_env=()):
    argument = argument.strip().rstrip(',').strip()
    raw = re.fullmatch(r'r(\#*)"(.*)"\1', argument, re.S)
    if raw:
        return raw.group(2)
    if argument.startswith('"') and argument.endswith('"'):
        literal = argument[1:-1]
        # Complex Rust escapes are deliberately not guessed. The unresolved expression
        # expands validation rather than misidentifying a physical input.
        if '\\' in literal:
            raise Refused('escaped Rust include path needs an explicit input contract')
        try:
            return json.loads('"' + literal + '"')
        except json.JSONDecodeError as error:
            raise Refused('unresolved Rust include string') from error
    env = re.fullmatch(r'env!\s*\(\s*"([A-Z0-9_]+)"\s*\)', argument)
    if env:
        if env.group(1) == 'CARGO_MANIFEST_DIR':
            return '\0REPO/' + package_root
        if env.group(1) == 'OUT_DIR' or env.group(1) in generated_env:
            return '\0GENERATED'
        raise Refused('include depends on unregistered environment input: ' + env.group(1))
    concat = re.fullmatch(r'concat!\s*\((.*)\)', argument, re.S)
    if concat:
        body = concat.group(1)
        code = rust_code_view(body)
        depth, start, pieces = 0, 0, []
        for i, char in enumerate(code):
            if char == '(':
                depth += 1
            elif char == ')':
                depth -= 1
            elif char == ',' and depth == 0:
                pieces.append(body[start:i]); start = i + 1
        if body[start:].strip():
            pieces.append(body[start:])
        return ''.join(include_argument(piece, package_root, generated_env) for piece in pieces)
    raise Refused('unresolved include expression')


def included_inputs(tree, owners):
    """Literal includes, including old-tree consumers of deleted/renamed fixtures.

The census does not exempt arbitrary research data. Unknown non-document inputs still
expand to all jobs. Build-generated flag data is explicitly registered below.
"""
    inputs = defaultdict(set)
    source_consumers = defaultdict(set)
    links = tree.symlinks()
    runtime_paths = None
    def linked_input(path):
        fixed_prefix = re.split(r'[\[{}*?]', path, maxsplit=1)[0].rstrip('/')
        return any(path == alias or path.startswith(alias + '/')
                   or alias.startswith(path.rstrip('/') + '/')
                   or (fixed_prefix != path and
                       (alias.startswith(fixed_prefix) or fixed_prefix.startswith(alias + '/')))
                   or fnmatch.fnmatchcase(alias, path)
                   or fnmatch.fnmatchcase(alias, path.rstrip('/') + '/**') for alias in links)

    def runtime_target(path):
        nonlocal runtime_paths
        parts = []
        for part in path.split('/'):
            if part in ('', '.'):
                continue
            if part == '..':
                if not parts:
                    raise Refused('runtime fixture path escapes repository: ' + path)
                parent = '/'.join(parts)
                if any(re.search(r'[{}*?\[]', component) for component in parts):
                    raise Refused('runtime fixture parent traversal depends on a pattern: ' + path)
                # The filesystem follows symlinks before applying '..'. Check the
                # traversed spelling, including prefixes normalized so far, before
                # removing anything. Lexical cancellation alone is not proof.
                if linked_input(parent):
                    raise Refused('runtime fixture parent traversal contains a symlink: ' + path)
                if runtime_paths is None:
                    runtime_paths = tree.paths()
                if not any(candidate.startswith(parent + '/') for candidate in runtime_paths):
                    raise Refused('runtime fixture parent directory is unresolved: ' + parent)
                parts.pop()
            else:
                parts.append(part)
        return '/'.join(parts)

    for path in links:
        package = owner(path, owners)
        if package is None:
            continue
        # Source module resolution through symlinks depends on both lexical and
        # physical locations. Until that transitive graph is modelled, expand.
        raise Refused('crate symlink needs a transitive input contract: ' + path)
    contracts = json.loads(Path(__file__).with_name('validation_inputs.json').read_text())
    if contracts.get('schema') != 'memra-validation-build-inputs-v1':
        raise Refused('unrecognized build input contracts')
    generated = defaultdict(set)
    for prefix, package in owners.items():
        manifest = tomllib.loads(tree.read(prefix + '/Cargo.toml'))
        build = manifest.get('package', {}).get('build', 'build.rs')
        if build is False:
            continue
        if not isinstance(build, str):
            raise Refused('unmodelled build script declaration')
        path = posixpath.normpath(prefix + '/' + build)
        if path not in tree.paths(prefix):
            if build != 'build.rs':
                raise Refused('declared build script missing')
            continue
        contract = contracts['build_scripts'].get(path)
        if not contract or contract.get('package') != package or hashlib.sha256(tree.read(path).encode()).hexdigest() != contract['sha256']:
            raise Refused('build-script input contract changed or missing: ' + path)
        generated[package].update(contract['generated_env'])
        for external in contract['external_inputs']:
            inputs[external].add(package)
    options = ['--untracked'] if isinstance(tree, LocalTree) else []
    refs = [] if isinstance(tree, LocalTree) else [tree.ref]
    result = subprocess.run(['git', '-C', str(tree.repo), 'grep', '-l', '-z', *options, '-E',
                             r'include|path|research|docs', *refs, '--', 'crates'],
                            capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise Refused('include census failed')
    pattern = re.compile(r'\b(include(?:_str|_bytes)?)\s*!\s*\(')
    for raw in result.stdout.split(b'\0'):
        if not raw:
            continue
        path = raw.decode() if isinstance(tree, LocalTree) else raw.decode().split(':', 1)[1]
        package = owner(path, owners)
        if package is None:
            raise Refused('include has no package owner')
        source = tree.read(path)
        string_spans = []
        code = rust_code_view(source, string_spans)
        prefix = next(k for k, v in owners.items() if v == package)
        # A conditional path can select a module whose own includes are outside
        # the scanned crate. Do not guess cfg truth or treat it as a data reader.
        for match in re.finditer(r'#\s*\[\s*(?:r#)?cfg_attr\b', code):
            end, depth = match.end(), 1
            while end < len(code) and depth:
                if code[end] == '[':
                    depth += 1
                elif code[end] == ']':
                    depth -= 1
                end += 1
            if depth:
                raise Refused('unterminated conditional Rust attribute')
            if re.search(r'\bpath\s*=', code[match.end():end - 1]):
                raise Refused('conditional Rust module path needs a transitive input contract')
        literals = []
        for match in pattern.finditer(code):
            start, end, depth = match.end(), match.end(), 1
            while end < len(code) and depth:
                if code[end] == '(':
                    depth += 1
                elif code[end] == ')':
                    depth -= 1
                end += 1
            if depth:
                raise Refused('unterminated include expression')
            try:
                literal = include_argument(source[start:end - 1], prefix, generated[package])
            except Refused as error:
                contract = contracts.get('dynamic_includes', {}).get(path)
                if contract and hashlib.sha256(source.encode()).hexdigest() == contract['sha256']:
                    for input_pattern in contract['inputs']:
                        inputs[input_pattern].add(package)
                    continue
                raise Refused(f'{path}:{source.count(chr(10), 0, start) + 1}: {error}') from error
            if literal.startswith('\0GENERATED'):
                continue  # the build-script contract below owns these inputs
            literals.append((literal, match.group(1) == 'include'))
        for match in re.finditer(r'#\s*\[\s*(?:r#)?path\s*=', code):
            spans = [(start, end) for start, end in string_spans
                     if start >= match.end() and not code[match.end():start].strip()]
            if not spans:
                raise Refused('unresolved Rust module path attribute')
            start, end = spans[0]
            literals.append((include_argument(source[start:end], prefix, generated[package]), True))
        for literal, compiled_source in literals:
            if literal.startswith('\0REPO/'):
                spelling = literal[len('\0REPO/'):]
            elif literal.startswith('/'):
                raise Refused('absolute include is outside the declared checkout')
            else:
                spelling = posixpath.join(posixpath.dirname(path), literal)
            if '\0' in spelling:
                raise Refused('include escapes repository')
            # Check the complete resolved literal before cancelling parents. Rust
            # follows aliases first, including ones hidden across concat! pieces.
            target = runtime_target(spelling)
            if linked_input(target):
                raise Refused('included symlink needs a transitive input contract: ' + target)
            if compiled_source:
                target_owner = owner(target, owners)
                if target_owner is None:
                    raise Refused('external Rust source needs a transitive input contract: ' + target)
                if target_owner != package:
                    # The complete source-owning crate is scanned below. Treat all its
                    # source and external inputs as dependencies of the textual reader,
                    # even without a Cargo dependency declaration.
                    target_prefix = next(k for k, v in owners.items() if v == target_owner)
                    inputs[target_prefix + '/**'].add(package)
                    source_consumers[target_owner].add(package)
            inputs[target].add(package)
        # Runtime test/fixture readers also matter even when they do not use include!.
        # Rooted literals and formatted prefixes conservatively reach their package.
        # Bare directory names cover the whole subtree; comments and generated-source
        # strings cannot create fake references because the lexer supplies literal spans.
        for start, end in string_spans:
            token = source[start:end]
            if token.startswith(('br', 'cr', 'b"', 'c"')):
                continue
            try:
                value = include_argument(token, prefix, generated[package])
            except Refused:
                continue
            if value in ('research', 'docs'):
                if linked_input(value):
                    raise Refused('runtime fixture subtree contains a symlink: ' + value)
                inputs[value + '/**'].add(package)
            elif '://' not in value and re.search(r'(?:^|/)(?:research|docs)/', value):
                # A format parameter cannot establish a narrower dependency than its
                # fixed prefix. This can select extra work, never omit a consumer.
                match = re.search(r'(?:research|docs)/', value)
                rooted = value[match.start():]
                if linked_input(rooted):
                    raise Refused('runtime fixture symlink needs an input contract: ' + rooted)
                rooted = runtime_target(rooted)
                if linked_input(rooted):
                    raise Refused('normalized runtime fixture symlink needs an input contract: ' + rooted)
                if not rooted:
                    inputs['**'].add(package)
                    continue
                # A string can be a literal path or a pattern consumed by a reader.
                # Retain both interpretations; escaping must never drop glob reach.
                for input_pattern in {re.sub(r'\{[^}]*\}', '*', rooted),
                                      re.sub(r'\{[^}]*\}', '*', glob.escape(rooted))}:
                    inputs[input_pattern].add(package)
                    # A literal may be a directory later extended with join()/read_dir().
                    inputs[input_pattern.rstrip('/') + '/**'].add(package)
    # memra-engine/build.rs emits the boot-audit registry from this source.
    if 'memra-engine' in owners.values():
        inputs['docs/FLAGS.md'].add('memra-engine')
    while True:
        additions = 0
        for packages in inputs.values():
            readers = set().union(*(source_consumers[p] for p in packages))
            additions += len(readers - packages)
            packages.update(readers)
        if not additions:
            break
    return inputs


def matches_input(path, pattern):
    # Literal filenames may contain glob metacharacters. Also keep literal
    # directory descendants while supporting explicitly conservative glob rules.
    return (path == pattern or fnmatch.fnmatchcase(path, pattern)
            or (pattern.endswith('/**') and path.startswith(pattern[:-3] + '/')))


def input_consumers(path, inputs):
    return set().union(*(packages for pattern, packages in inputs.items()
                         if matches_input(path, pattern))) if inputs else set()


def closure(direct, graph):
    affected = set(direct)
    reasons = {}
    while True:
        new = {name for name, dependencies in graph.items() if dependencies & affected} - affected
        if not new:
            return affected, reasons
        for name in new:
            reasons[name] = sorted(graph[name] & affected)
        affected.update(new)


def edge_decisions(graph, affected, direct):
    """Binary decisions over declared input/dependent edges, with witnesses.

These are CPU dependency edges, not a claim that every dynamic model execution path
has been discovered. Native obligations remain a separate explicit plan section.
"""
    decisions = []
    for package in sorted(graph):
        decisions.append({'from': 'changed-inputs', 'to': package, 'affected': package in direct,
                          'reason': 'Changed declared input' if package in direct else
                          'No changed declared input directly belongs to this package'})
        for dependency in sorted(graph[package]):
            decisions.append({'from': dependency, 'to': package, 'affected': dependency in affected,
                              'reason': 'Affected dependency reaches this consumer' if dependency in affected else
                              'Dependency has unchanged declared inputs and no affected predecessor'})
    return decisions


def full(reason, changed=()):
    return {'schema': 'memra-validation-plan-v1', 'mode': 'full', 'reason': reason,
            'changed': list(changed), 'packages': [], 'direct_packages': [], 'dependency_reasons': {},
            'jobs': dict.fromkeys(JOBS, True), 'requires_cuda': True,
            'cpu_contracts': [{'id': name, **c} for name, c in TOOL_CONTRACTS.items() if c.get('required')],
            'native': {'scope': 'full', 'requirements': [reason],
            'qualification': False}, 'omitted': {}}


def native_scope(paths, native_requirements, packages):
    if not packages:
        return {'scope': 'harness' if native_requirements else 'none',
                'requirements': sorted(native_requirements), 'qualification': False}
    if any(p.endswith(('.cu', '.cuh', '.h', '.cpp', '.c')) or '/build.rs' in p or
           p in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml') for p in paths):
        scope, requirements = 'full', ['Shared native/build input: full applicable target battery and affected model gates']
    elif any(p.startswith(('crates/memra-engine/', 'crates/memra-kv/', 'crates/memra-runtime/',
                           'crates/memra-gguf/', 'crates/memra-reference/')) for p in paths):
        scope, requirements = 'native-impact', ['Use operation/model reach plus kernel, checkpoint, cache, spec and topology gates for the changed program; unknown reach expands to full']
    elif any(p.startswith('crates/memra-sampling/') for p in paths):
        scope, requirements = 'sampling', ['CPU sampler oracle and native target-distribution checks on affected plain/graph/spec routes, including vendor-default serving and cache reuse']
    elif any(p.startswith('crates/memra-server/') for p in paths):
        scope, requirements = 'serving', ['Affected API/admission/usage/stream/cancel/cache cells on a pinned real model; scheduler or model-route changes include every reachable route']
    elif any(p.startswith('crates/memra-tokenizer/') for p in paths):
        scope, requirements = 'tokenizer', ['Artifact tokenizer/template parity and served request goldens for affected families']
    elif any(p.startswith('crates/memra-cli/') for p in paths):
        scope, requirements = 'qualification-tooling', ['CPU oracle/record/refusal contracts; exercise any changed native collector on its exact model and target without promoting support from mocks']
    else:
        scope, requirements = 'native-impact', ['Determine downstream behavioral reach before native qualification; do not infer independence from a package name']
    return {'scope': scope, 'requirements': requirements + sorted(native_requirements), 'qualification': False}


def native_probe_inputs(tree):
    inputs = defaultdict(set)
    paths = set(tree.paths('tools/fast-gate'))
    for registry in ('tools/fast-gate/models.tsv', 'tools/fast-gate/accept-cells.tsv'):
        if registry not in paths:
            continue
        for line in tree.read(registry).splitlines():
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            probe = line.split('\t', 1)[0]
            for path in re.findall(r'(?:research|probe)/[^\s\x22\x27;]+', line):
                inputs[path].add(registry + ':' + probe)
    return inputs


def make_plan(paths, base_tree, head_tree):
    paths = sorted(set(paths))
    if not paths:
        return full('empty or unavailable change set')
    for path in paths:
        if PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or any(c in path for c in '\n\r\t\0'):
            return full('noncanonical changed path')
    try:
        base_graph, base_owners = workspace(base_tree)
        graph, owners = workspace(head_tree)
        if graph != base_graph or owners != base_owners:
            return full('workspace dependency or membership changed', paths)
        if any(p in ('Cargo.toml', 'Cargo.lock', 'rust-toolchain.toml') or
               p.startswith('.cargo/') or (p.startswith('.github/') and not p.startswith('.github/ISSUE_TEMPLATE/')) or p.endswith(('/Cargo.toml', '/build.rs'))
               for p in paths):
            return full('compiler/build/workflow/dependency input changed', paths)
        includes = included_inputs(head_tree, owners)
        for path, packages in included_inputs(base_tree, base_owners).items():
            includes[path].update(packages)
        direct, contracts, native_requirements = set(), set(), set()
        support_data = set()
        for tree in (base_tree, head_tree):
            resolved = support_record_data_inputs(tree)
            support_data.update(resolved['required'])
            support_data.update(resolved['optional'])
        contract_paths = set(base_tree.paths('tools')) | set(head_tree.paths('tools'))
        probe_inputs = native_probe_inputs(head_tree)
        for pattern, probes in native_probe_inputs(base_tree).items():
            probe_inputs[pattern].update(probes)
        for path in paths:
            if path in support_data:
                contracts.add('support-records')
                native_requirements.update(TOOL_CONTRACTS['support-records']['native'])
            for pattern, probes in probe_inputs.items():
                if matches_input(path, pattern):
                    native_requirements.add('Changed native probe input ' + path + ': rerun pinned assertions for ' + ', '.join(sorted(probes)))
            package = owner(path, owners)
            if package:
                direct.add(package)
                direct.update(input_consumers(path, includes))
                continue
            consumers = input_consumers(path, includes)
            if consumers:
                direct.update(consumers)
                continue
            matches = [name for name, c in TOOL_CONTRACTS.items() if path in c['inputs']
                       and ('presence' not in c or any(p in contract_paths for p in c['presence']))]
            if matches:
                contracts.update(matches)
                for name in matches:
                    native_requirements.update(TOOL_CONTRACTS[name]['native'])
                continue
            if path in support_data:
                continue
            # Receipt data is not a compiler input unless a declared include, generated
            # input, or runtime fixture reader reaches it. Standalone research programs
            # retain their own experiment requirements, not an unrelated engine build.
            if path.startswith('research/'):
                if Path(path).suffix in ('.rs', '.cu', '.cuh', '.cpp', '.c', '.py', '.sh'):
                    native_requirements.add('Execute the changed standalone research probe and its controls: ' + path)
                continue
            if path.endswith('.md') and not path.startswith('crates/'):
                continue
            if path == 'LICENSE' or path.startswith('.github/ISSUE_TEMPLATE/'):
                continue
            return full('unmodelled input: ' + path, paths)
        affected, dependency_reasons = closure(direct, graph)
        jobs = {
            'build': bool(affected), 'clippy': bool(affected),
            'server': 'memra-server' in affected, 'engine': 'memra-engine' in affected,
            'portable': bool(affected & PORTABLE), 'core': bool(affected & CORE),
            'lanes': 'memra-lanes' in affected, 'arch': bool(affected & NATIVE),
            'publish': bool(affected - {'memra-probe'}),
        }
        return {'schema': 'memra-validation-plan-v1', 'mode': 'scoped',
                'reason': 'dependency closure plus declared input contracts', 'changed': paths,
                'direct_packages': sorted(direct), 'packages': sorted(affected),
                'dependency_reasons': dependency_reasons, 'jobs': jobs,
                'edge_decisions': edge_decisions(graph, affected, direct),
                'requires_cuda': bool(affected & NATIVE),
                'cpu_contracts': [{'id': name, **TOOL_CONTRACTS[name]} for name in sorted(contracts)],
                'native': native_scope(paths, native_requirements, affected),
                'omitted': {job: 'No changed package or reverse dependency reaches this execution component'
                            for job, selected in jobs.items() if not selected}}
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        return full('dependency/input analysis unavailable: ' + str(error), paths)


def event_plan(repo, event, pr_base, before, head):
    try:
        if event not in ('pull_request', 'push'):
            return full('unrecognized event')
        base = pr_base if event == 'pull_request' else before
        if not base or not head or base == '0' * 40:
            return full('missing event base/head')
        # Freeze moving branch names once. Every later diff/read must see the same tree.
        base = git(repo, 'rev-parse', '--verify', base + '^{commit}').decode().strip()
        head = git(repo, 'rev-parse', '--verify', head + '^{commit}').decode().strip()
        if event == 'pull_request':
            base = git(repo, 'merge-base', base, head).decode().strip()
        paths = [p.decode() for p in git(repo, 'diff', '--no-ext-diff', '--no-renames',
                                        '--name-only', '-z', base, head, '--').split(b'\0') if p]
        plan = make_plan(paths, Tree(repo, base), Tree(repo, head))
        preserve_contract_obligations(plan, Tree(repo, base), Tree(repo, head))
        plan.update(base=base, head=head)
        return plan
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        return full('Git change set unavailable: ' + str(error))


def local_plan(repo, base):
    try:
        git(repo, 'rev-parse', '--verify', base + '^{commit}')
        paths = set()
        for args in [('diff', '--no-ext-diff', '--no-renames', '--name-only', '-z', base, '--'),
                     ('ls-files', '--others', '--exclude-standard', '-z')]:
            paths.update(x.decode() for x in git(repo, *args).split(b'\0') if x)
        masked = [x for x in git(repo, 'ls-files', '-v', '-z').split(b'\0')
                  if x and (x[:1].islower() or x[:1] == b'S')]
        if masked:
            return full('index-masked inputs prevent a scoped decision', sorted(paths))
        ignored = git(repo, 'ls-files', '--others', '--ignored', '--exclude-standard', '-z',
                      '--', '.cargo', 'crates', 'tools').split(b'\0')
        if any(x and not (b'/__pycache__/' in x and x.endswith(b'.pyc')) for x in ignored):
            return full('ignored build/test inputs prevent a scoped decision', sorted(paths))
        plan = make_plan(paths, Tree(repo, base), LocalTree(repo))
        preserve_contract_obligations(plan, Tree(repo, base), LocalTree(repo))
        plan.update(base=base, head='WORKTREE', qualification=False)
        return plan
    except (OSError, UnicodeError, subprocess.SubprocessError) as error:
        return full('local change set unavailable: ' + str(error))


def preserve_contract_obligations(plan, before, after):
    if plan['mode'] != 'full':
        return
    existing = {c['id'] for c in plan['cpu_contracts']}
    paths = set(before.paths('tools', 'research', 'docs')) | set(after.paths('tools', 'research', 'docs'))
    for name, contract in TOOL_CONTRACTS.items():
        if name not in existing and any(p in paths for p in contract.get('presence', contract['inputs'])):
            plan['cpu_contracts'].append({'id': name, **contract})


def emit(plan):
    print('code=' + str(any(plan['jobs'].values())).lower())
    print('reason=' + plan['reason'].replace('\n', ' ')[:500])
    for name in JOBS:
        print(name + '=' + str(plan['jobs'][name]).lower())
    print('packages=' + ','.join(plan['packages']))
    print('requires_cuda=' + str(plan['requires_cuda']).lower())
    print('mode=' + plan['mode'])
    print('contracts=' + (','.join(item['id'] for item in plan['cpu_contracts']) or 'none'))


def cargo_packages(value, root):
    # Empty/malformed/missing output expands to workspace, never a command injection or skip.
    if not value:
        return ['--workspace']
    names = value.split(',')
    known = {tomllib.loads(p.read_text())['package']['name'] for p in (root / 'crates').glob('*/Cargo.toml')}
    if len(names) != len(set(names)) or any(not re.fullmatch(r'memra-[a-z0-9-]+', n) or n not in known for n in names):
        return ['--workspace']
    return [arg for n in sorted(names) for arg in ('-p', n)]


def publish_packages(value, root):
    selected = cargo_packages(value, root)
    if selected == ['--workspace']:
        return ['--workspace', '--exclude', 'memra-probe']
    # Package verification must include workspace dependencies, not silently resolve an
    # older published dependency with the same version as current source.
    graph, _ = workspace(LocalTree(root))
    names = set(value.split(','))
    while True:
        dependencies = set().union(*(graph[n] for n in names)) - names
        if not dependencies:
            break
        names.update(dependencies)
    return [arg for n in sorted(names - {'memra-probe'}) for arg in ('-p', n)]


def cpu_contract_names(root, selected):
    # A successful classifier explicitly distinguishes no affected contracts from
    # missing/failed selection, which must still run all available contracts.
    names = selected.split(',') if selected and selected != 'none' else []
    if names and (len(set(names)) != len(names) or any(n not in TOOL_CONTRACTS for n in names)):
        raise Refused('unknown or duplicated CPU contract')
    data = support_record_data_inputs(root, directory=True, allow_unknown_reader=True)
    if selected == 'none' and data is not None:
        return []
    available = [n for n, c in TOOL_CONTRACTS.items()
                 if c.get('required') or any((root / p).exists() for p in c.get('presence', c['inputs']))]
    if data is None:
        # Planning expands unknown readers. Execution also expands stale subsets,
        # then runs the real census without trusting the unsupported data graph.
        names = available
    else:
        names = names or available
    for name in names:
        if not all((root / p).is_file() for p in TOOL_CONTRACTS[name]['inputs']):
            raise Refused('selected contract input is missing: ' + name)
        if name == 'support-records' and data is not None:
            for path in data['required']:
                if not (root / path).is_file():
                    raise Refused('selected contract input is missing: support-records: ' + path)
    return names


def run_cpu_contract(contract, root):
    requirements = contract.get('python_requirements')
    if not requirements:
        subprocess.run(contract['cpu'], cwd=root, check=True)
        return
    # Keep third-party test dependencies out of the caller's Python environment.
    # Cleanup also happens when installation or the actual test command fails.
    with tempfile.TemporaryDirectory(prefix='memra-validation-python-') as directory:
        subprocess.run([sys.executable, '-m', 'venv', directory], check=True)
        python = str(Path(directory) / 'bin/python')
        subprocess.run([python, '-m', 'pip', 'install', '--disable-pip-version-check',
                        '-r', str(root / requirements)], check=True)
        env = dict(os.environ, PATH=str(Path(directory) / 'bin') + os.pathsep + os.environ.get('PATH', ''),
                   OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        subprocess.run(contract['cpu'], cwd=root, env=env, check=True)


def scoped_bin_targets(value, root):
    packages = cargo_packages(value, root)
    if packages == ['--workspace']:
        return ['--workspace', '--bins']
    try:
        metadata = json.loads(subprocess.check_output(
            ['cargo', 'metadata', '--no-deps', '--format-version', '1', '--offline', '--locked'],
            cwd=root, stderr=subprocess.PIPE))
        selected = set(value.split(','))
        found = {p['name'] for p in metadata['packages'] if p['name'] in selected}
        if found != selected:
            raise Refused('selected package missing from Cargo metadata')
        bins = sorted({t['name'] for p in metadata['packages'] if p['name'] in selected
                       for t in p['targets'] if t['kind'] == ['bin']})
        if not bins:
            return ['--workspace', '--bins']
        # Keep workspace feature unification while filtering the actual binary targets.
        return ['--workspace', *[arg for name in bins for arg in ('--bin', name)]]
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return ['--workspace', '--bins']


def feature_compatible(packages, root, *, development):
    """Refuse package narrowing if it changes or ambiguously splits feature programs."""
    if packages == ['--workspace']:
        return False, 'whole workspace selected'
    def features(selection):
        output = subprocess.check_output(
            ['cargo', 'tree', '--locked', '--offline', '--prefix', 'none', '--format', '{p}|{f}',
             '--no-dedupe', '-e', 'normal,build,dev' if development else 'normal,build', *selection],
            cwd=root, stderr=subprocess.PIPE).decode()
        rows = defaultdict(set)
        for line in output.splitlines():
            if '|' in line:
                package, enabled = line.rsplit('|', 1)
                rows[package].add(tuple(sorted(x for x in enabled.split(',') if x)))
        if not rows:
            raise Refused('empty Cargo feature graph')
        return rows
    try:
        baseline, selected = features(['--workspace']), features(packages)
        differences = [p for p, values in selected.items()
                       if len(values) != 1 or baseline.get(p) != values]
        if differences:
            return False, 'feature programs differ: ' + ', '.join(sorted(differences))
        return True, 'resolved dependency feature programs match workspace'
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        return False, 'feature graph unavailable: ' + str(error)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    ci = commands.add_parser('ci')
    ci.add_argument('event', nargs='?', default='')
    ci.add_argument('pr_base', nargs='?', default='')
    ci.add_argument('before', nargs='?', default='')
    ci.add_argument('head', nargs='?', default='')
    ci.add_argument('repo', nargs='?', type=Path, default=ROOT)
    ci.add_argument('--json', action='store_true')
    local = commands.add_parser('local')
    local.add_argument('--repo', type=Path, default=ROOT)
    local.add_argument('--base', default='origin/main')
    census = commands.add_parser('census')
    census.add_argument('ref'); census.add_argument('repo', nargs='?', type=Path, default=ROOT)
    cargo = commands.add_parser('cargo')
    cargo.add_argument('action', choices=['build', 'clippy', 'publish'])
    cargo.add_argument('--packages', default=os.environ.get('CI_VALIDATION_PACKAGES', ''))
    cargo.add_argument('--dry-command', action='store_true')
    contracts = commands.add_parser('contracts')
    contracts.add_argument('--selected', default=os.environ.get('CI_VALIDATION_CONTRACTS', ''))
    commands.add_parser('check-registry')
    args = parser.parse_args()
    if args.command == 'check-registry':
        graph, _ = workspace(LocalTree(ROOT))
        known = CORE | PORTABLE | NATIVE | {'memra-lanes', 'memra-runtime'}
        missing = set(graph) - known
        if missing:
            raise Refused('workspace packages need an explicit CI execution component: ' + ', '.join(sorted(missing)))
        print('Validation registry covers all ' + str(len(graph)) + ' workspace packages')
    elif args.command == 'contracts':
        names = cpu_contract_names(ROOT, args.selected)
        for name in names:
            contract = TOOL_CONTRACTS[name]
            print('CPU contract: ' + name, flush=True)
            run_cpu_contract(contract, ROOT)
    elif args.command == 'local':
        print(json.dumps(local_plan(args.repo, args.base), indent=2))
    elif args.command == 'ci':
        plan = event_plan(args.repo, args.event, args.pr_base, args.before, args.head)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            emit(plan)
    elif args.command == 'census':
        try:
            tree = Tree(args.repo, args.ref)
            owners = {p.rsplit('/', 1)[0]: p.split('/')[1] for p in tree.paths('crates') if p.endswith('/Cargo.toml')}
            print('\n'.join(sorted(p for p in included_inputs(tree, owners) if p.startswith('research/'))))
        except (OSError, ValueError, KeyError, subprocess.SubprocessError):
            print('?')
    else:
        packages = cargo_packages(args.packages, ROOT)
        if args.action == 'build':
            cmd = ['cargo', 'build', '--release', *scoped_bin_targets(args.packages, ROOT)]
        elif args.action == 'clippy':
            compatible, reason = feature_compatible(packages, ROOT, development=True)
            if not compatible:
                packages = ['--workspace']
            print('Clippy scope: ' + reason, file=sys.stderr)
            cmd = ['cargo', 'clippy', '--release', '--all-targets', *packages, '--', '-D', 'warnings']
        else:
            packages = publish_packages(args.packages, ROOT)
            if not packages:
                raise Refused('no publishable selected package')
            compatible, reason = feature_compatible(packages, ROOT, development=False)
            if not compatible:
                packages = ['--workspace', '--exclude', 'memra-probe']
            print('Package scope: ' + reason, file=sys.stderr)
            cmd = ['cargo', 'publish', *packages, '--locked', '--dry-run']
        print(json.dumps(cmd), flush=True)
        if not args.dry_command:
            return subprocess.call(cmd, cwd=ROOT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
