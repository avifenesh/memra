"""Pinned ordinary build/input provenance for the native loader regression gate."""
from pathlib import Path, PurePosixPath
import hashlib
import json
import os
import re
import stat
import sys
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PREFIX = 'research/modelplan-onboarding-bound-census-20260920/native-acceptance-20260922/'
SHARED = 'research/modelplan-onboarding-rewrite-identity-20260920/'
HELPERS = ('tools/check_hardware_gate.py', 'tools/release_qualification.py',
           'tools/release_inputs.py', 'tools/release_input_view.py',
           SHARED+'native_build_record.py', SHARED+'qualify-native.py',
           SHARED+'native_env_controller.py', SHARED+'native_finalization.py', PREFIX+'native_binding.py')
SHA = re.compile(r'[0-9a-f]{64}')
COMMIT = re.compile(r'[0-9a-f]{40}')
CASE_ID = re.compile(r'(gguf|hf)-[a-z][a-z0-9_]{0,40}')


def require(condition, reason):
    if not condition:
        raise RuntimeError(reason)


def unique(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON field: '+key)
        result[key] = value
    return result


def parse(data):
    return json.loads(data, object_pairs_hook=unique)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path, executable=False):
    path = Path(path)
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), 'nonregular or symlinked input: '+str(path))
    require(not executable or os.access(path, os.X_OK), 'input is not executable')
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, 'rb') as stream:
        before = os.fstat(stream.fileno())
        require((before.st_dev,before.st_ino)==(info.st_dev,info.st_ino), 'input replaced before hashing')
        h = hashlib.sha256()
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
        after = os.fstat(stream.fileno())
        require((before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'input changed while hashing')
    return {'bytes': before.st_size, 'sha256': h.hexdigest()}


def relative(name):
    require(isinstance(name, str) and name and '\\' not in name and '\0' not in name,
            'invalid relative path')
    p = PurePosixPath(name)
    require(not p.is_absolute() and p.as_posix() == name and
            all(part not in ('.', '..') for part in p.parts), 'noncanonical or escaping path')
    return p


def contained(root, name):
    p = relative(name)
    path = root
    for part in p.parts:
        path = path / part
        require(not path.is_symlink(), 'symlink in contained path: '+name)
    resolved = path.resolve(strict=True)
    require(resolved.is_relative_to(root), 'input escaped root: '+name)
    return resolved


def load_module(name, path, data):
    module = types.ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    exec(compile(data, str(path), 'exec'), module.__dict__)
    return module


class Binding:
    def __init__(self, selection, selected_sha, evidence, binary, fixtures):
        self.selection_path = selection.resolve(strict=True)
        identity(self.selection_path)
        raw = self.selection_path.read_bytes()
        require(SHA.fullmatch(selected_sha) is not None and hashlib.sha256(raw).hexdigest() == selected_sha,
                'externally selected binding digest mismatch')
        self.sha = selected_sha
        self.value = parse(raw)
        v = self.value
        require(v['schema'] == 'memra-bound-loader-selection-v1', 'wrong selection schema')
        require(COMMIT.fullmatch(v['source_commit']) is not None and COMMIT.fullmatch(v['source_tree']) is not None,
                'invalid exact source selection')
        require(SHA.fullmatch(v['inputs_sha256']) is not None, 'invalid input closure identity')
        require(digest(HERE/'run-native.py') == v['controller_sha256'], 'controller differs from selection')
        require(digest(HERE/'generate-fixtures.py') == v['generator_sha256'], 'fixture generator differs from selection')
        require(set(v['helpers']) == set(HELPERS), 'incomplete controller helper closure')
        captured = {}
        for name in HELPERS:
            path = contained(ROOT, name)
            captured[name] = path.read_bytes()
            require(hashlib.sha256(captured[name]).hexdigest() == v['helpers'][name], 'controller helper changed: '+name)
        # Load transitive local imports from the same verified captured bytes before
        # their consumers execute. Replacing any cached same-name module prevents
        # an ambient/preloaded helper from supplying validation; sys.path is unchanged.
        load_module('check_hardware_gate', ROOT/'tools/check_hardware_gate.py', captured['tools/check_hardware_gate.py'])
        self.q = load_module('release_qualification', ROOT/'tools/release_qualification.py', captured['tools/release_qualification.py'])
        self.inputs = load_module('release_inputs', ROOT/'tools/release_inputs.py', captured['tools/release_inputs.py'])
        self.view = load_module('release_input_view', ROOT/'tools/release_input_view.py', captured['tools/release_input_view.py'])
        self.controller = load_module('bound_loader_signal_controller', ROOT/(SHARED+'native_env_controller.py'), captured[SHARED+'native_env_controller.py'])
        self.finalization = load_module('bound_loader_finalization', ROOT/(SHARED+'native_finalization.py'), captured[SHARED+'native_finalization.py'])
        # Existing lease verifier and its adjacent modules have all been fingerprinted above.
        self.admission = load_module('bound_loader_lease', ROOT/(SHARED+'qualify-native.py'), captured[SHARED+'qualify-native.py'])
        self.evidence_root = evidence.resolve(strict=True)
        require(evidence.is_dir() and not evidence.is_symlink(), 'invalid build evidence root')
        class ContainedEvidence(self.q.Evidence):
            def read(self, name):
                path = contained(self.root, name)
                require(stat.S_ISREG(path.lstat().st_mode), 'nonregular build evidence')
                return path.read_bytes()
        self.evidence = ContainedEvidence(self.evidence_root)
        self.binary = binary.resolve(strict=True)
        require(not binary.is_symlink(), 'test binary cannot be a symlink')
        require(self.binary == contained(self.evidence_root, v['binary']['path']), 'binary not selected build artifact')
        self.fixtures = fixtures.resolve(strict=True)
        require(fixtures.is_dir() and not fixtures.is_symlink(), 'invalid fixture root')
        self.manifest = None

    def verify(self):
        v = self.value
        require(digest(self.selection_path) == self.sha, 'selection changed during run')
        require(digest(HERE/'run-native.py') == v['controller_sha256'], 'controller drift')
        require(digest(HERE/'generate-fixtures.py') == v['generator_sha256'], 'generator drift')
        for name, expected in v['helpers'].items():
            require(identity(contained(ROOT, name))['sha256'] == expected, 'helper drift: '+name)
        source = self.evidence.obj(v['source'])
        require(source['commit'] == v['source_commit'] and source['tree'] == v['source_tree'] and
                source['inputs_sha256'] == v['inputs_sha256'], 'selected source/input mismatch')
        require(self.q.commit(ROOT, 'HEAD') == v['source_commit'], 'current source is not selected commit')
        self.inputs.verify_checkout(ROOT, v['source_commit'])
        proof = self.q.verify_source(source, ROOT, v['source_commit'])
        require(proof['tested_commit'] == proof['candidate_commit'] == v['source_commit'] and
                not proof['publication_equivalent'], 'fresh run cannot use publication equivalence')
        stock = self.evidence.obj(v['stock_build'])
        self.q.validate_build(stock, source, v['source'], self.evidence)
        test = self.evidence.obj(v['test_build'])
        require(test['schema'] == 'memra-bound-loader-test-build-v1' and type(test['exit_code']) is int and
                test['exit_code'] == 0 and test['started_utc'] and test['completed_utc'], 'incomplete test build')
        require(test['source'] == v['source'] and test['stock_build'] == v['stock_build'], 'test build source differs')
        for name in ('recipe', 'compiler_environment', 'source_before', 'source_after',
                     'input_view_before', 'input_view_after', 'cuda_arch', 'docs_rs', 'cuda_visible_devices'):
            require(test[name] == stock[name], 'test build changed controlled input: '+name)
        cargo_at = stock['command'].index('/toolchain/bin/cargo')
        expected = stock['command'][:cargo_at] + ['/toolchain/bin/cargo', 'test', '--locked', '--release',
                    '--no-run', '--message-format=json', '-p', 'memra-engine', '--test', 'bound_loader_gpu']
        require(test['command'] == expected, 'wrong controlled bound-loader build command')
        artifact = test['artifact']
        require(artifact == v['binary'], 'selected test artifact differs from build')
        require(identity(self.binary, executable=True) == {k:artifact[k] for k in ('bytes','sha256')},
                'test binary bytes changed')
        with self.binary.open('rb') as stream:
            header = stream.read(20)
        require(len(header) == 20 and header[:6] == b'\x7fELF\x02\x01' and header[18:20] == b'\x3e\0',
                'test artifact is not x86_64 native ELF')
        relative(artifact['path'])
        require('/target/release/deps/' in '/'+artifact['path'] and
                self.binary.name.startswith('bound_loader_gpu-'), 'wrong test artifact location')
        events = [parse(line) for line in self.evidence.bound(test['events']).splitlines() if line.strip()]
        self.evidence.bound(test['stderr'])
        finished = [e.get('success') is True for e in events if e.get('reason') == 'build-finished']
        matches = [e for e in events if e.get('reason') == 'compiler-artifact' and e.get('executable') and
                   e.get('target',{}).get('name') == 'bound_loader_gpu']
        require(finished == [True] and len(matches) == 1, 'missing/duplicate Cargo test completion')
        e = matches[0]
        require(e.get('fresh') is False and e.get('profile',{}).get('test') is True and
                e.get('target',{}).get('kind') == ['test'] and
                e['executable'] == '/target/release/deps/'+self.binary.name,
                'not the fresh selected Cargo test artifact')
        self.verify_fixtures()
        return {'source':v['source_commit'], 'inputs_sha256':v['inputs_sha256'],
                'stock_build_sha256':v['stock_build']['sha256'], 'test_build_sha256':v['test_build']['sha256'],
                'binary_sha256':v['binary']['sha256'], 'controller_sha256':v['controller_sha256'],
                'fixtures_sha256':v['fixtures_sha256'], 'selection_sha256':self.sha}

    def verify_fixtures(self):
        manifest_path = contained(self.fixtures, 'cases.json')
        raw = manifest_path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.value['fixtures_sha256'], 'fixture manifest drift')
        m = parse(raw)
        cases, files = m['cases'], m['files']
        require(isinstance(cases,list) and len(cases) == 20, 'require exactly 20 fixture cases')
        require(all(isinstance(c,dict) and isinstance(c.get('id'),str) and CASE_ID.fullmatch(c['id']) for c in cases),
                'invalid case id')
        require(len({c['id'] for c in cases}) == 20, 'duplicate case id')
        require(all(type(c['expect_error']) is bool and isinstance(c['expected_error_fragment'],str) and
                    bool(c['expected_error_fragment']) == c['expect_error'] for c in cases), 'invalid expected outcome')
        require(sum(not c['expect_error'] for c in cases) == 7, 'require 7 valid and 13 malformed cases')
        expected_files = set()
        for c in cases:
            require(c['format'] in ('gguf','hf'), 'invalid fixture format')
            require(c['id'].startswith(c['format']+'-'), 'case id/format mismatch')
            want = c['id']+'.gguf' if c['format']=='gguf' else c['id']
            require(c['path'] == want, 'case path differs from canonical id')
            relative(c['path'])
            if c['format']=='gguf': expected_files.add(c['path'])
            else: expected_files.update((c['path']+'/config.json',c['path']+'/model.safetensors'))
            require(c['native_shape']==[256,64] and type(c['nvfp4_query']) is bool, 'invalid fixture geometry')
            if not c['expect_error']:
                require(SHA.fullmatch(c['expected_output_sha256']) is not None and
                        type(c['expected_output_bytes']) is int and c['expected_output_bytes']>0 and
                        c['expected_output_kind'] in ('Float','Quant'), 'invalid expected output')
        require(isinstance(files,list) and all(isinstance(f,dict) for f in files), 'invalid fixture file manifest')
        names = [str(relative(f['path'])) for f in files]
        require(len(set(names)) == len(names) and set(names) == expected_files, 'incomplete fixture closure')
        actual = set()
        for base, dirs, paths in os.walk(self.fixtures, followlinks=False):
            for name in dirs:
                require(not (Path(base)/name).is_symlink(), 'symlinked fixture directory')
            for name in paths:
                path = Path(base)/name
                require(stat.S_ISREG(path.lstat().st_mode), 'nonregular fixture input')
                actual.add(str(path.relative_to(self.fixtures)))
        require(actual == expected_files|{'cases.json'}, 'unlisted fixture input')
        for f in files:
            require(type(f['bytes']) is int and f['bytes']>0 and SHA.fullmatch(f['sha256']) is not None,
                    'invalid fixture file identity')
            require(identity(contained(self.fixtures,f['path'])) == {k:f[k] for k in ('bytes','sha256')},
                    'fixture input bytes drifted: '+f['path'])
        self.manifest = m
