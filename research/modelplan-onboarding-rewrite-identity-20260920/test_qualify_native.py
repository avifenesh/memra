"""CPU regressions for lease ownership and source-bound owned builds; no GPU execution."""
import copy
import importlib.util
import os
import json
import shutil
import subprocess
import sys
from pathlib import Path
import stat
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

spec = importlib.util.spec_from_file_location('qualify_native', Path(__file__).with_name('qualify-native.py'))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class LeaseTests(unittest.TestCase):
    def setUp(self):
        self.gpu = 'GPU-12345678-abcd-abcd-abcd-123456789abc'
        self.second = 'GPU-22345678-abcd-abcd-abcd-123456789abc'
        self.lease = {'wrapper_pid': 4101, 'child_pid': 4102, 'requested_uuids': [self.gpu],
                      'lock_order': [self.gpu], 'lock_files': {self.gpu: f'/tmp/memra-gpu-locks/{self.gpu}.lock'}}
        self.info = SimpleNamespace(st_mode=stat.S_IFREG | 0o600, st_dev=os.makedev(8, 1), st_ino=731)
        self.rows = '4: FLOCK ADVISORY WRITE 4101 08:01:731 0 EOF\n'

    def verify(self, lease=None, visible=None, ancestors=None, rows=None):
        gate.verify_lock_records(lease or self.lease, visible or self.gpu,
                                 {4101, 4102, os.getpid()} if ancestors is None else ancestors,
                                 self.rows if rows is None else rows, lambda _: self.info)

    def test_digest_does_not_require_python311_file_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'bytes'
            for payload in (b'', b'\x00\xffcheckpoint\n' * 100000):
                path.write_bytes(payload)
                expected = gate.hashlib.sha256(payload).hexdigest()
                with patch.object(gate.hashlib, 'file_digest', None, create=True):
                    self.assertEqual(gate.digest(path), expected)

    def test_real_wrapper_shape_accepts_exact_exclusive_card(self):
        self.verify()

    def test_marker_without_ancestor_owned_flock_refuses(self):
        for rows in ['', self.rows.replace('4101', '4103'), self.rows.replace('FLOCK', 'POSIX'),
                     self.rows.replace('WRITE', 'READ'), self.rows.replace('731', '732'),
                     self.rows.replace('0 EOF', '0 4095')]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.verify(rows=rows)
        with self.assertRaises(RuntimeError):
            self.verify(ancestors={4102, os.getpid()})

    def test_visible_device_order_and_exact_set_are_mandatory(self):
        for visible in ['0', self.second, f'{self.gpu},{self.second}']:
            with self.subTest(visible=visible), self.assertRaises(RuntimeError):
                self.verify(visible=visible)
        extra = copy.deepcopy(self.lease)
        extra['lock_files'][self.second] = f'/tmp/memra-gpu-locks/{self.second}.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=extra)

    def test_interrupted_negative_control_cannot_pass(self):
        text = 'REWRITE_IDENTITY_GATE_FAIL: MODEL_LOAD: does not bind numeric_program_sha256=abc'
        refusal = 'does not bind numeric_program_sha256='
        self.assertTrue(gate.case_passed(1, refusal, text))
        for code in [-9, -15, 137, 143, 2, 0]:
            with self.subTest(code=code):
                self.assertFalse(gate.case_passed(code, refusal, text))
        self.assertFalse(gate.case_passed(1, refusal, refusal))

    def test_output_hashes_require_every_prompt_exactly_once(self):
        lines = [f'OUTPUT stage=installed-eager prompt={i} values=1 sha256={str(i) * 64}' for i in range(3)]
        self.assertEqual(set(gate.output_hashes('\n'.join(lines), 'installed-eager')), {'0', '1', '2'})
        for bad in ['\n'.join(lines[:2]), '\n'.join(lines + lines[:1]), '']:
            with self.subTest(text=bad), self.assertRaises(RuntimeError):
                gate.output_hashes(bad, 'installed-eager')

    def test_noncanonical_or_partial_lockset_refuses(self):
        wrong = copy.deepcopy(self.lease)
        wrong['lock_files'][self.gpu] = '/tmp/memra-gpu.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=wrong)
        partial = copy.deepcopy(self.lease)
        partial['requested_uuids'] = [self.second, self.gpu]
        partial['lock_order'] = sorted(partial['requested_uuids'])
        partial['lock_files'][self.second] = f'/tmp/memra-gpu-locks/{self.second}.lock'
        with self.assertRaises(RuntimeError):
            self.verify(lease=partial, visible=','.join(partial['requested_uuids']), rows=self.rows.replace('731', '799'))
        duplicate = copy.deepcopy(self.lease)
        duplicate['requested_uuids'] *= 2
        with self.assertRaises(RuntimeError):
            self.verify(lease=duplicate)


class GPUReached(Exception):
    pass


class BuildRecordTests(unittest.TestCase):
    """Use real Git and an executable CPU Cargo fixture, not a hand-written success receipt."""
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / 'source'
        self.root.mkdir()
        self.builder = gate.build_record
        for name in (self.builder.PRODUCER, self.builder.RUNNER):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(gate.ROOT / name, target)
        for name, content in {'Cargo.toml': '[workspace]\n', 'Cargo.lock': 'version = 4\n',
                              'rust-toolchain.toml': '[toolchain]\nchannel = "fixture"\n',
                              '.gitignore': '*.ignored\n', 'crates/native/build.rs': 'fn main() {}\n',
                              'crates/native/src/lib.rs': 'pub fn native() {}\n',
                              '.cargo/config.toml': '[build]\njobs = 2\n'}.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.git('init', '-q')
        self.commit()
        self.tool_dir = self.base / 'tools'
        self.tool_dir.mkdir()
        script = '''import json, os, pathlib, sys
here = pathlib.Path(__file__).parent
name = pathlib.Path(__file__).name
if name == 'rustup':
    print(here / sys.argv[-1]); sys.exit(0)
if '--version' in sys.argv or '-Vv' in sys.argv:
    print(name + ' CPU fixture 1'); sys.exit(0)
control_path = here / 'control.json'
control = json.loads(control_path.read_text()) if control_path.exists() else {}
(here / 'invocation.json').write_text(json.dumps({'args': sys.argv, 'env': dict(os.environ)}))
if control.get('failure'):
    print('deliberate build failure', file=sys.stderr); sys.exit(7)
target = pathlib.Path(sys.argv[sys.argv.index('--target-dir') + 1]) / 'release'
target.mkdir(parents=True)
for i, arg in enumerate(sys.argv):
    if arg != '--bin': continue
    name = sys.argv[i + 1]
    if name == control.get('omit'): continue
    path = target / name
    path.write_text('#!/bin/sh\\nexit 0\\n# ' + name + '\\n')
    path.chmod(0o755)
    print(json.dumps({'reason':'compiler-artifact', 'target':{'name':name},
                      'executable':str(path), 'fresh':control.get('fresh', False)}))
if not control.get('no_finish'):
    print(json.dumps({'reason':'build-finished','success':True}))
if control.get('change_source'):
    pathlib.Path('crates/native/src/lib.rs').write_text('edited during build')
if control.get('change_tool'):
    with (here / 'nvcc').open('a') as stream: stream.write('\\n# changed compiler\\n')
'''
        for name in ('rustup', 'cargo', 'rustc', 'nvcc', 'cc', 'c++', 'gcc', 'g++', 'ar', 'ld'):
            path = self.tool_dir / name
            path.write_text(f'#!{sys.executable}\n' + script)
            path.chmod(0o755)
        self.home = self.base / 'home'
        self.home.mkdir()
        self.environment = patch.dict(os.environ, {'HOME': str(self.home), 'PATH': f'{self.tool_dir}:/usr/bin:/bin'}, clear=True)
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.out = self.base / 'owned-build'
        self.receipt = self.out / 'build.json'
        self.binary_dir = self.out / 'target/release'
        self.args = SimpleNamespace(model=self.base / 'model', out=self.base / 'qualification',
                                    binary_dir=self.binary_dir, build_record=self.receipt,
                                    mtp_model=None, mtp_sha256=None)

    def git(self, *args):
        return subprocess.check_output(['git', *args], cwd=self.root, stderr=subprocess.STDOUT)

    def commit(self):
        self.git('add', '.')
        self.git('-c', 'user.name=CPU fixture', '-c', 'user.email=fixture@example.invalid', 'commit', '-qm', 'fixture')

    def build(self, **control):
        (self.tool_dir / 'control.json').write_text(json.dumps(control))
        return self.builder.build(self.root, self.out, self.tool_dir / 'nvcc', '120a')

    def verify(self, expected=None):
        return self.builder.verify_build_record(self.root, self.receipt, self.binary_dir, expected)

    def before_gpu(self, accepts=False, mutate=None):
        """Exercise the real runner up to nvidia-smi, with only lease/model checks replaced."""
        reached = []
        original = subprocess.Popen

        def no_gpu(command, *args, **kwargs):
            if str(command[0]) == 'nvidia-smi' or Path(command[0]).parent == self.binary_dir:
                reached.append(command)
                raise GPUReached()
            return original(command, *args, **kwargs)

        def model_source(_):
            if mutate:
                mutate()
            return {'source': 'CPU fixture'}

        with patch.object(gate, 'ROOT', self.root), patch.object(gate, 'verify_lease', return_value={'requested_uuids': []}), \
                patch.object(gate, 'verify_source', side_effect=model_source), \
                patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'CPU-only-lease'}), \
                patch.object(subprocess, 'Popen', side_effect=no_gpu):
            with self.assertRaises(GPUReached if accepts else RuntimeError):
                gate.run(self.args)
        self.assertEqual(len(reached), 1 if accepts else 0)
        self.assertFalse((self.args.out / 'result.json').exists())

    def test_owned_success_binds_source_build_inputs_and_all_executables(self):
        record = self.build()
        verified, record_hash = self.verify()
        self.assertEqual(record, verified)
        self.assertEqual(record_hash, self.builder.digest(self.receipt))
        self.assertEqual(set(record['binaries']), set(self.builder.BINARIES))
        self.assertIn('crates/native/build.rs', record['source']['files'])
        self.assertIn('Cargo.lock', record['source']['files'])
        self.assertIn(str(self.root / '.cargo/config.toml'), record['inputs']['cargo_configs'])
        invocation = json.loads((self.tool_dir / 'invocation.json').read_text())
        self.assertIn('--locked', invocation['args'])
        self.assertEqual(invocation['env']['MEMRA_CUDA_ARCH'], '120a')
        self.assertEqual(invocation['env']['CUDA_VISIBLE_DEVICES'], '')
        self.before_gpu(accepts=True)
        binaries = json.loads((self.args.out / 'binaries.json').read_text())
        self.assertEqual(binaries['run-spec'], self.builder.digest(self.binary_dir / 'run-spec'))

    def test_missing_and_historical_manual_records_refuse_before_gpu(self):
        self.before_gpu()
        self.out.mkdir()
        self.receipt.write_text(json.dumps({'source_commit': self.git('rev-parse', 'HEAD').decode().strip(),
                                           'returncode': 0, 'binaries': {}}))
        historical = self.receipt.read_bytes()
        self.before_gpu()
        self.assertEqual(self.receipt.read_bytes(), historical)
        self.args.build_record = None
        self.before_gpu()

    def test_clean_new_checkout_refuses_stale_build(self):
        self.build()
        (self.root / 'crates/native/src/lib.rs').write_text('new source\n')
        self.commit()
        self.before_gpu()

    def test_dirty_source_untracked_and_ignored_inputs_refuse_before_gpu(self):
        self.build()
        for name in ('crates/native/src/lib.rs', 'crates/new.rs', 'crates/extra.ignored'):
            path = self.root / name
            previous = path.read_bytes() if path.exists() else None
            with self.subTest(name=name):
                path.write_text('unbuilt input\n')
                self.before_gpu()
            if previous is None:
                path.unlink()
            else:
                path.write_bytes(previous)

    def test_staged_source_and_hidden_index_changes_refuse(self):
        self.build()
        name = 'crates/native/src/lib.rs'
        original = (self.root / name).read_bytes()
        (self.root / name).write_text('staged\n')
        self.git('add', name)
        self.before_gpu()
        (self.root / name).write_bytes(original)
        self.git('add', name)
        for flag in ('assume-unchanged', 'skip-worktree'):
            with self.subTest(flag=flag):
                self.git('update-index', f'--{flag}', name)
                (self.root / name).write_text('hidden modification\n')
                self.before_gpu()
                (self.root / name).write_bytes(original)
                self.git('update-index', f'--no-{flag}', name)

    def test_stale_missing_and_nonexecutable_tools_include_optional_run_spec(self):
        self.build()
        for name in self.builder.BINARIES:
            path = self.binary_dir / name
            original = path.read_bytes()
            with self.subTest(name=name):
                path.write_bytes(original + b'stale')
                self.before_gpu()
                path.write_bytes(original)
        path = self.binary_dir / 'run-spec'
        self.args.mtp_model = self.base / 'mtp.gguf'
        self.args.mtp_model.write_bytes(b'CPU artifact')
        self.args.mtp_sha256 = self.builder.digest(self.args.mtp_model)
        path.chmod(0o644)
        self.before_gpu()
        path.unlink()
        self.before_gpu()

    def test_binary_replaced_after_initial_preflight_refuses_before_first_gpu(self):
        self.build()
        self.before_gpu(mutate=lambda: (self.binary_dir / 'memra').write_bytes(b'replaced'))

    def test_source_changed_after_initial_preflight_refuses_before_first_gpu(self):
        self.build()
        self.before_gpu(mutate=lambda: (self.root / 'Cargo.lock').write_text('changed\n'))

    def test_incomplete_tampered_duplicate_and_failed_records_refuse_before_gpu(self):
        record = self.build()
        for key in ('source', 'inputs', 'inputs_sha256', 'command', 'binaries', 'logs', 'producer', 'completed_utc'):
            incomplete = copy.deepcopy(record)
            del incomplete[key]
            self.receipt.write_text(json.dumps(incomplete))
            with self.subTest(missing=key):
                self.before_gpu()
        for change in ({'returncode': 7}, {'returncode': False}, {'status': 'started'},
                       {'inputs_sha256': '0' * 64}, {'binaries': {}}, {'source': {}}):
            self.receipt.write_text(json.dumps({**record, **change}))
            with self.subTest(change=change):
                self.before_gpu()
        self.receipt.write_text('{"schema":"one","schema":"two"}')
        self.before_gpu()
        self.receipt.write_text('{')
        self.before_gpu()

    def test_external_compiler_config_and_log_changes_refuse(self):
        self.build()
        for path in (self.tool_dir / 'nvcc', self.out / 'cargo-events.jsonl', self.out / 'cargo-stderr.log'):
            original = path.read_bytes()
            path.write_bytes(original + b'\n#changed\n')
            with self.subTest(path=path):
                self.before_gpu()
            path.write_bytes(original)
        config = self.home / '.cargo/config.toml'
        config.parent.mkdir()
        config.write_text('[build]\nrustflags = ["--cfg", "changed"]\n')
        self.before_gpu()

    def test_pinned_record_cannot_be_replaced_after_preflight(self):
        record = self.build()
        _, pinned = self.verify()
        record['completed_utc'] = 'different'
        self.receipt.write_text(json.dumps(record))
        with self.assertRaisesRegex(RuntimeError, 'changed after preflight'):
            self.verify(pinned)

    def test_incomplete_build_inputs_even_with_recomputed_hash_refuse(self):
        record = self.build()
        for field in ('tools', 'environment', 'cargo_configs'):
            broken = copy.deepcopy(record)
            del broken['inputs'][field]
            broken['inputs_sha256'] = self.builder.content_hash(broken['inputs'])
            self.receipt.write_text(json.dumps(broken))
            with self.subTest(field=field):
                self.before_gpu()
        broken = copy.deepcopy(record)
        del broken['inputs']['tools']['nvcc']
        broken['inputs_sha256'] = self.builder.content_hash(broken['inputs'])
        self.receipt.write_text(json.dumps(broken))
        self.before_gpu()

    def test_interrupted_build_never_writes_record(self):
        original = subprocess.run

        def interrupt(command, *args, **kwargs):
            if 'build' in command:
                raise KeyboardInterrupt()
            return original(command, *args, **kwargs)

        with patch.object(subprocess, 'run', side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.build()
        self.assertFalse(self.receipt.exists())

    def test_build_drops_ambient_flags_and_owns_new_target(self):
        with patch.dict(os.environ, {'RUSTFLAGS': '--cfg unrecorded', 'CARGO_TARGET_DIR': '/unowned/target',
                                    'RUSTC_WRAPPER': '/unowned/wrapper', 'MEMRA_CUTLASS': '/unowned/headers'}):
            record = self.build()
        environment = record['inputs']['environment']
        for key in ('RUSTFLAGS', 'CARGO_TARGET_DIR', 'RUSTC_WRAPPER', 'MEMRA_CUTLASS'):
            self.assertNotIn(key, environment)
        self.assertEqual(environment['RUSTC'], record['inputs']['tools']['rustc']['path'])
        self.assertEqual(record['command'][record['command'].index('--target-dir') + 1], str(self.out / 'target'))

    def test_unsuccessful_incomplete_or_incremental_build_never_writes_record(self):
        for control in ({'failure': True}, {'no_finish': True}, {'omit': 'run-spec'}, {'fresh': True}):
            with self.subTest(control=control), self.assertRaises(RuntimeError):
                self.build(**control)
            self.assertFalse(self.receipt.exists())
            shutil.rmtree(self.out)

    def test_source_or_compiler_change_during_build_never_writes_record(self):
        with self.assertRaises(RuntimeError):
            self.build(change_source=True)
        self.assertFalse(self.receipt.exists())
        (self.root / 'crates/native/src/lib.rs').write_text('pub fn native() {}\n')
        shutil.rmtree(self.out)
        with self.assertRaises(RuntimeError):
            self.build(change_tool=True)
        self.assertFalse(self.receipt.exists())

    def test_existing_output_is_never_blessed_or_overwritten(self):
        self.build()
        original = self.receipt.read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual(self.receipt.read_bytes(), original)

    def test_docs_rs_builds_and_config_injection_refuse_without_building(self):
        with patch.dict(os.environ, {'DOCS_RS': '1'}), self.assertRaisesRegex(RuntimeError, 'DOCS_RS'):
            self.build()
        config = self.home / '.cargo/config.toml'
        config.parent.mkdir()
        config.write_text('[env]\nDOCS_RS = "1"\n')
        with self.assertRaisesRegex(RuntimeError, 'DOCS_RS'):
            self.build()
        self.assertFalse(self.receipt.exists())
        self.assertFalse((self.tool_dir / 'invocation.json').exists())

    def simulated_qualification(self, change_after_launch=False, change_after_last_case=False):
        """Model native responses only; keep real source/build/binary acceptance checks."""
        original_run, original_popen, original_write = subprocess.run, subprocess.Popen, gate.write_json
        launched = []

        def run(command, *args, **kwargs):
            if command[0] == 'nvidia-smi':
                kwargs['stdout'].write(b'gpu_uuid,pid,process_name,used_memory\n')
                return subprocess.CompletedProcess(command, 0)
            return original_run(command, *args, **kwargs)

        def popen(command, *args, **kwargs):
            native = Path(command[0]).parent in (self.binary_dir, self.args.out)
            if command[0] != 'nvidia-smi' and not native:
                return original_popen(command, *args, **kwargs)
            process = SimpleNamespace(returncode=0, poll=lambda: 0, wait=lambda **kw: 0, terminate=lambda: None)
            if not native:
                return process
            launched.append(command)
            log_name = Path(kwargs['stdout'].name).stem
            if log_name == 'inspect':
                bundle = self.args.out / 'bundle'
                bundle.mkdir()
                (bundle / 'artifact.lock').write_text('CPU fixture\n')
            refusals = {'missing-bundle': 'read artifact.lock',
                        'different-numerical-program': 'does not bind numeric_program_sha256=',
                        'different-weights-same-geometry': 'does not bind artifact_sha256=',
                        'different-executable-same-source': 'does not bind implementation_sha256='}
            if log_name in refusals:
                process.returncode = 1
                text = f'REWRITE_IDENTITY_GATE_FAIL: {refusals[log_name]}\n'
            else:
                text = ''.join(f'OUTPUT stage={stage} prompt={i} values=1 sha256={str(i) * 64}\n'
                               for stage in ('installed-eager', 'check-eager', 'fresh-kv-diagnostic') for i in range(3))
            kwargs['stdout'].write(text.encode())
            if change_after_launch:
                (self.binary_dir / 'memra').write_bytes(b'replaced during native case')
            return process

        def write(path, value):
            original_write(path, value)
            if change_after_last_case and path.name == 'cases.json' and value[-1]['case'] == 'legacy-batch-regression':
                (self.root / 'Cargo.lock').write_text('changed before publication\n')

        with patch.object(gate, 'ROOT', self.root), patch.object(gate, 'verify_lease', return_value={'requested_uuids': []}), \
                patch.object(gate, 'verify_source', return_value={'source': 'CPU fixture'}), \
                patch.object(gate, 'mutate_weight', return_value={'change': 'CPU fixture'}), \
                patch.dict(os.environ, {'MEMRA_GPU_LEASE_FILE': 'CPU-only-lease'}), \
                patch.object(subprocess, 'run', side_effect=run), patch.object(subprocess, 'Popen', side_effect=popen), \
                patch.object(gate, 'write_json', side_effect=write), patch('builtins.print'):
            gate.run(self.args)
        return launched

    def test_unchanged_build_can_publish_mocked_cases_bound_to_record(self):
        self.build()
        self.assertEqual(len(self.simulated_qualification()), 11)
        result = json.loads((self.args.out / 'result.json').read_text())
        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['build_record_sha256'], self.builder.digest(self.receipt))

    def test_binary_replaced_during_native_case_cannot_publish_passed_case(self):
        self.build()
        with self.assertRaisesRegex(RuntimeError, 'stale-binary'):
            self.simulated_qualification(change_after_launch=True)
        self.assertFalse((self.args.out / 'cases.json').exists())
        self.assertFalse((self.args.out / 'result.json').exists())

    def test_source_changed_after_last_case_cannot_publish_passed_result(self):
        self.build()
        with self.assertRaisesRegex(RuntimeError, 'clean source'):
            self.simulated_qualification(change_after_last_case=True)
        self.assertTrue((self.args.out / 'cases.json').exists())
        self.assertFalse((self.args.out / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
