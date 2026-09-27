#!/usr/bin/env python3
"""Fresh interpreter helper/CLI controls; simulated build records, no lease or native launch."""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest

from test_native_runner import Fixture, HERE, write


class FreshCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shared = tempfile.TemporaryDirectory(prefix='541-fresh-cli-fixtures-')
        cls.fixtures = Path(cls.shared.name) / 'fixtures'
        subprocess.run([sys.executable, '-B', str(HERE / 'generate-fixtures.py'), str(cls.fixtures)],
                       check=True, stdout=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls):
        cls.shared.cleanup()

    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix='541-fresh-cli-control-')
        self.f = Fixture(Path(self.scratch.name), self.fixtures)
        self.marker = self.f.base / 'unrelated-helper-used'

    def tearDown(self):
        self.scratch.cleanup()

    def nearby_helpers(self):
        # These files are importable from the script directory. They are not the
        # selected tools helpers, and must not participate in validation.
        for name in ('check_hardware_gate', 'release_qualification', 'release_inputs', 'release_input_view'):
            (self.f.repo / self.f.prefix / (name + '.py')).write_text(
                'from pathlib import Path\n'
                f'Path({str(self.marker)!r}).write_text("unrelated helper executed")\n'
                'raise RuntimeError("unrelated helper supplied validation")\n')

    def run_cli(self, expected_error=None, *, preloaded=False):
        f = self.f
        for optimize in (False, True):
            with self.subTest(optimize=optimize, preloaded=preloaded):
                out = f.base / ('out-optimized' if optimize else 'out-normal')
                runner = f.repo / f.prefix / 'run-native.py'
                args = [str(runner), '--selection', str(f.selection_path),
                        '--selection-sha256', f.selection_sha, '--evidence-root', str(f.evidence),
                        '--fixtures', str(f.fixtures), '--binary', str(f.binary),
                        '--output', str(out), '--wall-seconds', '20']
                command = [sys.executable, '-B', *(['-O'] if optimize else [])]
                if preloaded:
                    # This is a new interpreter with deliberately unrelated cached
                    # modules, not a patched lease, validator, or launch boundary.
                    code = (
                        'import sys,types,runpy\nfrom pathlib import Path\n'
                        'def unrelated(name):\n'
                        f' Path({str(self.marker)!r}).write_text(name)\n'
                        ' raise RuntimeError("unrelated helper supplied validation")\n'
                        'for name in ("check_hardware_gate","release_qualification",'
                        '"release_inputs","release_input_view"):\n'
                        ' m=types.ModuleType(name);m.__getattr__=unrelated;sys.modules[name]=m\n'
                        'sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name="__main__")\n'
                    )
                    command.extend(['-c', code])
                command.extend(args)
                env = {k: v for k, v in os.environ.items()
                       if k != 'PYTHONPATH' and k != 'DOCS_RS' and not k.startswith('MEMRA_')}
                env['CUDA_VISIBLE_DEVICES'] = ''
                process = subprocess.run(command, cwd=f.base, env=env, capture_output=True, timeout=30)
                result = json.loads((out / 'result.json').read_text())
                if os.environ.get('NATIVE_RUNNER_CPU_EVIDENCE'):
                    saved = Path(os.environ['NATIVE_RUNNER_CPU_EVIDENCE']) / self._testMethodName / str(int(optimize))
                    saved.mkdir(parents=True, exist_ok=False)
                    (saved / 'stdout.log').write_bytes(process.stdout)
                    (saved / 'stderr.log').write_bytes(process.stderr)
                    write(saved / 'result.json', result)
                    write(saved / 'invocation.json', {
                        'scope': 'fresh actual CLI import/source/build validation; simulated CPU records; no lease shim',
                        'command': command, 'PYTHONPATH_present': 'PYTHONPATH' in env,
                        'exit_code': process.returncode, 'optimize': optimize,
                        'helper_selection': f.selection['helpers'],
                        'unrelated_helper_executed': self.marker.exists(),
                    })
                self.assertEqual(process.returncode, 1, result)
                self.assertEqual(result['status'], 'failed')
                self.assertEqual(result['results'], [])
                self.assertNotIn('active_child', result)
                self.assertFalse(self.marker.exists(), 'an unrelated module participated in validation')
                if expected_error is None:
                    self.assertEqual(result['binding']['source'], f.source['commit'])
                    self.assertEqual(result['binding']['selection_sha256'], f.selection_sha)
                    refusal = ('MEMRA_GPU_LEASE_FILE missing' if sys.platform == 'linux'
                               else 'GPU phase requires Linux /proc lock verification')
                    self.assertIn(refusal, result['error'])
                    self.assertNotIn('lease', result)
                else:
                    self.assertIn(expected_error, result['error'])
                    self.assertNotIn('binding', result)

    def test_fresh_cli_without_pythonpath_reaches_real_lease_refusal(self):
        self.run_cli()

    def test_nearby_same_name_helpers_cannot_supply_validation(self):
        self.nearby_helpers()
        self.run_cli()

    def test_preloaded_same_name_helpers_cannot_supply_validation(self):
        self.run_cli(preloaded=True)

    def test_missing_selected_helper_cannot_fall_back_to_nearby_helper(self):
        self.nearby_helpers()
        (self.f.repo / 'tools/check_hardware_gate.py').unlink()
        self.run_cli('check_hardware_gate.py')

    def test_missing_transitive_selection_refuses_despite_nearby_helper(self):
        self.nearby_helpers()
        del self.f.selection['helpers']['tools/check_hardware_gate.py']
        self.f.seal()
        self.run_cli('incomplete controller helper closure')

    def test_wrong_transitive_hash_refuses_before_validation_helpers_execute(self):
        self.nearby_helpers()
        self.f.selection['helpers']['tools/check_hardware_gate.py'] = '0' * 64
        self.f.seal()
        self.run_cli('controller helper changed: tools/check_hardware_gate.py')

    def test_changed_source_is_not_validated_by_preloaded_helper(self):
        (self.f.repo / 'Cargo.lock').write_text('# changed source\n')
        self.run_cli('actual tracked input differs', preloaded=True)

    def test_failed_build_is_not_validated_by_nearby_helper(self):
        self.nearby_helpers()
        self.f.test['exit_code'] = 1
        self.f.update_test()
        self.run_cli('incomplete test build')


if __name__ == '__main__':
    unittest.main(verbosity=2)
