"""Native command transport with stub binaries; never GPU/serving qualification."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location('battery', ROOT/'tools/tier-battery.py')
B = importlib.util.module_from_spec(spec)
spec.loader.exec_module(B)


class NativeRunnerTests(unittest.TestCase):
    def stub_smi(self, root, fail=False):
        path = root/'nvidia-smi'
        path.write_text('#!'+sys.executable+'\n'+(
            'import sys\nprint("ERROR: telemetry unavailable",file=sys.stderr)\nsys.exit(7)\n' if fail else
            'import sys,time\n'
            'if "-lms" in sys.argv:\n'
            ' print("timestamp, index, power.draw",flush=True)\n'
            ' while True:\n'
            '  print("STUB, 0, 600",flush=True); time.sleep(.25)\n'
            'else: print("pid, process_name, used_memory\\n123, stub, 4 MiB")\n'))
        path.chmod(0o755)
        return path

    def test_fake_binary_same_interface_with_live_sampler_transport(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = ROOT/'crates/memra-tier/tests/battery/fake_runner.py'
            out = root/'outputs'; out.mkdir()
            record = B.SubprocessRunner(str(self.stub_smi(root))).run(
                [sys.executable,str(fake),'--out',str(out),'--arm','on'],
                root/'run.log',echo=False)
            self.assertEqual(record['result']['migrated_bytes'],3)
            self.assertEqual(record['status'],'executed-not-qualified')
            self.assertFalse(record['qualification'])
            self.assertEqual(record['gpu_telemetry']['interval_ms'],250)
            self.assertEqual(set(record['compute_apps']),{'before','after'})
            B.evidence(root,record['raw_log'])
            self.assertTrue((root/'run.capture.json').exists())

    def test_failure_quote_and_compute_apps_at_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = B.SubprocessRunner(str(self.stub_smi(root))).run(
                [sys.executable,'-c',"import sys; print('CUDA_ERROR_OUT_OF_MEMORY',file=sys.stderr); sys.exit(9)"],
                root/'run.log',echo=False)
            self.assertEqual(record['exit_code'],9)
            self.assertEqual(record['failure_quote'],'CUDA_ERROR_OUT_OF_MEMORY')
            self.assertIn('failure',record['compute_apps'])
            B.evidence(root,record['compute_apps']['failure']['raw_log'])

    def test_missing_executable_and_missing_sampler_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = B.SubprocessRunner(str(root/'absent-smi')).run([str(root/'absent')],root/'run.log',echo=False)
            self.assertEqual(record['exit_code'],127)
            self.assertEqual(record['gpu_telemetry']['status'],'unavailable')
            self.assertIn('ERROR: launch failed:',record['failure_quote'])

    def test_timeout_and_failed_sampler_never_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = B.SubprocessRunner(str(self.stub_smi(root,fail=True))).run(
                [sys.executable,'-c',"import time; print('starting',flush=True); time.sleep(9)"],
                root/'run.log',timeout=.2,echo=False)
            self.assertTrue(record['timed_out'])
            self.assertEqual(record['failure_quote'],'died, cause unknown — repro needed')
            self.assertIn(record['gpu_telemetry']['status'],('empty','exited-early'))
            self.assertEqual(record['status'],'failed')

    def test_cli_exit_status_and_raw_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root/'capture'
            proc = subprocess.run(
                [sys.executable, str(ROOT/'tools/tier-battery.py'), '--rig', 'rtx5090',
                 '--timeout', '5', '--out', str(out), '--execute', sys.executable,
                 '-c', "import sys; print('ERROR: deliberate native refusal'); sys.exit(9)"],
                env={**os.environ, 'PATH': str(root/'no-tools')}, capture_output=True, timeout=10)
            self.assertEqual(proc.returncode, 9, proc.stderr)
            record = json.loads((out/'command.capture.json').read_text())
            self.assertEqual(record['failure_quote'], 'ERROR: deliberate native refusal')
            self.assertFalse(record['qualification'])
            self.assertTrue(json.loads((out/'lock.json').read_text())['acquired'])
            B.evidence(out, record['raw_log'])

    def test_success_without_result_is_not_a_qualification(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = B.SubprocessRunner(str(root/'no-smi')).run(
                [sys.executable, '-c', "print('plain native output')"], root/'run.log', echo=False)
            self.assertEqual(record['status'], 'executed-not-qualified')
            self.assertIsNone(record['result'])
            self.assertFalse(record['qualification'])

    def test_malformed_result_keeps_raw_and_is_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            record = B.SubprocessRunner(str(root/'absent-smi')).run(
                [sys.executable,'-c',"print('RESULT not-json')"],root/'run.log',echo=False)
            self.assertEqual(record['status'],'failed')
            self.assertIsNotNone(record['parse_error'])
            self.assertEqual((root/'run.log').read_text(),'RESULT not-json\n')


class BootstrapTests(unittest.TestCase):
    def run_script(self, root, script=None, branch='lane/spill-integ-test'):
        source = ROOT/'tools/tier-rig-bootstrap.sh'
        if script is not None:
            source = root/'bootstrap.sh'; source.write_text(script)
        out = root/'out'
        proc = subprocess.run(['bash',str(source),'--dry-run','--out',str(out)],
                              env={**os.environ,'BRANCH':branch},cwd=ROOT,
                              capture_output=True,text=True,timeout=30)
        return proc, out

    def test_every_stage_stubbed_and_no_hardware_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            proc,out = self.run_script(Path(tmp))
            self.assertEqual(proc.returncode,0,proc.stderr)
            report = json.loads((out/'BOOTSTRAP.json').read_text())
            self.assertEqual(report['status'],'dry-run-complete-not-qualified')
            self.assertEqual(report['cuda_acceptance'],'stubbed-not-hardware')
            self.assertTrue(all(s['stubbed'] for s in report['steps']))
            names = {s['name'] for s in report['steps']}
            self.assertTrue({'accept-1','accept-gap','accept-2','build-1','remote-branch',
                             'minimum-source','rustup-install','openssl-development'} <= names)
            self.assertEqual(report['minimum_commit'], '020d20479cd686835c0fb7743040947d0fc2723b')
            self.assertEqual(report['nvcc'],'/stub/cuda-13.2/bin/nvcc')
            self.assertFalse(json.loads((out/'TOPOLOGY.json').read_text())['route_qualification'])
            before = (out/'BOOTSTRAP.json').read_bytes()
            proc,_ = self.run_script(Path(tmp))
            self.assertNotEqual(proc.returncode,0)
            self.assertEqual(before,(out/'BOOTSTRAP.json').read_bytes())

    def test_main_and_unset_branch_refuse(self):
        for branch in ('','main','lane/spill-../main'):
            with tempfile.TemporaryDirectory() as tmp:
                proc,out = self.run_script(Path(tmp),branch=branch)
                self.assertNotEqual(proc.returncode,0)
                self.assertFalse(out.exists())

    def test_low_power_and_bad_readback_and_unknown_remote_refuse(self):
        original = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        mutations = [
            original.replace('32607, 600.00, 600.00','32607, 600.00, 575.00'),
            original.replace('32607, 600.00, 600.00','32607, NaN, 600.00'),
            original.replace("'rustc 1.97.0 (stub)\\n'", "'rustc 1.96.0 (stub)\\n'"),
            original.replace("'compute_120a\\n'", "'compute_90a\\n'"),
            original.replace("code = 0", "code = 1 if label.startswith('nvcc-version') else 0"),
            original.replace("code = 0", "code = 1 if label == 'minimum-source' else 0"),
            original.replace("code = 0", "code = 1 if label == 'build-1' else 0"),
            original.replace("'ACCEPT: 8589934592 bytes cudaMalloc+memset+full-readback MATCH\\n', timeout=180", "'ERROR: readback failed\\n', timeout=180"),
            original.replace("'1'*40+'\\trefs/heads/'+branch+'\\n'", "''"),
        ]
        for script in mutations:
            self.assertNotEqual(script,original)
            with tempfile.TemporaryDirectory() as tmp:
                proc,out = self.run_script(Path(tmp),script=script)
                self.assertNotEqual(proc.returncode,0)
                self.assertEqual(json.loads((out/'BOOTSTRAP.json').read_text())['status'],'blocked')
    def test_acceptance_failure_retains_source_and_occupied_gpu_refuses(self):
        original = (ROOT/'tools/tier-rig-bootstrap.sh').read_text()
        for label in ('accept-1', 'accept-2'):
            script = original.replace('code = 0', f"code = 9 if label == '{label}' else 0")
            with tempfile.TemporaryDirectory() as tmp:
                proc, out = self.run_script(Path(tmp), script=script)
                self.assertEqual(proc.returncode, 2)
                self.assertTrue((out/'accept.cu').is_file())
                self.assertTrue((out/'compute-after.log').is_file())
                report = json.loads((out/'BOOTSTRAP.json').read_text())
                self.assertEqual(report['status'], 'blocked')
                self.assertNotIn('build-0', {s['name'] for s in report['steps']})
        script = original.replace("'pid, process_name, used_memory [MiB]\\n'",
                                  "'pid, process_name, used_memory [MiB]\\n123, other, 4 MiB\\n'")
        with tempfile.TemporaryDirectory() as tmp:
            proc, out = self.run_script(Path(tmp), script=script)
            self.assertEqual(proc.returncode, 2)
            self.assertIn('GPU occupied', json.loads((out/'BOOTSTRAP.json').read_text())['blocker'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
