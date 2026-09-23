"""CPU controls for the actual native gate math and evidence finalization."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, HERE / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load('gemma_hidden_runner', 'qualify-gemma-hidden.py')


class GemmaHiddenControls(unittest.TestCase):
    def test_f32_prerequisite_census_and_order_are_required(self):
        f32 = ''.join(f'GEMMA_F32_CASE_PASS case=f32-p{p}-t{n}\n' for p in (0, 11) for n in (15, 16, 17))
        generators = ''.join(f'GEMMA_F32_GENERATE_CASE_PASS case={api}-t{n}\n' for api in ('generate', 'generate-with') for n in (15, 16, 17))
        f32 += generators
        verdict = 'GEMMA_F32_PARITY_PASS streams=6 atol=0.005\n'
        hpost = ''.join(f'GEMMA_CASE_PASS case={name}\n' for name in gate.case_names())
        gate.validate_export_census(f32 + verdict + hpost)
        for invalid in (hpost, f32 + hpost, f32 + f32 + verdict + hpost,
                        f32.replace('p11-t17', 'p11-t18') + verdict + hpost,
                        f32.replace('generate-with-t17', 'generate-with-t18') + verdict + hpost,
                        f32.replace(generators, '') + verdict + hpost,
                        hpost + f32 + verdict, f32 + verdict + verdict + hpost):
            with self.assertRaises(RuntimeError):
                gate.validate_export_census(invalid)

    def test_actual_scalar_oracle_and_fixed_boundary(self):
        source = (ROOT/'crates/memra-engine/src/bin/rewrite_identity_gate/gemma_hidden.rs').read_text()
        # Compile the actual CPU oracle/comparator and their native-module tests.
        def function(name):
            body = source[source.index('fn '+name+'('):]
            opening = body.index('{'); depth = 1
            for i in range(opening+1, len(body)):
                depth += (body[i] == '{') - (body[i] == '}')
                if not depth:
                    return body[:i+1]
            raise ValueError('unclosed function')
        prefix = 'type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;\n'
        prefix += 'fn require(ok: bool, message: impl Into<String>) -> Result<()> { if ok {Ok(())} else {Err(message.into().into())} }\n'
        prefix += source[source.index('const ATOL:'):source.index('const WIDTHS:')]
        program = prefix + function('normalized_reference') + function('close') + source[source.index('#[cfg(test)]'):]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'test.rs'; path.write_text(program)
            subprocess.run(['rustc','--edition=2024','--test',str(path),'-o',str(Path(tmp)/'test')], check=True)
            subprocess.run([str(Path(tmp)/'test'),'--nocapture'],check=True)

    def test_cross_hpost_requires_every_raw_reference_and_consumer(self):
        with tempfile.TemporaryDirectory() as tmp:
            off,on = Path(tmp)/'off',Path(tmp)/'on'; off.mkdir();on.mkdir()
            with self.assertRaisesRegex(RuntimeError,'missing/empty'):
                gate.cross_hpost(off,on)
            for case in gate.case_names():
                for suffix in ['t1-raw','expected-normalized','actual-pooling'] + [f'{kind}-logits-{i}' for kind in ('t1','actual') for i in range(5)]:
                    for directory in (off,on):
                        (directory/f'{case}-{suffix}.f32').write_bytes(b'\0\0\x80?')
            result = gate.cross_hpost(off,on)
            self.assertEqual(len(result),156)
            self.assertTrue(all(x['bitwise_equal'] for x in result.values()))
            (on/'direct-p0-t16-actual-pooling.f32').write_bytes(b'\0\0\0@')
            self.assertFalse(gate.cross_hpost(off,on)['direct-p0-t16-actual-pooling.f32']['bitwise_equal'])

    def test_failed_attempt_preserves_and_quarantines_both_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for hpost in (0,1):
                directory=out/f'hpost-{hpost}';directory.mkdir()
                (directory/'rewrite-receipts.tsv').write_bytes(b'original\n')
            gate.fail_attempt(out,{'status':'passed'},'numeric failure',[])
            self.assertEqual(json.loads((out/'result.json').read_text())['status'],'failed')
            for hpost in (0,1):
                self.assertEqual((out/f'hpost-{hpost}/rewrite-receipts.failed.tsv').read_bytes(),b'original\n')
                self.assertFalse((out/f'hpost-{hpost}/rewrite-receipts.tsv').exists())

    def test_archive_failure_cannot_leave_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);bundle=out/'hpost-0';bundle.mkdir()
            (bundle/'rewrite-receipts.tsv').write_text('new')
            (bundle/'rewrite-receipts.failed.tsv').write_text('old')
            (bundle/'rewrite-receipts.failed-1.tsv').write_text('older')
            with patch.object(gate.native.finalization,'write_evidence_manifest',side_effect=OSError('manifest failure')):
                gate.fail_attempt(out,{'status':'passed'},'cancelled',[])
            result=json.loads((out/'result.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertFalse((bundle/'rewrite-receipts.tsv').exists())
            self.assertEqual(len(result['finalization_errors']),1)
            self.assertEqual((bundle/'rewrite-receipts.failed.tsv').read_text(),'old')
            self.assertEqual((bundle/'rewrite-receipts.failed-1.tsv').read_text(),'older')
            self.assertEqual((bundle/'rewrite-receipts.failed-2.tsv').read_text(),'new')

    def test_initial_failed_publication_still_quarantines_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            for hpost in (0,1):
                directory=out/f'hpost-{hpost}';directory.mkdir()
                (directory/'rewrite-receipts.tsv').write_bytes(b'original\n')
            (out/'result.json').write_text('{"status":"passed"}\n')
            publish=gate.native.finalization.publish_result
            calls=[]
            def fail_first(*args,**kwargs):
                calls.append(True)
                if len(calls)==1:raise OSError('cannot publish initial failure')
                return publish(*args,**kwargs)
            with patch.object(gate.native.finalization,'publish_result',side_effect=fail_first):
                gate.fail_attempt(out,{'status':'passed'},'cancelled',[])
            self.assertEqual(json.loads((out/'result.json').read_text())['status'],'failed')
            self.assertEqual(json.loads((out/'result.publication-error.json').read_text())['status'],'passed')
            for hpost in (0,1):
                self.assertEqual((out/f'hpost-{hpost}/rewrite-receipts.failed.tsv').read_bytes(),b'original\n')
                self.assertFalse((out/f'hpost-{hpost}/rewrite-receipts.tsv').exists())

    def test_stalled_inventory_child_is_bounded_and_reaped(self):
        processes=[]
        popen=subprocess.Popen
        def spawn(*args,**kwargs):
            child=popen(*args,**kwargs);processes.append(child);return child
        with tempfile.TemporaryFile() as log, patch.object(gate.native,'verify_lease',return_value={}), patch.object(gate.subprocess,'Popen',side_effect=spawn):
            with self.assertRaises(gate.native.controller.ControlTimeout):
                gate.run_owned([sys.executable,'-c','import time; time.sleep(60)'],log,dict(__import__('os').environ),time.monotonic()+0.1)
        self.assertEqual(len(processes),1)
        self.assertIsNotNone(processes[0].poll())


if __name__ == '__main__':
    unittest.main()
