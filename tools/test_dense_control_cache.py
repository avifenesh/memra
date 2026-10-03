"""CPU cache admission fixtures; actual CUDA compiler evidence is separate."""
import os
import contextlib
import io
import copy
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import dense_control_cache as cache

class DenseCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.entry = self.root / 'entry'
        self.out = self.root / 'out'
        self.entry.mkdir(); self.out.mkdir()
        self.expected = {'schema': cache.SCHEMA, 'inputs': {'header.cuh': {'sha256': 'a'*64}}}
        for name in cache.NAMES:
            shutil.copy2('/bin/true', self.entry / name)
        self.record = {'context': self.expected, 'payloads': cache.payloads(self.entry)}
        self.save()
    def save(self):
        (self.entry / 'manifest.json').write_bytes(cache.canonical(self.record))
    def refused(self):
        with self.assertRaises((cache.Miss, OSError, ValueError)):
            cache.restore(self.entry, self.expected, self.out)
    def test_native_payload_copies_are_independent(self):
        cache.restore(self.entry, self.expected, self.out)
        self.assertEqual(cache.payloads(self.out), self.record['payloads'])
        self.assertNotEqual((self.entry/cache.NAMES[0]).stat().st_ino, (self.out/cache.NAMES[0]).stat().st_ino)
    def test_corrupt_payload(self):
        with (self.entry/cache.NAMES[0]).open('ab') as f: f.write(b'corrupt')
        self.refused()
    def test_changed_header_context(self):
        self.record['context'] = {'schema': cache.SCHEMA, 'inputs': {}}
        self.save(); self.refused()
    def test_missing_context(self):
        del self.record['context']; self.save(); self.refused()
    def test_missing_payload(self):
        (self.entry/cache.NAMES[0]).unlink(); self.refused()
    def test_unexpected_member(self):
        (self.entry/'extra').write_text('extra'); self.refused()
    def test_payload_symlink(self):
        p = self.entry/cache.NAMES[0]; p.unlink(); p.symlink_to('/bin/true'); self.refused()
    def test_payload_mode_change(self):
        (self.entry/cache.NAMES[0]).chmod(0o644); self.refused()
    def test_truncated_manifest(self):
        (self.entry/'manifest.json').write_text('{'); self.refused()
    def test_missing_payload_identity(self):
        del self.record['payloads'][cache.NAMES[0]]; self.save(); self.refused()
    def test_non_elf_payload(self):
        p = self.entry/cache.NAMES[0]; p.write_text('wrong'); p.chmod(0o755)
        with self.assertRaises(cache.Miss): cache.payloads(self.entry)
    def test_nested_inventory_mutations(self):
        tree = self.root/'tree'; tree.mkdir(); (tree/'header.cuh').write_text('old')
        before = cache.inventory([tree]); (tree/'header.cuh').write_text('new')
        self.assertNotEqual(before, cache.inventory([tree]))
        before = cache.inventory([tree]); (tree/'optional.cuh').write_text('present')
        self.assertNotEqual(before, cache.inventory([tree]))
    def test_link_target_identity(self):
        tree = self.root/'tree'; tree.mkdir(); (tree/'first').write_text('first'); (tree/'second').write_text('second'); (tree/'alias').symlink_to('first')
        before = cache.inventory([tree]); (tree/'alias').unlink(); (tree/'alias').symlink_to('second')
        self.assertNotEqual(before, cache.inventory([tree]))
    def test_every_ambient_override_refuses_before_inventory(self):
        for name in cache.OVERRIDES:
            with self.subTest(name=name), patch.object(cache, 'inventory', side_effect=AssertionError('must refuse first')):
                with self.assertRaises(cache.Miss): cache.context(self.root, Path('/nvcc'), [], {name:'override'})
    def test_recipe_retains_all_six_and_flags(self):
        self.assertEqual(len(cache.NAMES), 6)
        for name in cache.NAMES:
            command = cache.recipe(Path('/cuda/bin/nvcc'), name, self.out)
            self.assertEqual(command[1:1+len(cache.FLAGS)], list(cache.FLAGS))
            self.assertIn('tools/'+name+'.cu', command)
            self.assertEqual(command[-2:], ['-o',str(self.out/name)])

    def test_manifest_symlink(self):
        manifest = self.entry/'manifest.json'
        outside = self.root/'outside.json'
        manifest.rename(outside); manifest.symlink_to(outside)
        self.refused()
    def test_entry_symlink(self):
        real = self.root/'real'; self.entry.rename(real); self.entry.symlink_to(real, target_is_directory=True)
        self.refused()
    def test_output_symlink_does_not_modify_external_file(self):
        outside = self.root/'external'; outside.write_text('keep')
        (self.out/cache.NAMES[0]).symlink_to(outside)
        cache.restore(self.entry, self.expected, self.out)
        self.assertEqual(outside.read_text(), 'keep')
        self.assertFalse((self.out/cache.NAMES[0]).is_symlink())
    def test_manifest_wrong_type(self):
        (self.entry/'manifest.json').write_text('[]'); self.refused()
    def test_elf_wrong_machine(self):
        p = self.entry/cache.NAMES[0]; data = bytearray(p.read_bytes()); data[18:20] = b'\xb7\x00'; p.write_bytes(data)
        with self.assertRaises(cache.Miss): cache.payloads(self.entry)
    def test_optional_expansion_changes_source_key(self):
        def output(command, **kwargs):
            if '-M' not in command: return self.expansion
            return b'dense-control: input.cuh\n'
        (self.root/'input.cuh').write_text('header')
        with patch.object(cache.subprocess, 'check_output', side_effect=output), patch.object(cache, 'phase_commands', return_value=([['gcc','-E','input.cuh']], {}, {})):
            self.expansion = b'optional=false'; before = cache.preprocess(self.root, Path('/cuda/bin/nvcc'))
            self.expansion = b'optional=true'; after = cache.preprocess(self.root, Path('/cuda/bin/nvcc'))
            self.assertNotEqual(before, after)
    def test_unrelated_rust_and_docs_do_not_enter_source_key(self):
        (self.root/'input.cuh').write_text('header')
        def output(command, **kwargs):
            return b'expanded' if '-M' not in command else b'dense-control: input.cuh\n'
        with patch.object(cache.subprocess, 'check_output', side_effect=output), patch.object(cache, 'phase_commands', return_value=([['gcc','-E','input.cuh']], {}, {})):
            before = cache.preprocess(self.root, Path('/cuda/bin/nvcc'))
            (self.root/'unrelated.rs').write_text('change'); (self.root/'README.md').write_text('change')
            self.assertEqual(before, cache.preprocess(self.root, Path('/cuda/bin/nvcc')))
    def test_transitive_header_byte_change_changes_source_key(self):
        header = self.root/'input.cuh'; header.write_text('header')
        def output(command, **kwargs):
            return b'expanded' if '-M' not in command else b'dense-control: input.cuh\n'
        with patch.object(cache.subprocess, 'check_output', side_effect=output), patch.object(cache, 'phase_commands', return_value=([['gcc','-E','input.cuh']], {}, {})):
            before = cache.preprocess(self.root, Path('/cuda/bin/nvcc')); header.write_text('new header')
            self.assertNotEqual(before, cache.preprocess(self.root, Path('/cuda/bin/nvcc')))

    def test_unrelated_ci_metadata_does_not_change_environment_identity(self):
        before = cache.compiler_environment({'PATH':'/usr/bin', 'GITHUB_RUN_ID':'one'})
        after = cache.compiler_environment({'PATH':'/usr/bin', 'GITHUB_RUN_ID':'two'})
        self.assertEqual(before, after)
    def test_actual_compiler_environment_changes_identity(self):
        self.assertNotEqual(cache.compiler_environment({'PATH':'/usr/bin'}), cache.compiler_environment({'PATH':'/usr/bin:/opt'}))
    def test_unknown_compiler_environment_refuses(self):
        for name in ('NVCC_SOMETHING', 'LD_AUDIT', 'GCC_NEW_OVERRIDE'):
            with self.subTest(name=name), self.assertRaises(cache.Miss): cache.compiler_environment({name:'present'})
    def test_profile_mutation_refuses(self):
        profile = self.root/'nvcc.profile'; profile.write_text(cache.PROFILE)
        cache.admit_profile(profile)
        profile.write_text(cache.PROFILE+'\nCICC_PATH = /other\n')
        with self.assertRaises(cache.Miss): cache.admit_profile(profile)

    def test_same_symlink_target_content_change_changes_identity(self):
        tree = self.root/'tree'; tree.mkdir(); target = tree/'target'; target.write_text('before'); (tree/'library.so').symlink_to(target)
        before = cache.inventory([tree]); target.write_text('after')
        self.assertNotEqual(before, cache.inventory([tree]))
    def test_directory_symlink_target_content_change_changes_identity(self):
        tree = self.root/'tree'; tree.mkdir(); target = tree/'target'; target.mkdir(); data = target/'device-tool'; data.write_text('before'); (tree/'bin').symlink_to(target, target_is_directory=True)
        before = cache.inventory([tree]); data.write_text('after')
        self.assertNotEqual(before, cache.inventory([tree]))
    def test_escaping_symlink_target_refuses(self):
        tree = self.root/'tree'; tree.mkdir(); outside = self.root/'outside'; outside.write_text('external'); (tree/'library.so').symlink_to(outside)
        with self.assertRaises(cache.Miss): cache.inventory([tree])
    def test_unresolved_symlink_refuses(self):
        tree = self.root/'tree'; tree.mkdir(); (tree/'library.so').symlink_to('missing')
        with self.assertRaises(cache.Miss): cache.inventory([tree])
    def test_symlink_cycle_refuses(self):
        tree = self.root/'tree'; tree.mkdir(); (tree/'a').symlink_to('b'); (tree/'b').symlink_to('a')
        with self.assertRaises(cache.Miss): cache.inventory([tree])

    def stub(self):
        cuda = self.root/'cuda'; (cuda/'bin').mkdir(parents=True); (cuda/'lib64/stubs').mkdir(parents=True)
        nvcc = cuda/'bin/nvcc'; shutil.copy2('/bin/true', nvcc)
        stub = cuda/'lib64/stubs/libcuda.so'; shutil.copy2('/bin/true', stub)
        (self.out/'libcuda.so.1').symlink_to(stub)
        return nvcc
    def test_exact_owned_stub_prefix_admits(self):
        nvcc=self.stub()
        with patch.object(cache,'inventory',return_value={}),patch.object(cache,'preprocess',return_value={}):
            observed=cache.context(self.root,nvcc,[],{'LD_LIBRARY_PATH':str(self.out)+':'},self.out)
            self.assertEqual(observed['driver_stub'],cache.identity((self.out/'libcuda.so.1').resolve()))
    def test_inherited_loader_prefix_refuses(self):
        nvcc=self.stub()
        with self.assertRaises(cache.Miss):cache.context(self.root,nvcc,[],{'LD_LIBRARY_PATH':str(self.out)+':/ambient'},self.out)
    def test_unknown_library_in_owned_prefix_refuses(self):
        nvcc=self.stub(); (self.out/'libextra.so').write_text('unknown')
        with self.assertRaises(cache.Miss):cache.context(self.root,nvcc,[],{'LD_LIBRARY_PATH':str(self.out)+':'},self.out)
    def test_changed_owned_stub_symlink_refuses(self):
        nvcc=self.stub(); (self.out/'libcuda.so.1').unlink(); (self.out/'libcuda.so.1').symlink_to('/bin/true')
        with self.assertRaises(cache.Miss):cache.context(self.root,nvcc,[],{'LD_LIBRARY_PATH':str(self.out)+':'},self.out)
    def test_manifest_fifo_refuses_without_read(self):
        manifest=self.entry/'manifest.json'; manifest.unlink(); os.mkfifo(manifest)
        self.refused()
    def test_manifest_size_ceiling_refuses(self):
        with (self.entry/'manifest.json').open('wb') as writer:writer.truncate(64*1024*1024+1)
        self.refused()
    def test_payload_size_ceiling_refuses(self):
        with (self.entry/cache.NAMES[0]).open('wb') as writer:writer.truncate(96*1024*1024+1)
        self.refused()
    def test_noncanonical_compiler_miss_preserves_original_argv(self):
        nvcc=self.stub(); alias=self.root/'nvcc-alias';alias.symlink_to(nvcc)
        commands=[]
        def compile(command,**kwargs):
            commands.append(command);shutil.copy2('/bin/true',Path(command[-1]))
        with patch.object(cache.subprocess,'run',side_effect=compile),patch.object(cache,'visible_roots',side_effect=AssertionError('must miss first')),contextlib.redirect_stdout(io.StringIO()):
            cache.run(self.root,str(alias),self.out,self.root/'cache')
        self.assertEqual(len(commands),6)
        self.assertTrue(all(command[0]==str(alias) for command in commands))
    def test_assembler_external_input_refuses(self):
        (self.root/'input.cuh').write_text('header')
        with patch.object(cache,'phase_commands',return_value=([['gcc','-E','input.cuh']],{},{})),patch.object(cache.subprocess,'check_output',return_value=b'asm(".incbin external");'):
            with self.assertRaises(cache.Miss):cache.preprocess(self.root,Path('/cuda/bin/nvcc'))

    def test_unknown_dryrun_phase_refuses(self):
        result=type('Result',(),{'stderr':'#$ PATH=/usr/bin\n#$ LD_LIBRARY_PATH=\n#$ unexpected-compiler input.cu\n'})()
        with patch.object(cache.subprocess,'run',return_value=result):
            with self.assertRaises(cache.Miss):cache.phase_commands(self.root,Path('/cuda/bin/nvcc'),cache.NAMES[0])
    def test_incomplete_dryrun_phase_manifest_refuses(self):
        result=type('Result',(),{'stderr':'#$ PATH=/usr/bin\n#$ LD_LIBRARY_PATH=\n'})()
        with patch.object(cache.subprocess,'run',return_value=result):
            with self.assertRaises(cache.Miss):cache.phase_commands(self.root,Path('/cuda/bin/nvcc'),cache.NAMES[0])
    def test_escaped_dependency_spelling_refuses(self):
        def output(command,**kwargs):
            return b'expanded' if '-M' not in command else b'dense-control: input\\ name.cuh\n'
        with patch.object(cache,'phase_commands',return_value=([['gcc','-E','input.cuh']],{},{})),patch.object(cache.subprocess,'check_output',side_effect=output):
            with self.assertRaises(cache.Miss):cache.preprocess(self.root,Path('/cuda/bin/nvcc'))

    def test_boolean_identity_replaced_by_integer_refuses(self):
        tree=self.root/'inputs';tree.mkdir();self.expected['inputs']=cache.inventory([tree]);self.record=copy.deepcopy(self.record)
        self.record['context']['inputs'][str(tree)]['tree']=1
        self.save();self.refused()
    def test_integer_payload_identity_replaced_by_float_refuses(self):
        self.record['payloads'][cache.NAMES[0]]['bytes']=float(self.record['payloads'][cache.NAMES[0]]['bytes'])
        self.save();self.refused()
    def test_every_empty_ambient_override_refuses_first(self):
        for name in cache.OVERRIDES:
            with self.subTest(name=name),patch.object(cache,'inventory',side_effect=AssertionError('must refuse first')):
                with self.assertRaises(cache.Miss):cache.context(self.root,Path('/cuda/bin/nvcc'),[],{name:''})
    def test_duplicate_manifest_key_refuses(self):
        data=cache.canonical(self.record).decode()
        (self.entry/'manifest.json').write_text('{"context":{},'+data[1:])
        self.refused()
    def test_nonfinite_manifest_number_refuses(self):
        self.record['payloads'][cache.NAMES[0]]['bytes']=float('nan');self.save();self.refused()

    def fallback_script(self, failure=None):
        repo=self.root/'repo';(repo/'tools').mkdir(parents=True)
        shutil.copy2(cache.__file__,repo/'tools/dense_control_cache.py')
        shutil.copy2(Path(cache.__file__).with_name('test-dsv4-dense-control-policy.sh'),repo/'tools/test-dsv4-dense-control-policy.sh')
        cuda=self.root/'fallback-cuda';(cuda/'bin').mkdir(parents=True);(cuda/'lib64/stubs').mkdir(parents=True)
        shutil.copy2('/bin/true',cuda/'lib64/stubs/libcuda.so')
        counter=self.root/'compiler-count.log';compiler=cuda/'bin/nvcc'
        body='#!/usr/bin/env python3\nimport sys\nfrom pathlib import Path\n'
        body+='with Path('+repr(str(counter))+').open("a") as writer:writer.write("compile\\n")\n'
        if failure is not None:body+='raise SystemExit('+str(failure)+')\n'
        else:
            body+='out=Path(sys.argv[sys.argv.index("-o")+1])\n'
            body+='out.write_text('+repr('#!/bin/sh\n[ "$1" = --check-controls ] || exit 2\necho PASS synthetic_cpu_control\n')+')\n'
            body+='out.chmod(0o755)\n'
        compiler.write_text(body);compiler.chmod(0o755)
        env=dict(os.environ,MEMRA_NVCC=str(compiler),CARGO_TARGET_DIR=str(self.root/'target'))
        return repo,counter,env
    def test_real_uncached_script_executables_keep_all18_controls(self):
        repo,counter,env=self.fallback_script()
        result=cache.subprocess.run(['bash','tools/test-dsv4-dense-control-policy.sh'],cwd=repo,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertEqual(len(counter.read_text().splitlines()),6)
        self.assertEqual(sum(line=='PASS synthetic_cpu_control' for line in result.stdout.splitlines()),18)
        self.assertIn('"status": "uncached"',result.stdout)
        self.assertFalse((self.root/'target/dsv4-control-cache').exists())
    def test_real_compile_failure_runs_once_and_propagates_exact_exit(self):
        repo,counter,env=self.fallback_script(failure=17)
        result=cache.subprocess.run(['bash','tools/test-dsv4-dense-control-policy.sh'],cwd=repo,env=env,text=True,capture_output=True)
        self.assertEqual(result.returncode,17,result.stdout+result.stderr)
        self.assertEqual(counter.read_text().splitlines(),['compile'])
        self.assertNotIn('PASS synthetic_cpu_control',result.stdout)

    def test_complete_preprocessing_phase_permutation_keeps_identity(self):
        (self.root/'input.cuh').write_text('header')
        commands=[['gcc','-E','-DHOST'],['gcc','-E','-DFEATURE'],['gcc','-E','-DGENERIC']]
        def output(command,**kwargs):
            return cache.canonical(command) if '-M' not in command else b'dense-control: input.cuh\n'
        with patch.object(cache.subprocess,'check_output',side_effect=output):
            with patch.object(cache,'phase_commands',return_value=(commands,{},{})):
                before=cache.preprocess(self.root,Path('/cuda/bin/nvcc'))
            with patch.object(cache,'phase_commands',return_value=(list(reversed(commands)),{},{})):
                after=cache.preprocess(self.root,Path('/cuda/bin/nvcc'))
        self.assertEqual(before,after)
        self.assertTrue(all(len(record['phases'])==3 for record in before.values()))

    def owned_entry(self,storage,generation):
        record=copy.deepcopy(self.record);record['context']['generation']=generation
        key=cache.digest(cache.canonical(record['context']));entry=storage/key;entry.mkdir()
        (entry/'manifest.json').write_bytes(cache.canonical(record))
        return entry
    def test_retention_keeps_current_and_deletes_only_owned_entries(self):
        storage=self.root/'cache';storage.mkdir()
        current=self.owned_entry(storage,0);old=[self.owned_entry(storage,i) for i in (1,2)]
        foreign=storage/'foreign';foreign.mkdir();(foreign/'data').write_text('keep')
        self.assertEqual(cache.prune_cache(storage,current.name),2)
        self.assertTrue(current.is_dir());self.assertEqual((foreign/'data').read_text(),'keep')
        self.assertTrue(all(not entry.exists() for entry in old))
    def test_retention_preserves_foreign_schema_and_symlink_siblings(self):
        storage=self.root/'cache';storage.mkdir();current=self.owned_entry(storage,0)
        foreign=storage/('a'*64);foreign.mkdir();(foreign/'manifest.json').write_text('{"context":{"schema":"foreign"}}')
        alias=storage/('b'*64);alias.symlink_to(self.entry,target_is_directory=True)
        self.assertEqual(cache.prune_cache(storage,current.name),0)
        self.assertTrue(foreign.is_dir());self.assertTrue(alias.is_symlink());self.assertTrue(self.entry.is_dir())
    def test_transport_upload_uses_semantic_key_and_skips_unchanged_restore(self):
        workflow=Path(cache.__file__).parents[1]/'.github/workflows/ci.yml'
        text=workflow.read_text();save=text.split('- uses: actions/cache/save@',1)[1].split('  # ── clippy:',1)[0]
        self.assertIn('steps.dense_controls.outputs.cache-key',save)
        self.assertIn('steps.dense_cache.outputs.cache-matched-key !=',save)
        self.assertNotIn('github.run_id',save)
        self.assertIn("records = [json.loads(line)",text)

if __name__ == '__main__': unittest.main(verbosity=2)
