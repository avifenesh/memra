#!/usr/bin/env python3
"""Actual small Cargo producer/receiver witness. No Memra/native qualification."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).absolute().parent.parent
spec = importlib.util.spec_from_file_location('package_source', ROOT / 'tools/package_source_identity.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class PackageSourceIntegration(unittest.TestCase):
    def test_normal_cargo_source_and_input_contract(self):
        self.cargo_contract()

    def test_coherent_midcompiler_source_replacement(self):
        self.cargo_contract(midcompiler=True)

    def test_root_codegen_context(self):
        self.cargo_contract(codegen=True)

    def test_normal_dependency_without_generated_directory(self):
        self.cargo_contract(pure=True)

    def cargo_contract(self, midcompiler=False, codegen=False, pure=False):
        env = dict(os.environ)
        for key in ('RUSTC_WRAPPER', 'RUSTC_WORKSPACE_WRAPPER', 'CARGO_BUILD_RUSTC_WRAPPER', 'CARGO_BUILD_RUSTC_WORKSPACE_WRAPPER'):
            env.pop(key, None)
        def run(args, cwd, *, fail=None, context=None):
            result = subprocess.run(args, cwd=cwd, env=env if context is None else context,
                                    capture_output=True, text=True)
            if fail:
                self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(fail, result.stderr)
            else:
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            return result
        with tempfile.TemporaryDirectory(prefix='memra-package-input-test-') as temporary:
            base = Path(temporary)
            dep, app = base / 'dependency', base / 'package'
            for path in (dep / 'src', app / 'src'):
                path.mkdir(parents=True)
            (dep / 'Cargo.toml').write_text('[package]\nname="source-witness-dep"\nversion="0.0.1"\nedition="2024"\n')
            (dep / 'src/lib.rs').write_text('include!(concat!(env!("OUT_DIR"),"/generated.rs")); pub fn token()-> &\'static str{"baseline"}\n')
            (dep / 'build.rs').write_text('fn main(){std::fs::write(std::path::Path::new(&std::env::var("OUT_DIR").unwrap()).join("generated.rs"),r#"pub fn generated()-> &\'static str {"generated"}"#).unwrap();}\n')
            if pure:
                (dep / 'build.rs').unlink()
                (dep / 'src/lib.rs').write_text("pub fn token()-> &'static str{\"baseline\"}\n")
            run(['cargo', 'package', '--offline', '--no-verify', '--target-dir', str(base / 'archive')], dep)
            archive = base / 'archive/package/source-witness-dep-0.0.1.crate'
            vendor = app / 'vendor/source-witness-dep-0.0.1'
            vendor.mkdir(parents=True)
            files = {}
            with tarfile.open(archive) as source:
                for row in source.getmembers():
                    self.assertTrue(row.isfile())
                    name = Path(*Path(row.name).parts[1:]).as_posix()
                    m.relative_name(name)
                    body = source.extractfile(row).read()
                    path = vendor / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(body)
                    path.chmod(0o755 if row.mode & 0o100 else 0o644)
                    files[name] = hashlib.sha256(body).hexdigest()
            (vendor / '.cargo-checksum.json').write_text(json.dumps({'files': files, 'package': hashlib.sha256(archive.read_bytes()).hexdigest()}))
            (app / 'Cargo.toml').write_text('[package]\nname="memra-server"\nversion="0.0.1"\nedition="2024"\n[dependencies]\nsource-witness-dep="=0.0.1"\n')
            # Use the actual shared server producer and ownership implementation.
            shutil.copy2(ROOT / 'crates/memra-server/build.rs', app / 'build.rs')
            shutil.copy2(ROOT / 'crates/memra-server/src/build_id.rs', app / 'src/build_id.rs')
            support = app / 'build-support'
            support.mkdir()
            for name in ('package_source_identity.py', 'package_source_rustc.py'):
                shutil.copy2(ROOT / 'tools' / name, support / name)
            (app / '.cargo').mkdir()
            (app / '.cargo/config.toml').write_text('[source.crates-io]\nreplace-with="memra-package-source"\n[source.memra-package-source]\ndirectory="vendor"\n[build]\nrustc-wrapper="build-support/package_source_rustc.py"\n')
            (app / 'src/lib.rs').write_text('pub fn identity()-> &\'static str{env!("MEMRA_BUILD_ID")} pub fn source()-> &\'static str{env!("MEMRA_BUILD_ID_SRC")} #[cfg(not(unadmitted))] pub fn value()-> &\'static str{source_witness_dep::token()} #[cfg(unadmitted)] pub fn value()-> &\'static str{"wrong-cfg-body"}\n')
            (app / 'src/main.rs').write_text('fn main(){println!("{} {} {} {}",env!("MEMRA_BUILD_ID"),memra_server::identity(),memra_server::source(),memra_server::value());}\n')
            run(['cargo', 'generate-lockfile', '--offline'], app)
            output, expectations = base / 'output', base / 'expectations'
            output.mkdir(); expectations.mkdir()
            saved = dict(os.environ)
            try:
                os.environ.clear(); os.environ.update(env)
                snapshot = m.cargo_graph(app / 'Cargo.toml', 'x86_64-unknown-linux-gnu', [])
                keys = snapshot['payload']['packages']
                plan = {'env': {key: ['OUT_DIR'] for key in keys},
                        'generated': {key: [] if pure or key == snapshot['payload']['entry'] else ['generated.rs'] for key in keys},
                        'cfgs': {key: [] for key in keys},
                        'codegen': {'embed-bitcode': ['no'], 'debuginfo': ['0', '2']},
                        'targets': ['memra_server'], 'supplementary': {}}
                declarations = base / 'declarations.json'
                declarations.write_text(json.dumps(plan))
                cap = m.prepare_package(app / 'Cargo.toml', 'x86_64-unknown-linux-gnu', [], declarations, output, expectations)
            finally:
                os.environ.clear(); os.environ.update(saved)
            command = ['cargo', 'build', '--offline', '--locked', '--target-dir', str(output)]
            if midcompiler:
                # An owned real compiler proxy changes the source and coherently
                # reseals every capsule layer after rustc returns. Freshness must
                # retain the captured capsule, not accept any later valid one.
                proxy = base / 'owned-compiler.py'
                mutation = base / 'mutation.json'
                proxy.write_text("#!/usr/bin/python3 -I\n" +
                    "import hashlib,json,os,re,stat,subprocess,sys\nfrom pathlib import Path\n" +
                    "compiler=" + repr(cap['compiler']['path']) + "\n" +
                    "pairs=re.findall(r'--jobserver-(?:auth|fds)=(\\d+),(\\d+)',os.environ.get('CARGO_MAKEFLAGS',''))\n" +
                    "assert len(set(pairs))<=1; descriptors=tuple(int(x) for x in pairs[0]) if pairs else ()\n" +
                    "assert not descriptors or (len(set(descriptors))==2 and all(2<x<65536 and stat.S_ISFIFO(os.fstat(x).st_mode) for x in descriptors))\n" +
                    "code=subprocess.run([compiler,*sys.argv[1:]],check=False,pass_fds=descriptors).returncode\n" +
                    "if code==0 and '--crate-name' in sys.argv and sys.argv[sys.argv.index('--crate-name')+1]=='memra_server':\n" +
                    " root=Path(" + repr(str(app)) + "); source=root/'src/lib.rs'; info=source.stat()\n" +
                    " source.write_bytes(source.read_bytes().replace(b'wrong-cfg-body',b'drift-cfg-body')); os.utime(source,ns=(info.st_atime_ns,info.st_mtime_ns))\n" +
                    " cpath=root/'.memra-package-source.json'; c=json.loads(cpath.read_bytes()); key=c['snapshot']['payload']['entry']\n" +
                    " canonical=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()\n" +
                    " digest=lambda x:hashlib.sha256(canonical(x)).hexdigest()\n" +
                    " c['snapshot']['payload']['packages'][key]['files']['src/lib.rs']['sha256']=hashlib.sha256(source.read_bytes()).hexdigest()\n" +
                    " c['snapshot']['sha256']=digest(c['snapshot']['payload'])\n" +
                    " c['source_seal']=digest({'source':c['snapshot']['payload'],'recipe':c['recipe'],'supplementary':c['supplementary'],'ambient':c['ambient']})\n" +
                    " c['seal']=digest({k:v for k,v in c.items() if k!='seal'}); cpath.write_bytes(canonical(c)+b'\\n')\n" +
                    " Path(" + repr(str(mutation)) + ").write_text(json.dumps({'mtime_preserved':source.stat().st_mtime_ns==info.st_mtime_ns,'new_seal':c['seal']}))\n" +
                    "sys.exit(code)\n")
                proxy.chmod(0o755)
                cap['compiler'] = {'path': str(proxy), 'file': m.regular(base, proxy.name)}
                cap['seal'] = m.digest({k: v for k, v in cap.items() if k != 'seal'})
                (app / m.RESERVED).write_bytes(m.canonical(cap) + b'\n')
                context = dict(env); context['RUSTC'] = str(proxy)
                result = run(command, app, context=context, fail='package capsule/source changed during compiler execution')
                self.assertNotIn('jobserver', result.stderr)
                print(json.dumps({'compiler_boundary_stderr': result.stderr, 'jobserver_warning': False}))
                changed = json.loads(mutation.read_bytes())
                self.assertTrue(changed['mtime_preserved'])
                self.assertNotEqual(changed['new_seal'], cap['seal'])
                self.assertNotEqual(m.package_capsule(app), cap)
                self.assertFalse(list(output.glob('identity-*.json')))
                self.assertFalse(list(expectations.glob('identity-*.json')))
                return
            run(command, app)
            binary = output / 'debug/memra-server'
            baseline = run([str(binary)], app).stdout.strip().split()
            self.assertEqual(baseline[0], baseline[1])
            self.assertEqual(baseline[2:], ['package-source-v1', 'baseline'])
            run(command, app)
            self.assertEqual(run([str(binary)], app).stdout.strip().split(), baseline)
            if pure:
                records=[json.loads(path.read_bytes()) for path in output.rglob('*.memra-source.json')]
                dependent=[row for row in records if row['package'] != snapshot['payload']['entry']]
                self.assertTrue(dependent)
                for row in dependent:
                    self.assertEqual(row['input_tuple']['own_generated'], {})
                    self.assertEqual(row['input_tuple']['nonidentity_env']['OUT_DIR'], {'present': False, 'sha256': None})
                print(json.dumps({'normalized_dependency_build_rs': False, 'OUT_DIR_present': False, 'ordinary_cached_LIB_BIN_same_identity': True}))
                return
            rustc = ['cargo', 'rustc', '--offline', '--locked', '--lib', '--target-dir', str(output), '--']
            if codegen:
                run(rustc + ['-C', 'debuginfo=0'], app, fail='duplicate nonidentity compiler codegen choice')
                self.assertEqual(run([str(binary)], app).stdout.strip().split(), baseline)
                print(json.dumps({'normal_LIB_BIN_same_identity': True, 'root_only_admitted_codegen_change': 'explicitly refused duplicate choice before compilation'}))
                return
            run(rustc + ['--cfg=unadmitted'], app, fail='unknown actual compiler cfg')
            run(rustc + ['--cfg', 'unadmitted'], app, fail='unknown actual compiler cfg')
            run(rustc + ['--cfg=unadmitted', '--cfg', 'unadmitted'], app, fail='duplicate compiler cfg')
            run(rustc + ['--test'], app, fail='unadmitted implicit compiler test mode')
            # No same-mtime source, new member, executable mode or mixed source
            # may reuse the declared package identity on a cached Cargo build.
            source = vendor / 'src/lib.rs'
            original, info = source.read_bytes(), source.stat()
            source.write_bytes(original.replace(b'baseline', b'tampered'))
            os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
            run(command, app, fail='fresh package source/mode/membership differs')
            source.write_bytes(original); os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns))
            source.chmod(0o755)
            run(command, app, fail='fresh package source/mode/membership differs')
            source.chmod(0o644)
            extra = vendor / 'src/extra.rs'; extra.write_text('pub fn extra(){}\n')
            run(command, app, fail='fresh package source/mode/membership differs'); extra.unlink()
            run(command, app)
            alternate = base / 'alternate'; shutil.copytree(app / 'vendor', alternate)
            run(command + ['--config', 'source.memra-package-source.directory=' + json.dumps(str(alternate))], app,
                fail='actual compiler manifest/source root is unbound')
            no_wrapper = dict(env); no_wrapper['RUSTC_WRAPPER'] = ''
            run(['cargo', 'build', '--offline', '--locked', '--target-dir', str(base / 'degraded')], app, context=no_wrapper)
            self.assertEqual(run([str(base / 'degraded/debug/memra-server')], app).stdout.split()[2], 'degraded')
            # Atomically published custody cannot be replaced or follow a link.
            leaf = expectations / 'atomic.json'; m.immutable_json(leaf, {'current': True})
            with self.assertRaises(m.Refused): m.immutable_json(leaf, {'current': False})
            link = expectations / 'alias.json'; link.symlink_to(leaf)
            with self.assertRaises((m.Refused, OSError)): m.immutable_json(link, {'current': True})
            self.assertEqual(cap['schema'], m.PACKAGE_SCHEMA)


if __name__ == '__main__':
    unittest.main()
