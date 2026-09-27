import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import tempfile
import json
import subprocess
import shutil

spec = importlib.util.spec_from_file_location("qualify", Path(__file__).with_name("qualify.py"))
qualify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qualify)


class LeaseValidation(unittest.TestCase):
    def setUp(self):
        self.uuids = ["GPU-bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
                      "GPU-aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"]
        self.lease = {"wrapper_pid": 101, "child_pid": 102,
                      "requested_uuids": self.uuids, "lock_order": sorted(self.uuids),
                      "lock_files": {u: f"/tmp/memra-gpu-locks/{u}.lock" for u in self.uuids}}
        self.stats = {self.lease["lock_files"][u]: (0, 42, n) for n, u in enumerate(self.uuids, 10)}
        self.rows = "1: FLOCK ADVISORY WRITE 101 00:2a:10 0 EOF\n2: FLOCK ADVISORY WRITE 101 00:2a:11 0 EOF\n"

    def check(self, **overrides):
        args = dict(lease=self.lease, visible=self.uuids, chain={1, 101, 102, 103},
                    lock_rows=self.rows, stats=self.stats, count=2)
        args.update(overrides)
        qualify.validate_lease(**args)

    def test_requested_visibility_order_is_preserved_but_locks_are_sorted(self):
        self.check()

    def test_partial_or_foreign_locks_are_rejected(self):
        for rows in [self.rows.splitlines()[0], self.rows.replace("WRITE 101", "WRITE 999"),
                     self.rows.replace("WRITE", "READ")]:
            with self.subTest(rows=rows), self.assertRaises(RuntimeError):
                self.check(lock_rows=rows)

    def test_extra_cards_or_wrong_visibility_are_rejected(self):
        with self.assertRaises(RuntimeError):
            self.check(count=1)
        with self.assertRaises(RuntimeError):
            self.check(visible=list(reversed(self.uuids)))
        extra = dict(self.stats, **{"/tmp/memra-gpu-locks/GPU-cccc-3333.lock": (0, 42, 12)})
        with self.assertRaises(RuntimeError):
            self.check(stats=extra, lock_rows=self.rows + "3: FLOCK ADVISORY WRITE 101 00:2a:12 0 EOF\n")

    def test_a_lease_cannot_borrow_another_sessions_wrapper(self):
        with self.assertRaises(RuntimeError):
            self.check(chain={1, 102, 103})

    def test_two_gpu_paths_cannot_alias_one_lock_inode(self):
        stats = {path: (0, 42, 10) for path in self.stats}
        with self.assertRaises(RuntimeError):
            self.check(stats=stats)

    def test_short_uuid_aliases_are_rejected(self):
        lease = copy.deepcopy(self.lease)
        lease["requested_uuids"] = ["GPU-bbbbbbbb", "GPU-aaaaaaaa"]
        with self.assertRaisesRegex(RuntimeError, "GPU visibility"):
            self.check(lease=lease, visible=lease["requested_uuids"])

    def test_noncanonical_path_or_unsorted_lock_order_is_rejected(self):
        lease = copy.deepcopy(self.lease)
        lease["lock_order"] = self.uuids
        with self.assertRaises(RuntimeError):
            self.check(lease=lease)
        lease = copy.deepcopy(self.lease)
        lease["lock_files"][self.uuids[0]] = "/tmp/unrelated.lock"
        with self.assertRaises(RuntimeError):
            self.check(lease=lease)


class ArtifactSelection(unittest.TestCase):
    def test_an_unverified_repack_marker_cannot_override_locked_safetensors(self):
        with tempfile.TemporaryDirectory() as root:
            models = Path(root)
            directory = models / "step_fp8"
            directory.mkdir()
            data = directory / "model.safetensors"
            data.write_bytes(b"fixture")
            artifact = {"files": [{"path": data.name, "size": 7, "sha256": qualify.sha(data)}]}
            with patch.object(qualify, "artifacts", return_value={"step_fp8": artifact}):
                qualify.verify_artifact("step_fp8", models)
                (directory / "manifest.json").write_text("{}")
                with self.assertRaisesRegex(RuntimeError, "inventory differs"):
                    qualify.verify_artifact("step_fp8", models)


class BuildReceipts(unittest.TestCase):
    def exercise_build(self, compiler_source="env", mutate_compiler=False):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repo"
            bins = root / "target/release"
            bins.mkdir(parents=True)
            for name in [*qualify.BINS, "memra-server", "focused-test"]:
                (bins / name).write_bytes(b"compiled fixture")
            pinned = Path(directory) / "cuda-13.1/bin/nvcc"
            ambient = Path(directory) / "ambient/nvcc"
            for path, version in [(pinned, "13.1"), (ambient, "13.2")]:
                path.parent.mkdir(parents=True)
                path.write_text(f"#!/bin/sh\necho 'Cuda compilation tools, release {version}'\n")
                path.chmod(0o755)
            out = Path(directory) / "receipts"
            selected = []
            real_run = subprocess.run
            def fake_run(argv, **kwargs):
                if argv[0] != "cargo":
                    return real_run(argv, **kwargs)
                env = kwargs["env"]
                # A missing override lets an ambient compiler win, as build.rs did.
                chosen = env.get("MEMRA_NVCC") or shutil.which("nvcc", path=env["PATH"])
                selected.append(chosen)
                self.assertEqual(chosen, str(pinned.resolve()))
                self.assertNotIn("MEMRA_FAST", env)
                if mutate_compiler and len(selected) == 1:
                    pinned.write_text(pinned.read_text() + "# replaced during compilation\n")
                if "--message-format=json" in argv:
                    message = {"target": {"name": "step_rope_load_gpu"},
                               "executable": str(bins / "focused-test")}
                    kwargs["stdout"].write((json.dumps(message) + "\n").encode())
                return subprocess.CompletedProcess(argv, 0)
            def fake_git(argv, **kwargs):
                return "a" * 40 + "\n" if argv[1] == "rev-parse" else ""
            argv = ["qualify.py", "build", "--out", str(out)]
            env = {"MEMRA_NVCC": str(pinned), "MEMRA_FAST": "1",
                   "PATH": str(ambient.parent)}
            if compiler_source == "cli":
                argv += ["--nvcc", str(pinned)]
                env["MEMRA_NVCC"] = str(ambient)
            with patch.object(qualify, "ROOT", root), \
                 patch.object(qualify.sys, "argv", argv), \
                 patch.dict(qualify.os.environ, env), \
                 patch.object(qualify.signal, "signal"), \
                 patch.object(qualify.subprocess, "run", side_effect=fake_run), \
                 patch.object(qualify.subprocess, "check_output", side_effect=fake_git):
                qualify.main()
            self.assertEqual(json.loads((out / "build-engine.json").read_text())["exit_code"], 0)
            manifest = json.loads((out / "build.json").read_text())
            self.assertEqual(manifest["commit"], "a" * 40)
            self.assertIn("focused", manifest["binaries"])
            self.assertEqual(len(selected), 3)
            self.assertEqual(manifest["compiler"]["path"], str(pinned.resolve()))
            self.assertEqual(manifest["compiler"]["sha256"], qualify.sha(pinned))
            self.assertIn("release 13.1", manifest["compiler"]["version"])

    def test_command_receipts_do_not_collide_with_the_build_manifest(self):
        self.exercise_build()

    def test_explicit_environment_compiler_pin_cannot_fall_back_to_ambient(self):
        self.exercise_build()

    def test_cli_compiler_pin_overrides_environment(self):
        self.exercise_build("cli")

    def test_compiler_replacement_during_build_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "compiler changed during build"):
            self.exercise_build(mutate_compiler=True)

    def test_qualification_build_refuses_an_unpinned_compiler(self):
        with tempfile.TemporaryDirectory() as directory, \
             patch.object(qualify.sys, "argv", ["qualify.py", "build", "--out", directory]), \
             patch.dict(qualify.os.environ, {}, clear=True), \
             patch.object(qualify.signal, "signal"), \
             patch.object(qualify.subprocess, "check_output", side_effect=["a" * 40, ""]), \
             patch.object(qualify, "run") as run:
            with self.assertRaises(SystemExit):
                qualify.main()
            run.assert_not_called()


if __name__ == "__main__":
    unittest.main()
