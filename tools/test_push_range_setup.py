#!/usr/bin/env python3
"""Setup-failure containment for the push-range fixture; no real source mutation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GIT = shutil.which("git")


def snapshot(root):
    return {str(p.relative_to(root)): (p.stat().st_mode & 0o777,
            hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*") if p.is_file()}


class SetupContainment(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="push-range-containment-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.outside = self.root / "outside"
        self.scratch = self.root / "scratch"
        self.bin = self.root / "bin"
        for p in (self.outside, self.scratch, self.bin):
            p.mkdir()
        for name in ("docs/keep.md", "crates/keep.rs", "sentinel"):
            p = self.outside / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("outside fixture; must remain unchanged\n")
        for name in ("test_push_range.sh", "push-range.sh", "hooks/pre-push"):
            p = self.outside / "tools" / name
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / "tools" / name, p)
        self.env = {k: v for k, v in os.environ.items()
                    if not k.startswith(("GIT_", "MEMRA_")) and k != "BASH_ENV"}
        def git(*args):
            subprocess.run([GIT, "-C", str(self.outside), *args], env=self.env,
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        git("init", "-q", "-b", "sentinel")
        git("config", "user.name", "Outside sentinel")
        git("config", "user.email", "outside@example.invalid")
        git("config", "core.hooksPath", "/dev/null")
        git("add", ".")
        git("commit", "-qm", "outside sentinel")
        git("remote", "add", "origin", "https://example.invalid/never-contact")
        # Keep failure injection inside this throwaway tree. The git shim also
        # refuses any unexpected mutation outside the owned scratch directories.
        shim = self.bin / "shim.py"
        shim.write_text(f"#!{sys.executable}\n" + r'''
import json, os, pathlib, subprocess, sys
kind = pathlib.Path(sys.argv[0]).name
args = sys.argv[1:]
mode = os.environ['INJECT']
base = pathlib.Path(os.environ['SCRATCH']).resolve()
outside = pathlib.Path(os.environ['OUTSIDE']).resolve()
cwd = pathlib.Path.cwd().resolve()
with open(os.environ['CALLS'], 'a') as stream:
    stream.write(json.dumps({'kind': kind, 'args': args, 'cwd': str(cwd)}) + '\n')
if kind == 'mktemp':
    if mode == 'mktemp': sys.exit(73)
    if mode == 'outside-result': print(outside); sys.exit(0)
    result = subprocess.run([os.environ['REAL_MKTEMP'], *args], capture_output=True, text=True)
    if result.returncode == 0 and mode == 'results':
        pathlib.Path(result.stdout.strip(), 'results').mkdir()
    print(result.stdout, end=''); sys.exit(result.returncode)
if kind == 'mkdir':
    if mode == 'mkdir': sys.exit(73)
    os.execv(os.environ['REAL_MKDIR'], [os.environ['REAL_MKDIR'], *args])
if kind == 'rm':
    if not cwd.is_relative_to(base): sys.exit(97)
    targets = [pathlib.Path(x).resolve() for x in args if not x.startswith('-')]
    if any(not p.is_relative_to(base) for p in targets): sys.exit(97)
    os.execv(os.environ['REAL_RM'], [os.environ['REAL_RM'], *args])
if kind == 'git':
    if 'init' in args:
        target = pathlib.Path(args[-1]).resolve()
        if not target.is_relative_to(base): sys.exit(97)
        if (mode == 'bare-init' and '--bare' in args) or (mode == 'work-init' and '--bare' not in args):
            sys.exit(73)
        if mode == 'missing-git-dir' and '--bare' not in args:
            target.mkdir(); sys.exit(0)
    elif not cwd.is_relative_to(base) and args != ['rev-parse', '--local-env-vars']:
        sys.exit(97)
    rc = subprocess.run([os.environ['REAL_GIT'], *args]).returncode
    if mode == 'owner-marker' and 'init' in args:
        pathlib.Path(args[-1]).parent.joinpath('.fixture-owner').write_text('foreign\n')
    sys.exit(rc)
sys.exit(98)
''')
        shim.chmod(0o755)
        for name in ("git", "mkdir", "mktemp", "rm"):
            (self.bin / name).symlink_to(shim)
        bash_env = self.root / "bash-env"
        bash_env.write_text(r'''
cd() {
    case "${*: -1}" in
        */first-push-clean)
            [ "$INJECT" = cd ] && return 73
            [ "$INJECT" = wrong-cwd ] && return 0
            ;;
    esac
    builtin cd "$@"
}
git() {
    "$SHIM_GIT" "$@"
    local rc=$?
    if [ "$INJECT" = destructive-cwd ] && [ "$*" = "checkout -q --orphan lane/orphan-root" ]; then
        builtin cd "$OUTSIDE"
    fi
    return "$rc"
}
''')
        self.env.update(PATH=str(self.bin) + os.pathsep + self.env["PATH"],
            TMPDIR=str(self.scratch), SCRATCH=str(self.scratch), OUTSIDE=str(self.outside),
            CALLS=str(self.root / "calls.jsonl"), REAL_GIT=GIT,
            REAL_MKTEMP=shutil.which("mktemp"), REAL_MKDIR=shutil.which("mkdir"),
            REAL_RM=shutil.which("rm"),
            SHIM_GIT=str(self.bin / "git"), BASH_ENV=str(bash_env))

    def run_case(self, mode, expected):
        before = snapshot(self.outside)
        Path(self.env["CALLS"]).write_text("")
        result = subprocess.run(["bash", "tools/test_push_range.sh"], cwd=self.outside,
            env={**self.env, "INJECT": mode}, capture_output=True, text=True, timeout=120)
        self.assertEqual(snapshot(self.outside), before,
                         "outside source, sentinel, refs, config or index changed")
        calls = [json.loads(line) for line in Path(self.env["CALLS"]).read_text().splitlines()]
        outside_calls = [call for call in calls
                         if not Path(call["cwd"]).is_relative_to(self.scratch)
                         and (call["kind"] == "rm" or (call["kind"] == "git"
                         and "init" not in call["args"]
                         and call["args"] != ["rev-parse", "--local-env-vars"]))]
        self.assertEqual(outside_calls, [], "a guard allowed an outside Git/cleanup operation")
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        return result.stdout + result.stderr

    def test_setup_failures_leave_outside_checkout_and_metadata_untouched(self):
        for mode in ("mktemp", "outside-result", "results", "mkdir", "bare-init",
                     "work-init", "missing-git-dir", "cd", "wrong-cwd", "owner-marker"):
            with self.subTest(mode=mode):
                self.run_case(mode, 2)

    def test_destructive_arm_rechecks_its_cwd(self):
        self.run_case("destructive-cwd", 2)

    def test_inherited_git_context_cannot_retarget_fixture_initialization(self):
        self.env.update(GIT_DIR=str(self.outside / ".git"),
                        GIT_WORK_TREE=str(self.outside),
                        GIT_INDEX_FILE=str(self.outside / ".git/index"))
        self.run_case("work-init", 2)

    def test_expected_negative_gate_outcomes_are_preserved(self):
        output = self.run_case("none", 0)
        self.assertIn("18 passed / 0 failed", output)


if __name__ == "__main__":
    unittest.main()
