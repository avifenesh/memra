"""The battery's CPU tests take the two rig lock NAMES under a private directory.

Lead ruling 12 (research/spill-lead-20260919/INTEGRATION-DAY12.md, 2026-09-21): "The tier
battery's Python tests take the real rig lock path; they must take a private lock path under
test, so a serving job on the rig cannot redden a CPU suite." The integ10 battery's first
attempt hit `REFUSED: [Errno 11] Resource temporarily unavailable` in 20 collector tests while a
serve-smoke held /tmp/memra-5090.lock; with both real lock paths held, 24 of this suite's tests
failed (research/spill-d-20260919/day13/pytest-control-lock-held.log).

The seam is MEMRA_TIER_BATTERY_LOCK_DIR (tools/tier-battery.py `lock_table`, and the same rule
in tools/tier-rig-bootstrap.sh, whose --dry-run holds the rig lock on purpose): the canonical
NAMES `memra-5090.lock` and `memra-gpu.lock` re-rooted under a private directory. Nothing else
changes: the rig->name table, the refusal on contention, the receipts (which record the private
path and therefore never validate against the canonical table in a process without the seam).
Production tools that pin the two names by literal (tools/tier-lock-proof.py, the two legacy
kv-host-spill gates) are NOT given a seam: the tests that drive them already run them from a
fixture copy, and `substitute` rewrites the literal in that copy only.

Every TestCase whose tests take a lock mixes this in. Tests that validate COMMITTED receipts
(recorded under the canonical paths) run their validator under `canonical_locks()` or hand a
child `canonical_env()`.

The seam alone moves nothing (review of PR #592): a collector `--execute`/`--dry-run` and a
bootstrap run refuse under the seam unless the process also passes FLAG, so an inherited
environment variable can never move a real campaign off the canonical lock; every launch site
in this suite passes FLAG, and `test_day10.py` / `test_collector.py` carry the red arms.
"""
import contextlib
import os
from pathlib import Path
import tempfile
from unittest import mock

SEAM = 'MEMRA_TIER_BATTERY_LOCK_DIR'
FLAG = '--private-lock-dir-for-tests'
CANONICAL = ('/tmp/memra-5090.lock', '/tmp/memra-gpu.lock')
ROOT = Path(__file__).resolve().parents[4]


class PrivateLockMixin:
    """setUp: a fresh private lock directory, exported to children and patched into the
    battery module's LOCKS table (set `BATTERY` to the module the test file loaded)."""

    BATTERY = None

    def setUp(self):
        super().setUp()
        self._private_tmp = tempfile.TemporaryDirectory(prefix='tier-private-lock-')
        self.addCleanup(self._private_tmp.cleanup)
        self.lock_dir = Path(self._private_tmp.name)
        self._seam_patches = [mock.patch.dict(os.environ, {SEAM: str(self.lock_dir)})]
        if self.BATTERY is not None:
            self._seam_patches.append(
                mock.patch.dict(self.BATTERY.LOCKS, self.BATTERY.lock_table(self.lock_dir)))
        for patcher in self._seam_patches:
            patcher.start()
            self.addCleanup(patcher.stop)

    def private(self, canonical):
        """The private path for one of the two canonical names; anything else is refused."""
        if canonical not in CANONICAL:
            raise ValueError(f'not a canonical rig lock: {canonical!r}')
        return str(self.lock_dir / Path(canonical).name)

    def substitute(self, text):
        """Rewrite the canonical literals in a FIXTURE COPY of a tool (never the tracked file)."""
        for canonical in CANONICAL:
            text = text.replace(canonical, self.private(canonical))
        return text

    def proof_copy(self, directory=None):
        """A copy of tools/tier-lock-proof.py that accepts the private paths. The tracked tool
        accepts exactly the two canonical paths and no environment switch, by design."""
        target = Path(directory or self.lock_dir) / 'tier-lock-proof.py'
        target.write_text(self.substitute((ROOT / 'tools/tier-lock-proof.py').read_text()))
        return target

    def canonical_env(self):
        """Environment for a child that must see the canonical table (committed receipts)."""
        return {key: value for key, value in os.environ.items() if key != SEAM}

    @contextlib.contextmanager
    def canonical_locks(self):
        """Temporarily restore the canonical table and environment inside one test."""
        for patcher in reversed(self._seam_patches):
            patcher.stop()
        try:
            yield
        finally:
            for patcher in self._seam_patches:
                patcher.start()
