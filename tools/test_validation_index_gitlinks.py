#!/usr/bin/env python3
"""LocalTree gitlink detection reads the index's current content, not a stale cached answer."""
import hashlib
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

import validation_plan as vp


def g(repo, *args):
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True,
                          env=dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                                   GIT_AUTHOR_NAME='t', GIT_AUTHOR_EMAIL='t@t',
                                   GIT_COMMITTER_NAME='t', GIT_COMMITTER_EMAIL='t@t')).stdout


class IndexGitlinks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        g(self.repo, 'init', '-q')
        (self.repo / 'f').write_text('x\n')
        (self.repo / 'd').mkdir()
        (self.repo / 'd' / 'g').write_text('y\n')
        g(self.repo, 'add', 'f', 'd/g')
        g(self.repo, 'commit', '-qm', 'base')
        self.head = g(self.repo, 'rev-parse', 'HEAD').decode().strip()

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_forged_same_size_same_mtime_index_rewrite_is_seen(self):
        tree = vp.LocalTree(self.repo)
        self.assertEqual(tree.input_modes('f', recursive=False), {'f': '100644'})
        index = self.repo / '.git' / 'index'
        before = index.stat()
        # Turn the regular file into a gitlink in place, the way a hostile process sharing the
        # planner could: flip the entry's mode field, fix the trailing checksum, write the same
        # number of bytes into the same inode and put the mtime back.
        data = bytearray(index.read_bytes())
        self.assertEqual(data[:4], b'DIRC')
        entry = 12  # first entry: 'd/g' sorts before 'f'
        for _ in range(2):
            name_len = int.from_bytes(data[entry + 60:entry + 62], 'big') & 0xFFF
            name = bytes(data[entry + 62:entry + 62 + name_len])
            if name == b'f':
                break
            entry += (62 + name_len + 8) // 8 * 8
        self.assertEqual(name, b'f')
        data[entry + 24:entry + 28] = (0o160000).to_bytes(4, 'big')
        data[-20:] = hashlib.sha1(bytes(data[:-20])).digest()
        with open(index, 'r+b') as out:
            out.write(data)
        os.utime(index, ns=(before.st_atime_ns, before.st_mtime_ns))
        after = index.stat()
        self.assertEqual((after.st_ino, after.st_size, after.st_mtime_ns),
                         (before.st_ino, before.st_size, before.st_mtime_ns))
        self.assertIn(b'160000', g(self.repo, 'ls-files', '--stage', 'f'))
        self.assertEqual(vp.LocalTree(self.repo).input_modes('f', recursive=False), {'f': '160000'})

    def test_cached_and_pathspec_paths_agree(self):
        g(self.repo, 'update-index', '--add', '--cacheinfo', '160000,' + self.head + ',d/sub')
        tree = vp.LocalTree(self.repo)
        self.assertEqual(tree.input_modes('d')['d/sub'], '160000')
        self.assertNotIn('d/sub', tree.input_modes('d', recursive=False))
        # Non-literal prefixes keep Git's pathspec matching.
        self.assertEqual(tree.input_modes('d/*')['d/sub'], '160000')
        self.assertEqual(tree.input_modes('.')['d/sub'], '160000')


if __name__ == '__main__':
    unittest.main()
