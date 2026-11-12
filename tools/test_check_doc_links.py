#!/usr/bin/env python3
"""Fixture cases for tools/check-doc-links.py.

Each case builds a small git work tree, runs the checker on it and asserts the exact verdict:
the exit status, the broken-link lines and the link count. A checker that passes everything,
fails everything, misses a link kind or reads links inside code fails here.
"""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest

REPO = Path(__file__).resolve().parent.parent
CHECK = REPO / 'tools/check-doc-links.py'
# docs/FLAGS.md writes a formula as E[tok](j), which renders as a link to a file named j.
# The exclusion goes in the change that escapes it; the checker fails while it hides nothing.
TREE_EXCLUDES = ('docs/FLAGS.md',)


def load_checker():
    spec = importlib.util.spec_from_file_location('check_doc_links', CHECK)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocLinkFixtures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='memra-doc-links-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Keep the user's git configuration (global excludes, hooks) out of the fixture.
        self.env = dict(os.environ, GIT_CONFIG_NOSYSTEM='1', GIT_CONFIG_GLOBAL=os.devnull,
                        HOME=str(self.root))
        subprocess.run(['git', 'init', '-q', str(self.root)], check=True, env=self.env)

    def put(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip('\n'))

    def run_check(self, *args):
        result = subprocess.run([sys.executable, str(CHECK), '--repo', str(self.root), *args],
                                capture_output=True, text=True, env=self.env)
        return result.returncode, result.stdout.splitlines(), result.stderr

    def assert_clean(self, links, files, *args):
        code, out, err = self.run_check(*args)
        self.assertEqual(code, 0, out + [err])
        self.assertEqual(out[-1], f'check-doc-links: {links} relative links in {files} files, none broken')

    def assert_broken(self, expected, *args):
        code, out, err = self.run_check(*args)
        self.assertEqual(code, 1, err)
        self.assertEqual(out, expected)

    def test_valid_links_of_every_kind_pass(self):
        self.put('README.md', '''
            # memra

            See [the guide](docs/guide.md), [its setup](docs/guide.md#setup), [docs](docs/),
            [from the root](/docs/guide.md), [here](#memra) and ![a chart](docs/chart.svg).
            [Spaced](<docs/a file.md> "title") and [encoded](docs/a%20file.md).
            A [reference][guide] and <a href="docs/guide.md#setup">raw HTML</a>.
            A source line: [code](tools/run.py#L3) and [plain](docs/guide.md?plain=1#L2).
            Off-site links are not checked: [site](https://example.com/missing.md).

            [guide]: docs/guide.md#setup
            ''')
        self.put('docs/guide.md', '# Guide\n\n## Setup\n\nBack to [the start](../README.md#memra).\n')
        self.put('docs/a file.md', '# Spaced\n')
        self.put('docs/chart.svg', '<svg/>\n')
        self.put('tools/run.py', 'print()\n')
        self.assert_clean(13, 3)

    def test_missing_file_fails(self):
        self.put('docs/a.md', '# A\n\nText.\n\nSee [missing](nope.md) and [ok](a.md).\n')
        self.assert_broken(['docs/a.md:5: missing target nope.md'])

    def test_missing_anchor_fails(self):
        self.put('docs/a.md', '# A\n\n## Setup\n\n[good](b.md#b) [bad](b.md#install) [self](#setup)\n')
        self.put('docs/b.md', '# B\n')
        self.assert_broken(['docs/a.md:5: missing anchor b.md#install'])

    def test_duplicate_headings_take_numbered_suffixes(self):
        self.put('a.md', '''
            # A

            ## Setup

            ## Setup

            ## Setup

            [one](#setup) [two](#setup-1) [three](#setup-2) [four](#setup-3)
            ''')
        self.assert_broken(['a.md:9: missing anchor #setup-3'])

    def test_links_in_code_and_comments_are_ignored(self):
        self.put('a.md', '''
            # A

            ```sh
            [fenced](missing-1.md)
            ```

            ````md
            ```
            [nested fence](missing-2.md)
            ```
            ````

            ~~~
            [tilde fence](missing-3.md)
            ~~~

            Inline `[code](missing-4.md)` and ``a ` [double](missing-5.md)``.
            <!-- [comment](missing-6.md) -->

            After the code, [this one](missing-7.md) is a real link.
            ''')
        self.assert_broken(['a.md:20: missing target missing-7.md'])

    def test_explicit_id_and_name_anchors_resolve(self):
        self.put('a.md', '''
            # A

            <a id="old-name"></a>
            <a name="older-name"></a>

            ```html
            <a id="in-code"></a>
            ```

            [id](#old-name) [name](#older-name) [code id](#in-code)
            ''')
        self.assert_broken(['a.md:10: missing anchor #in-code'])

    def test_heading_slugs_follow_github(self):
        # The last three anchors are copied from GitHub's rendering of files in this repository.
        headings = {
            'API keys \u2014 multi-key (lane:api-keys 2026-08-05)': 'api-keys--multi-key-laneapi-keys-2026-08-05',
            '`MEMRA_FAST=0` oracle path': 'memra_fast0-oracle-path',
            '_Emphasis_ and snake_case_name': 'emphasis-and-snake_case_name',
            'C# ##': 'c',
            'Escaped \\_under\\_ [link](x.md) end <a id="x"></a>': 'escaped-_under_-link-end',
            'Why NOT BN=32 / facc[4][4] with 4 warps (the "blind shrink")':
                'why-not-bn32--facc44-with-4-warps-the-blind-shrink',
            'K4-wgmma v2: full chain IN-BAND, first timing 210.9µs (2026-07-27)':
                'k4-wgmma-v2-full-chain-in-band-first-timing-2109µs-2026-07-27',
            'Gate table (native class; C(d_b,d_q) = √(d_b·u_b² + d_q·u_q²), u_b=2⁻⁸, u_q=2⁻⁴)':
                'gate-table-native-class-cd_bd_q--d_bu_b--d_qu_q-u_b2-u_q2',
        }
        body = ''.join(f'## {heading}\n\n' for heading in headings)
        body += 'Setext heading\n---\n\n'
        body += ' '.join(f'[{i}](#{anchor})' for i, anchor in enumerate(headings.values()))
        body += ' [setext](#setext-heading) [wrong](#c-1)\n'
        self.put('x.md', '# X\n')
        self.put('a.md', body)
        self.assert_broken([f'a.md:{2 * len(headings) + 4}: missing anchor #c-1'])
        checker = load_checker()
        # Only ASCII letters are lowercased, as in GitHub's own example with a capital theta.
        self.assertEqual(checker.slug(checker.heading_text('Straße ΘΔ Ünï')), 'straße-ΘΔ-Ünï')

    def test_escaped_brackets_are_not_a_link(self):
        self.put('a.md', '# A\n\nA rate E\\[tok\\](j) and [tok\\](k) are text.\n\nA rate E[tok](j) is a link.\n')
        self.assert_broken(['a.md:5: missing target j'])

    def test_research_sources_are_checked_only_with_all(self):
        self.put('docs/a.md', '# A\n\n[lane](../research/lane/NOTES.md#result)\n')
        self.put('research/lane/NOTES.md', '# Notes\n\n## Result\n\n[gone](missing.json)\n')
        self.assert_clean(1, 1)
        self.assert_broken(['research/lane/NOTES.md:5: missing target missing.json'], '--all')

    def test_exclude_hides_a_known_break_and_refuses_when_stale(self):
        self.put('a.md', '# A\n\n[gone](missing.md)\n')
        self.put('b.md', '# B\n\n[a](a.md)\n')
        code, out, err = self.run_check('--exclude', 'a.md')
        self.assertEqual(code, 0, err)
        self.assertIn('a.md:3: missing target missing.md (excluded)', err)
        self.assertEqual(out[-1], 'check-doc-links: 2 relative links in 2 files, none broken (1 excluded)')
        code, out, err = self.run_check('--exclude', 'a.md', '--exclude', 'b.md')
        self.assertEqual(code, 1)
        self.assertIn('--exclude b.md hides no broken link', err)
        code, _, err = self.run_check('--exclude', 'nope.md')
        self.assertEqual(code, 2)
        self.assertIn('names no markdown source', err)

    def test_targets_outside_the_tree_ignored_or_deleted_are_missing(self):
        self.put('.gitignore', 'local.md\n')
        self.put('local.md', '# Local\n')
        self.put('gone.md', '# Gone\n')
        subprocess.run(['git', '-C', str(self.root), 'add', 'gone.md'], check=True, env=self.env)
        (self.root / 'gone.md').unlink()
        self.put('a.md', '# A\n\n[up](../outside.md) [ignored](local.md) [deleted](gone.md)\n')
        self.assert_broken(['a.md:3: missing target ../outside.md', 'a.md:3: missing target local.md',
                            'a.md:3: missing target gone.md'])

    def test_line_fragment_on_markdown_needs_plain_view(self):
        self.put('a.md', '# A\n\n[rendered](b.md#L2) [plain](b.md?plain=1#L2)\n')
        self.put('b.md', '# B\n\ntext\n')
        self.assert_broken(['a.md:3: missing anchor b.md#L2'])

    def test_repository_tree_has_no_broken_link(self):
        args = [arg for path in TREE_EXCLUDES for arg in ('--exclude', path)]
        result = subprocess.run([sys.executable, str(CHECK), '--repo', str(REPO), *args],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
