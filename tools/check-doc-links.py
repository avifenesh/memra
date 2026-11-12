#!/usr/bin/env python3
"""Check relative links and heading anchors in the repository's markdown.

Usage: tools/check-doc-links.py [--repo DIR] [--all] [--exclude PATH]...

Sources are the *.md files git knows (tracked, or untracked and not ignored) that exist in the
work tree, outside research/ unless --all is given. Files under research/ stay valid link
targets either way.

Every inline link, image, reference definition and HTML href/src whose target is relative (no
URL scheme, not protocol-relative) must name a file or directory in that set of paths; a
leading / is the repository root, as on GitHub. A #fragment on a markdown target must name an
anchor in that file: a heading slug, or an id/name attribute. A line fragment (#L12) counts only
on a ?plain=1 link or a non-markdown target, whose fragments are not checked. Links in fenced
code, inline code and HTML comments do not render on GitHub and are skipped. Indented code blocks
are not detected.

Heading slugs follow GitHub: take the heading's rendered text without leading or trailing
whitespace, lowercase its ASCII letters, drop every character that is not a letter, mark, decimal
digit, connector punctuation (such as _), hyphen or space, then turn each space into a hyphen. A
slug already taken gets -1, -2, ... in document order. YAML front matter has no headings.

--exclude PATH  Do not fail on broken links in PATH, a source file or a directory (trailing /).
                They are still listed on stderr. Use it only for a known break that another
                change owns. An exclusion that hides no broken link fails, so the change that
                fixes the file must also drop its exclusion. Repeatable.

Exit status: 0 with a count when nothing is broken. 1 with one `FILE:LINE: missing target LINK`
or `FILE:LINE: missing anchor LINK` line per broken link on stdout, or on a stale exclusion.
2 when git cannot list the files or an argument names no markdown source.
"""
from __future__ import annotations

import argparse
import bisect
import html
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys
import unicodedata
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ASCII_PUNCT = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
SCHEME = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*:')
FENCE_OPEN = re.compile(r'(`{3,}|~{3,})(.*)$')
FENCE_CLOSE = re.compile(r'(`{3,}|~{3,})[ \t]*$')
QUOTE_PREFIX = re.compile(r'^(?: {0,3}> ?)+')
ATX = re.compile(r'^ {0,3}(#{1,6})(?:[ \t]+(.*?))?[ \t]*$')
ATX_CLOSE = re.compile(r'(?:^|[ \t]+)#+$')
SETEXT = re.compile(r'^ {0,3}(=+|-+)[ \t]*$')
NOT_PARAGRAPH = re.compile(r'^(?: {4}|\t| {0,3}(?:[-+*]|\d{1,9}[.)])(?:[ \t]|$)| {0,3}[#>|<])')
BLANK_LINE = re.compile(r'\n[ \t]*\n')
REFDEF = re.compile(r'^ {0,3}\[((?:[^\]\\\n]|\\.)+)\]:[ \t]*\n?[ \t]*(<[^<>\n]*>|\S+)', re.M)
HTML_LINK = re.compile(r'<(?:a|img|source)\b[^>]*?\s(?:href|src)\s*=\s*'
                       r'(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'>]+))', re.I)
HTML_ID = re.compile(r'<[A-Za-z][^>]*?\s(?:id|name)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^\s"\'>]+))')
ASCII_LOWER = str.maketrans('ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')
LINE_FRAGMENT = re.compile(r'^L\d+(?:C\d+)?(?:-L\d+(?:C\d+)?)?$')
# Private-use stand-ins for backslash-escaped punctuation while heading markup is stripped.
ESCAPE_BASE = 0xF0000


def blank(text):
    """The same text with every character except newlines turned into a space."""
    return re.sub(r'[^\n]', ' ', text)


def mask_blocks(text):
    """Blank YAML front matter, fenced code blocks and HTML comments, keeping every offset."""
    lines = text.split('\n')
    out = []
    if lines and lines[0].strip() == '---':
        end = next((i for i in range(1, len(lines)) if lines[i].strip() in ('---', '...')), None)
        if end is not None:
            out = [blank(line) for line in lines[:end + 1]]
            lines = lines[end + 1:]
    fence = None
    for line in lines:
        inner = QUOTE_PREFIX.sub('', line).lstrip(' \t')
        if fence is not None:
            match = FENCE_CLOSE.match(inner)
            if match and match.group(1)[0] == fence[0] and len(match.group(1)) >= fence[1]:
                fence = None
            out.append(blank(line))
            continue
        match = FENCE_OPEN.match(inner)
        if match and not (match.group(1)[0] == '`' and '`' in match.group(2)):
            fence = (match.group(1)[0], len(match.group(1)))
            out.append(blank(line))
            continue
        out.append(line)
    return re.sub(r'<!--.*?-->', lambda m: blank(m.group()), '\n'.join(out), flags=re.S)


def code_spans(text):
    """(start, end) of every inline code span, CommonMark-style: a backtick run closes at the
    next run of the same length within the same paragraph."""
    runs = [[m.start(), m.end()] for m in re.finditer(r'`+', text)]
    breaks = [m.start() for m in BLANK_LINE.finditer(text)]
    spans, i = [], 0
    while i < len(runs):
        start, end = runs[i]
        if start > 0 and text[start - 1] == '\\':
            start += 1
            if start == end:
                i += 1
                continue
        width = end - start
        limit = bisect.bisect_left(breaks, end)
        stop = breaks[limit] if limit < len(breaks) else len(text)
        closer = next((j for j in range(i + 1, len(runs)) if runs[j][0] >= stop
                       or runs[j][1] - runs[j][0] == width), None)
        if closer is not None and runs[closer][0] < stop:
            spans.append((start, runs[closer][1]))
            i = closer + 1
        else:
            # No closer in this paragraph: the run is literal backticks.
            i += 1
    return spans


def mask_code(text):
    out = list(text)
    for start, end in code_spans(text):
        for k in range(start, end):
            if out[k] != '\n':
                out[k] = ' '
    return ''.join(out)


def escaped(text, pos):
    count = 0
    while pos - count - 1 >= 0 and text[pos - count - 1] == '\\':
        count += 1
    return count % 2 == 1


def unescape(text):
    return re.sub(r'\\([!-/:-@\[-`{-~])', r'\1', text)


def opening_bracket(text, close):
    """Index of the [ that matches the ] at `close`, within one paragraph, or None."""
    depth, k = 0, close - 1
    while k >= 0:
        char = text[k]
        if char == '\n' and text[max(0, k - 200):k].rstrip(' \t').endswith('\n'):
            return None
        if char in '[]' and not escaped(text, k):
            if char == ']':
                depth += 1
            elif depth == 0:
                return k
            else:
                depth -= 1
        k -= 1
    return None


def destination(text, start):
    """The destination of an inline link whose ( is just before `start`, or None when the
    text there is not a CommonMark link destination and optional title closed by )."""
    n, j = len(text), start
    while j < n and text[j] in ' \t':
        j += 1
    if j < n and text[j] == '\n':
        j += 1
        while j < n and text[j] in ' \t':
            j += 1
    if j < n and text[j] == '<':
        k = j + 1
        while k < n and text[k] not in '<>\n':
            k += 2 if text[k] == '\\' else 1
        if k >= n or text[k] != '>':
            return None
        dest, j = text[j + 1:k], k + 1
    else:
        depth, k = 0, j
        while k < n:
            char = text[k]
            if char == '\\' and k + 1 < n and text[k + 1] in ASCII_PUNCT:
                k += 2
                continue
            if char.isspace() or ord(char) < 0x20:
                break
            if char == '(':
                depth += 1
            elif char == ')':
                if depth == 0:
                    break
                depth -= 1
            k += 1
        if depth:
            return None
        dest, j = text[j:k], k
    t = j
    while t < n and text[t] in ' \t\n':
        t += 1
    if t > j and t < n and text[t] in '"\'(':
        closer = ')' if text[t] == '(' else text[t]
        u = t + 1
        while u < n and text[u] != closer:
            u += 2 if text[u] == '\\' else 1
        if u >= n:
            return None
        t = u + 1
        while t < n and text[t] in ' \t\n':
            t += 1
    if t < n and text[t] == ')':
        return unescape(dest)
    return None


def label_key(label):
    return ' '.join(label.split()).casefold()


def definitions(masked):
    """(offset, label key, destination) of each link reference definition. A definition
    cannot interrupt a paragraph, so it starts a block."""
    for match in REFDEF.finditer(masked):
        if match.group(1).startswith('^'):
            continue
        before = masked[:match.start()]
        previous = before[before.rfind('\n', 0, len(before) - 1) + 1:] if before else ''
        if before and previous.strip() and not REFDEF.match(previous) and not ATX.match(previous):
            continue
        dest = match.group(2)
        yield (match.start(), label_key(match.group(1)),
               unescape(dest[1:-1] if dest.startswith('<') else dest))


def link_targets(text):
    """(offset, raw target) of every link in markdown `text` that renders on GitHub."""
    masked = mask_code(mask_blocks(text))
    found = []
    for match in re.finditer(r'\]\(', masked):
        if escaped(masked, match.start()):
            continue
        if opening_bracket(masked, match.start()) is None:
            continue
        dest = destination(masked, match.end())
        if dest is not None:
            found.append((match.start(), dest))
    for offset, _, dest in definitions(masked):
        found.append((offset, dest))
    for match in HTML_LINK.finditer(masked):
        found.append((match.start(), html.unescape(next(g for g in match.groups() if g is not None))))
    return sorted(found)


def heading_text(raw, labels=frozenset()):
    """The rendered text of a heading's inline markdown, which GitHub slugs. `labels` are the
    document's reference definitions: [text][label] is a link only when its label is defined."""
    pieces, last = [], 0
    for start, end in code_spans(raw) + [(len(raw), len(raw))]:
        part = raw[last:start]
        part = re.sub(r'\\([!-/:-@\[-`{-~])', lambda m: chr(ESCAPE_BASE + ord(m.group(1))), part)
        part = re.sub(r'!\[[^\]]*\]\((?:[^()]|\([^()]*\))*\)', '', part)
        part = re.sub(r'\[([^\]]*)\]\((?:[^()]|\([^()]*\))*\)', r'\1', part)
        part = re.sub(r'\[([^\]]*)\]\[([^\]]*)\]', lambda m: m.group(1) if label_key(
            m.group(2) or m.group(1)) in labels else m.group(), part)
        part = re.sub(r'<((?:https?|ftp|mailto):[^<>\s]*)>', r'\1', part)
        part = re.sub(r'</?[A-Za-z][^<>]*>', '', part)
        previous = None
        while previous != part:
            previous = part
            part = re.sub(r'(?<![\w\\])(__?)(?=\S)(.+?)(?<=\S)\1(?!\w)', r'\2', part)
        part = html.unescape(part)
        part = ''.join(chr(ord(c) - ESCAPE_BASE) if ord(c) >= ESCAPE_BASE else c for c in part)
        pieces.append(part)
        if start < len(raw):
            code = raw[start:end].strip('`').replace('\n', ' ')
            if code.startswith(' ') and code.endswith(' ') and code.strip(' '):
                code = code[1:-1]
            pieces.append(code)
        last = end
    # GitHub trims the rendered text, so inline HTML or an image at either end leaves no hyphen.
    return ''.join(pieces).strip(' \t\n')


def slug(text):
    """GitHub's anchor for a heading whose rendered text is `text`. Only ASCII letters are
    lowercased (GitHub's own example keeps a Greek capital)."""
    kept = []
    for char in text.translate(ASCII_LOWER):
        category = unicodedata.category(char)
        if char in '- ' or category[0] in 'LM' or category in ('Nd', 'Pc'):
            kept.append(char)
    return ''.join(kept).replace(' ', '-')


class Slugger:
    """Unique slugs in document order, as GitHub assigns them."""

    def __init__(self):
        self.seen = {}

    def __call__(self, text):
        base = candidate = slug(text)
        while candidate in self.seen:
            self.seen[base] += 1
            candidate = f'{base}-{self.seen[base]}'
        self.seen[candidate] = 0
        return candidate


def anchors(text):
    """Every fragment that resolves in markdown `text`: heading slugs and id/name attributes."""
    blocks = mask_blocks(text)
    masked = mask_code(blocks)
    labels = frozenset(label for _, label, _ in definitions(masked))
    lines = blocks.split('\n')
    unique = Slugger()
    found = set()
    for i, line in enumerate(lines):
        line = QUOTE_PREFIX.sub('', line)
        match = ATX.match(line)
        if match:
            content = ATX_CLOSE.sub('', match.group(2) or '').strip()
            found.add(unique(heading_text(content, labels)))
            continue
        match = SETEXT.match(line)
        if match and i > 0:
            # The underline closes the paragraph above it, which starts after a blank line or
            # after a block that cannot continue into a paragraph (a heading or another rule).
            paragraph = []
            k = i - 1
            while k >= 0 and lines[k].strip():
                above = QUOTE_PREFIX.sub('', lines[k])
                if ATX.match(above) or SETEXT.match(above):
                    break
                paragraph.insert(0, above)
                k -= 1
            if paragraph and not any(NOT_PARAGRAPH.match(p) for p in paragraph):
                found.add(unique(heading_text(' '.join(p.strip() for p in paragraph), labels)))
    for match in HTML_ID.finditer(masked):
        found.add(html.unescape(next(g for g in match.groups() if g is not None)))
    return found


def git_paths(repo):
    result = subprocess.run(['git', '-C', str(repo), 'ls-files', '-z', '--cached', '--others',
                             '--exclude-standard'], capture_output=True)
    if result.returncode != 0:
        raise OSError(result.stderr.decode(errors='replace').strip() or 'git ls-files failed')
    return sorted({name for name in result.stdout.decode('utf-8', 'surrogateescape').split('\0') if name})


def resolve(source, target):
    """Repository path a relative link in `source` names, or None when it leaves the tree."""
    if target.startswith('/'):
        parts = []
    else:
        parts = list(PurePosixPath(source).parent.parts)
    for segment in target.split('/'):
        if segment == '..':
            if not parts:
                return None
            parts.pop()
        elif segment not in ('', '.'):
            parts.append(segment)
    return '/'.join(parts)


def check(repo, include_research=False, exclude=()):
    """(broken, hidden, stale, counts): broken links outside exclusions, broken links the
    exclusions hid, exclusions that hid nothing, and (links, files) checked."""
    paths = git_paths(repo)
    known = set(paths)
    directories = {''}
    for name in paths:
        cut = name.rfind('/')
        while cut > 0 and name[:cut] not in directories:
            directories.add(name[:cut])
            cut = name.rfind('/', 0, cut)
    # A path git lists but the work tree has deleted is neither a source nor a target.
    sources = [name for name in paths if name.endswith('.md')
               and (include_research or not name.startswith('research/'))
               and os.path.isfile(repo / name)]
    for item in exclude:
        if not any(name == item or (item.endswith('/') and name.startswith(item)) for name in sources):
            raise ValueError('--exclude names no markdown source: ' + item)
    cache = {}

    def anchors_of(name):
        if name not in cache:
            cache[name] = anchors((repo / name).read_text(encoding='utf-8', errors='replace'))
        return cache[name]

    broken, hidden, links = [], [], 0
    for name in sources:
        text = (repo / name).read_text(encoding='utf-8', errors='replace')
        newlines = [m.start() for m in re.finditer('\n', text)]
        for offset, raw in link_targets(text):
            if not raw or SCHEME.match(raw) or raw.startswith('//'):
                continue
            links += 1
            target, _, fragment = raw.partition('#')
            target, _, query = target.partition('?')
            target, fragment = unquote(target), unquote(fragment)
            line = bisect.bisect_right(newlines, offset - 1) + 1
            path = name if target == '' else resolve(name, target)
            problem = None
            if path is None or not ((path in known or path in directories)
                                    and os.path.lexists(repo / path)):
                problem = 'missing target'
            elif fragment and path in known and path.endswith('.md'):
                plain = 'plain=1' in query.split('&')
                if not (plain and LINE_FRAGMENT.match(fragment)) and fragment not in anchors_of(path):
                    problem = 'missing anchor'
            if problem:
                row = (name, line, problem, raw)
                excluded = any(name == item or (item.endswith('/') and name.startswith(item))
                               for item in exclude)
                (hidden if excluded else broken).append(row)
    stale = [item for item in exclude
             if not any(row[0] == item or (item.endswith('/') and row[0].startswith(item))
                        for row in hidden)]
    return broken, hidden, stale, (links, len(sources))


def main(argv=None):
    title, _, rules = __doc__.split('\n\n', 2)
    parser = argparse.ArgumentParser(description=title + '\n\n' + rules,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--repo', type=Path, default=ROOT, help='repository root (default: this checkout)')
    parser.add_argument('--all', action='store_true', help='also check sources under research/')
    parser.add_argument('--exclude', action='append', default=[], metavar='PATH',
                        help='do not fail on broken links in PATH (a file, or a directory ending in /); '
                             'fails when it hides nothing')
    args = parser.parse_args(argv)
    exclude = [item[2:] if item.startswith('./') else item for item in args.exclude]
    try:
        broken, hidden, stale, (links, files) = check(args.repo.resolve(), args.all, exclude)
    except (OSError, ValueError) as error:
        print('check-doc-links: ' + str(error), file=sys.stderr)
        return 2
    for name, line, problem, raw in broken:
        print(f'{name}:{line}: {problem} {raw}')
    sys.stdout.flush()
    for name, line, problem, raw in hidden:
        print(f'{name}:{line}: {problem} {raw} (excluded)', file=sys.stderr)
    for item in stale:
        print(f'check-doc-links: --exclude {item} hides no broken link; drop the exclusion',
              file=sys.stderr)
    summary = f'{links} relative links in {files} files'
    if broken or stale:
        print(f'check-doc-links: {len(broken)} broken of {summary}', file=sys.stderr)
        return 1
    print(f'check-doc-links: {summary}, none broken' + (f' ({len(hidden)} excluded)' if hidden else ''))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
