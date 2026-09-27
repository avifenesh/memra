#!/usr/bin/env python3
"""DAY85 section 0, the wider pass: every `rc=$?` in the lane's shell scripts with the statement before it, flagged when that
statement is an echo, cat, tee or similar, a pipeline without pipefail, or a closing fi or done, or when a command
substitution precedes the read on its line; each flag is then read by hand. usage: rc-scan-wide.py (from the lane dir)"""
import re, sys, pathlib
root = pathlib.Path('.')
flag = []
for f in sorted(root.rglob('*.sh')):
    try: lines = f.read_text(errors='replace').splitlines()
    except Exception: continue
    text = '\n'.join(lines)
    pipefail = 'pipefail' in text
    for i, line in enumerate(lines):
        if 'rc=$?' not in line and 'rc=\\$?' not in line: continue
        at = line.find('$?')
        before = line[:at]
        reasons = []
        # same-line command substitution or command before the $? read
        seg = before.split(';')[-1]
        if '$(' in seg: reasons.append('subst-before-on-line')
        # previous statement on the same line
        if ';' in before:
            prev = before.rsplit(';', 1)[0].strip().split(';')[-1].strip()
        else:
            j = i - 1
            while j >= 0 and (not lines[j].strip() or lines[j].strip().startswith('#')): j -= 1
            prev = lines[j].strip() if j >= 0 else ''
        if re.match(r'^(echo|printf|log|mark|snap|date|tee|:|touch|cat|mkdir|cp|mv|rm)\b', prev): reasons.append('prev-is-' + prev.split()[0])
        if '|' in prev and '||' not in prev and not pipefail: reasons.append('pipe-no-pipefail')
        if re.search(r'\|\s*(tee|grep|tail|head|cut|sed|awk)\b', prev) and not pipefail: reasons.append('pipe-tail-no-pipefail')
        if prev.endswith('fi') or prev.endswith('done'): reasons.append('after-' + prev.split()[-1])
        if reasons:
            flag.append((str(f), i + 1, reasons, prev[:110], line.strip()[:110]))
for f, n, r, p, l in flag:
    print(f"{f}:{n} {','.join(r)}\n   prev: {p}\n   line: {l}")
print(len(flag), 'flagged')
