"""Small strict reader for the Prometheus samples checked by serving gates.

promtool remains the exposition validator. This reader supplies exact scalar and
histogram checks without requiring a Python package in gate environments.
"""
from __future__ import annotations

import json
import math
import re

_SAMPLE = re.compile(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{(.*)\})?\s+([^\s]+)')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\[\\"n])*)"(?:,|$)')


def parse_samples(text: str) -> dict[tuple[str, tuple[tuple[str, str], ...]], float]:
    samples = {}
    for line in text.splitlines():
        if not line or line.startswith('#'):
            continue
        match = _SAMPLE.fullmatch(line)
        if not match:
            raise ValueError(f'malformed metric sample: {line!r}')
        name, raw_labels, raw_value = match.groups()
        labels = {}
        at = 0
        raw_labels = raw_labels or ''
        if raw_labels.endswith(','):
            raise ValueError(f'trailing label comma: {line!r}')
        while at < len(raw_labels):
            label = _LABEL.match(raw_labels, at)
            if not label or label.group(1) in labels:
                raise ValueError(f'malformed or duplicate labels: {line!r}')
            labels[label.group(1)] = json.loads('"' + label.group(2) + '"')
            at = label.end()
        key = (name, tuple(sorted(labels.items())))
        if key in samples:
            raise ValueError(f'duplicate metric sample: {line!r}')
        value = float(raw_value)
        if not math.isfinite(value):
            raise ValueError(f'non-finite metric sample: {line!r}')
        samples[key] = value
    return samples


def scalar(samples: dict, name: str, **labels: str) -> float:
    return samples[(name, tuple(sorted(labels.items())))]


def histogram(samples: dict, name: str, **labels: str) -> dict:
    count = scalar(samples, name + '_count', **labels)
    total = scalar(samples, name + '_sum', **labels)
    buckets = []
    for (metric, values), value in samples.items():
        row = dict(values)
        bound = row.pop('le', None)
        if metric == name + '_bucket' and row == labels and bound is not None:
            buckets.append((float(bound), value))
    if any(math.isnan(edge) or edge < 0 for edge, _ in buckets):
        raise ValueError(f'{name}: invalid bucket bound')
    buckets.sort()
    if len({edge for edge, _ in buckets}) != len(buckets):
        raise ValueError(f'{name}: duplicate numeric bucket bound')
    if not buckets or buckets[-1][0] != math.inf or buckets[-1][1] != count:
        raise ValueError(f'{name}: missing +Inf bucket or count mismatch')
    if total < 0 or count < 0 or count != int(count):
        raise ValueError(f'{name}: invalid count or sum')
    previous = 0
    for _, value in buckets:
        if value < previous or value > count or value != int(value):
            raise ValueError(f'{name}: non-cumulative buckets')
        previous = value
    return {'count': int(count), 'sum': total, 'buckets': buckets}
