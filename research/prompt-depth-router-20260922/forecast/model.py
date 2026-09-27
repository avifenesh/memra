"""Bounded byte features and a shared, shallow output-format predictor."""

import collections
import re

KINDS = ("prose", "code", "numeric", "unknown")
FEATURES = (
    "prompt_kind", "bytes_seen", "fenced", "trailing_ticks", "digits",
    "letters", "math_symbols", "code_symbols", "spaces", "newlines",
    "underscores", "line_indent", "trailing_colon", "code_intro_words",
    "code_keywords", "trailing_backtick",
)
INTRO = {b"code", b"python", b"rust", b"json", b"sql", b"implementation", b"function", b"script"}
KEYWORDS = {b"def", b"class", b"return", b"import", b"fn", b"let", b"const", b"function", b"select"}
MATH = b"+-*/=.,:%"
CODE = b"{}[];"
SPACE = b" \t\n\r\v\f"


class State:
    def __init__(self, hint):
        self.hint = hint
        self.window = collections.deque(maxlen=128)
        self.seen = 0
        self.fenced = False
        self.ticks = 0

    def observe(self, data):
        for byte in data:
            self.seen = min(512, self.seen + 1)
            self.ticks = min(4, self.ticks + 1) if byte == 96 else 0
            if self.ticks == 3:
                self.fenced = not self.fenced
            self.window.append(byte)

    def features(self):
        window = bytes(self.window)
        words = re.findall(rb"[a-z_]+", window.lower())
        line = window.rsplit(b"\n", 1)[-1]
        tail = window.rstrip(SPACE)
        return (
            self.hint, self.seen, int(self.fenced), min(3, self.ticks),
            sum(48 <= c <= 57 for c in window),
            sum(65 <= c <= 90 or 97 <= c <= 122 for c in window),
            sum(c in MATH for c in window), sum(c in CODE for c in window),
            window.count(b" "), window.count(b"\n"), window.count(b"_"),
            min(8, len(line) - len(line.lstrip(b" "))),
            int(tail.endswith(b":")),
            sum(word in INTRO for word in words),
            sum(word in KEYWORDS for word in words),
            int(tail.endswith(b"`")),
        )


def simple_rule(features):
    hint, seen, fenced, _, digits, letters, math = features[:7]
    if digits >= 4 and digits + math > letters:
        return 2
    if fenced or (features[12] and features[13]):
        return 1
    return hint if seen < 24 else 0


def format_map(data):
    """Independent, line-aware annotation of fenced material; 3 is ambiguous."""
    labels = bytearray(len(data))
    marker = None
    content_kind = 0
    offset = 0
    for line in data.splitlines(keepends=True):
        match = re.match(rb" {0,3}(`{3,}|~{3,})([^\r\n]*)", line)
        delimiter = False
        if match:
            fence, info = match.groups()
            if marker is None:
                marker = fence
                language = info.strip().split(None, 1)[0].lower() if info.strip() else b""
                content_kind = 3 if language in {b"text", b"plain", b"plaintext", b"md", b"markdown"} else 1
                delimiter = True
            elif fence[:1] == marker[:1] and len(fence) >= len(marker) and not info.strip():
                marker = None
                delimiter = True
        kind = 3 if delimiter else content_kind if marker is not None else 0
        labels[offset:offset + len(line)] = bytes([kind]) * len(line)
        offset += len(line)
    return labels


def future_kind(data, labels, start, end):
    content = data[start:end]
    known = [(c, label) for c, label in zip(content, labels[start:end]) if c not in SPACE]
    if len(known) < 4:
        return 3
    counts = collections.Counter(label for _, label in known)
    if counts[1] * 4 >= len(known) * 3:
        return 1
    if counts[0] * 4 < len(known) * 3:
        return 3
    digits = sum(48 <= c <= 57 for c in content)
    letters = sum(65 <= c <= 90 or 97 <= c <= 122 for c in content)
    math = sum(c in MATH for c in content)
    return 2 if digits >= 4 and digits + math > letters else 0


def fit_tree(rows, max_depth=4, min_leaf=64):
    """Deterministic class-balanced CART; leaf confidence uses raw counts."""
    training = [row for row in rows if row["label"] < 3]
    if not training or any(row["split"] != "calibration" for row in training):
        raise ValueError("tree fitting requires nonempty calibration-only rows")
    totals = collections.Counter(row["label"] for row in training)
    weights = [1 / totals[k] if totals[k] else 0 for k in range(3)]
    nodes = []

    def counts(subset):
        result = [0, 0, 0]
        for row in subset:
            result[row["label"]] += 1
        return result

    def impurity(raw):
        weighted = [n * w for n, w in zip(raw, weights)]
        total = sum(weighted)
        return total - sum(n * n for n in weighted) / total if total else 0

    def grow(subset, depth):
        raw = counts(subset)
        index = len(nodes)
        nodes.append({"feature": -1, "threshold": 0, "left": 0, "right": 0, "counts": raw})
        if depth == max_depth or len(subset) < 2 * min_leaf or sum(n > 0 for n in raw) < 2:
            return index
        best = None
        parent = impurity(raw)
        for feature in range(len(FEATURES)):
            ordered = sorted(subset, key=lambda row: row["features"][feature])
            left = [0, 0, 0]
            for position, row in enumerate(ordered[:-1], 1):
                left[row["label"]] += 1
                cut = row["features"][feature]
                if position < min_leaf or len(ordered) - position < min_leaf:
                    continue
                if cut == ordered[position]["features"][feature]:
                    continue
                right = [a - b for a, b in zip(raw, left)]
                gain = parent - impurity(left) - impurity(right)
                if gain > 1e-12 and (best is None or gain > best[0] + 1e-12):
                    best = gain, feature, cut
        if best is not None:
            _, feature, cut = best
            left = grow([r for r in subset if r["features"][feature] <= cut], depth + 1)
            right = grow([r for r in subset if r["features"][feature] > cut], depth + 1)
            nodes[index].update(feature=feature, threshold=cut, left=left, right=right)
        return index

    grow(training, 0)
    return {
        "features": FEATURES, "nodes": nodes, "max_depth": max_depth, "min_leaf": min_leaf,
        "confidence_fraction": [4, 5],
        "training_groups": sorted({row["group"] for row in training}),
        "training_class_counts": [totals[k] for k in range(3)],
    }


def predict(tree, features):
    node = tree["nodes"][0]
    while node["feature"] >= 0:
        node = tree["nodes"][node["left"] if features[node["feature"]] <= node["threshold"] else node["right"]]
    counts = node["counts"]
    best = max(range(3), key=lambda kind: counts[kind])
    return best if counts[best] * 5 >= sum(counts) * 4 else 3


def export_rust(tree):
    rows = []
    for node in tree["nodes"]:
        rows.append(
            "Node { feature: %d, threshold: %d, left: %d, right: %d, counts: [%s] }"
            % (node["feature"], node["threshold"], node["left"], node["right"],
               ", ".join(map(str, node["counts"])))
        )
    return "const TREE: &[Node] = &[\n    " + ",\n    ".join(rows) + "\n];\n"


class Policy:
    def __init__(self):
        self.k = 3
        self.previous = 3
        self.votes = 0
        self.first = True

    def select(self, kind):
        proposed = (2, 4, 4, 3)[kind]
        self.votes = self.votes + 1 if proposed == self.previous else 1
        self.previous = proposed
        if self.first:
            self.first = False
        elif self.votes >= 2:
            self.k += (proposed > self.k) - (proposed < self.k)
        return self.k
