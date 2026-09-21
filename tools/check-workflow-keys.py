#!/usr/bin/env python3
"""check-workflow-keys.py: refuse a GitHub workflow file with a duplicate mapping key.

WHY. On 2026-09-21 two PRs (#592, #590) each added a ci.yml job named `portable-suites`. Each
PR's own CI was green (its merge ref predated the other), main took both, and GitHub refused
the file: the push run had ZERO jobs and read "This run likely failed because of a workflow
file issue". Every push and PR after it was red at the workflow level, and no job in ci.yml
could catch it, because ci.yml itself was the file that did not parse. PyYAML's `safe_load`
keeps the LAST duplicate silently, so `python3 -c "import yaml; yaml.safe_load(...)"` passed on
that tree; a loader has to be told to raise.

WHAT. Every `.github/workflows/*.yml` and `*.yaml` is loaded with a SafeLoader whose mapping
constructor raises on the first duplicate key at any depth (jobs, steps, `with:`, `env:`,
`inputs:`). Zero workflow files is a failure too (a moved directory must not read as green).
Exit 0 prints one line per file with its job names; exit 1 names file, line, column and key.

WHERE. tools/hooks/pre-push (no skip switch: a tree whose workflow GitHub cannot parse is
unshippable) and the ci.yml `gates` job, which protects the OTHER workflow files; ci.yml can
only be protected from itself before the push, or by a PR run on a merge ref that is current.
Teeth: tools/test_workflow_keys.sh.
"""

import pathlib
import sys

import yaml


class Duplicate(Exception):
    pass


class StrictLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        if not isinstance(node, yaml.MappingNode):
            raise yaml.constructor.ConstructorError(
                None, None, f"expected a mapping node, but found {node.id}", node.start_mark
            )
        self.flatten_mapping(node)
        seen = {}
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if key in seen:
                first = seen[key]
                raise Duplicate(
                    f"duplicate mapping key {key!r} at line {key_node.start_mark.line + 1} "
                    f"column {key_node.start_mark.column + 1} (first at line {first.line + 1})"
                )
            seen[key] = key_node.start_mark
        return super().construct_mapping(node, deep=deep)


def check(path):
    with path.open("rb") as stream:
        data = yaml.load(stream, Loader=StrictLoader)
    if not isinstance(data, dict):
        raise Duplicate("top level is not a mapping")
    jobs = data.get("jobs")
    if not isinstance(jobs, dict) or not jobs:
        raise Duplicate("no `jobs:` mapping")
    return list(jobs)


def main(argv):
    root = pathlib.Path(argv[1]) if len(argv) > 1 else pathlib.Path(".github/workflows")
    files = sorted(p for p in root.glob("*.y*ml") if p.suffix in (".yml", ".yaml"))
    if not files:
        print(f"check-workflow-keys: FAIL: no workflow files under {root}", file=sys.stderr)
        return 1
    bad = 0
    for path in files:
        try:
            jobs = check(path)
        except (Duplicate, yaml.YAMLError) as error:
            print(f"check-workflow-keys: FAIL: {path}: {error}", file=sys.stderr)
            bad += 1
            continue
        print(f"check-workflow-keys: {path}: jobs {', '.join(jobs)}")
    if bad:
        print(f"check-workflow-keys: FAIL: {bad} of {len(files)} workflow files refused", file=sys.stderr)
        return 1
    print(f"check-workflow-keys: OK: {len(files)} workflow files, no duplicate mapping keys")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
