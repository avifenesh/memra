# Compiled-input reach, 2026-10-03

Two actual CPU compiler omissions are repaired. Conditional module roots and split
concat includes through symlink parents changed the independently compiled programs
from BEFORE to AFTER. The baseline planner omitted server checks; the candidate selects
full CPU checks. Direct module attributes and unsplit includes remain selected.

PROOF.json records the four two-commit programs, eight real compiler/execution outputs,
changed input paths, source/binary/helper/bundle/compiler hashes and both plans. The four
fixture Git bundles retain their input states. replay.py recreates the builds without
model weights or a GPU. B's complete runtime_target helper is preserved byte-for-byte,
and now checks the complete compiled include before parent cancellation. Conditional
paths expand without guessing cfg truth. Lexical example/comment controls stay scoped.

All105 CPU planner/coverage controls passed. The workspace registry accounts for14
packages. The four new grouped planner tests cover conditional/nested/unknown/inactive
paths, non-reader examples, unterminated attributes and complete split include aliases.
The final CI floor is105 at this base. CI and review still gate merge.

This is CPU dependency/validation tooling. Native selectors remain in shadow. No native
math, emitted runtime program, compiler/build defaults, artifacts/defaults, tolerance,
required native gates or support states change. Native qualification=false; GPU work0.
