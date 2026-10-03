# Compiled-input graph, 2026-10-03

Issue #932, one CPU-only lane. Two independent Rust readers change output when their
physical input changes, while the baseline CPU selector omits server checks.
Conditional module paths have unknown transitive reach. Split concat includes can
hide a symlink if parent components are normalized before physical traversal checks.

Reuse the complete physical traversal helper from #930. Resolve the complete compiled
literal first, then retain every alias until its parent traversal is checked. Unresolved
conditional module path attributes expand CPU checks without guessing cfg truth.
Preserve the unsplit include and direct-module controls, lexical comment/string boundaries,
existing runtime path checks, exact tool-choice receipt inputs and coverage context work.

The replay compiles and executes four independent two-commit fixture bundles. Only the
referenced input differs. BEFORE/AFTER outputs and source/binary/bundle/helper/compiler
identities are retained. Two baseline omissions must become full CPU selections; both
positive controls remain selected. The Python controls cover nested/unknown/inactive
conditional paths and alias traversal. Native selectors remain shadow and every result
has qualification=false. No native math/default/compiler/target/tolerance/support change.
Budget: $0 paid compute, bounded CPU checks, no GPU/new models/production access.
