#!/bin/bash
# REAL corpus for the 1M-context demonstration: public-domain Project Gutenberg prose,
# concatenated in a fixed order and banked by sha256 so every rung of the ladder is a
# prefix of the same immutable file. NOT synthetic repetition (cell charter: a repeated
# unit both compresses in the tokenizer and is exactly the shape the greedy-loop artifact
# law warns about).
#
# ~7.2 MB of prose comfortably covers ~1.05M glm4-tokenizer tokens at the measured
# chars/token ratio; the probe slices by characters and the server's usage.prompt_tokens
# is the honest count.
set -eu
D=${1:-$HOME/lane-1mdemo-vast-20260829}
mkdir -p "$D"
cd "$D"
# id: title (all Project Gutenberg plain-text UTF-8)
#  2600 War and Peace, Tolstoy            (~3.2 MB)
#  1184 The Count of Monte Cristo, Dumas  (~2.7 MB)
#  2701 Moby-Dick, Melville               (~1.2 MB)
#   145 Middlemarch, Eliot                (~1.8 MB)
for id in 2600 1184 2701 145; do
  f=pg$id.txt
  [ -s "$f" ] || curl -fsSL -o "$f" "https://www.gutenberg.org/cache/epub/$id/pg$id.txt"
done
cat pg2600.txt pg1184.txt pg2701.txt pg145.txt > corpus-1m.txt
wc -c corpus-1m.txt
sha256sum corpus-1m.txt | tee corpus-1m.sha256
