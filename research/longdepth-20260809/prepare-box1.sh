#!/usr/bin/env bash
# Build and freeze provenance on box1 without taking the GPU lock.
set -euo pipefail
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH

ROOT=$(git rev-parse --show-toplevel)
LANE=$ROOT/research/longdepth-20260809
RUN_ID=${LONGDEPTH_RUN_ID:?set LONGDEPTH_RUN_ID once for the matrix}
RUN=$LANE/raw/$RUN_ID
MODEL=${MEMRA_STEP37_GGUF:-$HOME/step37/models/step-3.7-flash/IQ4_XS/Step-3.7-flash-IQ4_XS-00001-of-00003.gguf}
DRAFT=${MEMRA_STEP37_DRAFT:-$HOME/step37/models/step-3.7-flash/Step3.7-flash-mtp-Q8_0.gguf}
PARSER_VENV=/tmp/memra-longdepth-parser-$RUN_ID
mkdir -p "$RUN"
cd "$ROOT"

test -f "$MODEL"
test -f "$DRAFT"
python3 -m venv "$PARSER_VENV"
"$PARSER_VENV/bin/python3" -m pip install --disable-pip-version-check --no-input \
  -r "$LANE/parser-requirements.txt" > "$RUN/parser-install.log" 2>&1
"$PARSER_VENV/bin/python3" -m pip freeze > "$RUN/parser-freeze.txt"
"$PARSER_VENV/bin/python3" "$LANE/detect.py" --self-test \
  | tee "$RUN/detector-self-test.log"

{
  hostname
  date -u +%FT%TZ
  uname -a
  git rev-parse HEAD
  git branch --show-current
  git status --short --branch
  rustc --version
  cargo --version
  nvcc --version
  nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader
  stat -c '%n %s bytes' "$MODEL" "$DRAFT" "${MODEL%-00001-of-00003.gguf}-00002-of-00003.gguf" "${MODEL%-00001-of-00003.gguf}-00003-of-00003.gguf"
} > "$RUN/provenance.txt" 2>&1

sha256sum \
  "$MODEL" \
  "${MODEL%-00001-of-00003.gguf}-00002-of-00003.gguf" \
  "${MODEL%-00001-of-00003.gguf}-00003-of-00003.gguf" \
  "$DRAFT" > "$RUN/artifact-sha256.txt"
sha256sum "$LANE/prompt.txt" "$LANE/assistant-prefix.txt" \
  "$LANE/request.py" "$LANE/detect.py" \
  "$LANE/parser-requirements.txt" \
  "$LANE/run-cell-box1.sh" \
  "$ROOT/crates/memra-tokenizer/src/bin/tok_chat_render.rs" > "$RUN/harness-sha256.txt"

cargo build --release -p memra-server --bin memra-server \
  > "$RUN/build-memra-server.log" 2>&1
cargo build --release -p memra-tokenizer --bin tok_span \
  > "$RUN/build-tok-span.log" 2>&1
cargo build --release -p memra-tokenizer --bin tok_chat_render \
  > "$RUN/build-tok-chat-render.log" 2>&1
target/release/tok_chat_render "$MODEL" "$LANE/prompt.txt" low \
  "$LANE/assistant-prefix.txt" "$RUN/rendered-prompt-low.txt" \
  | tee "$RUN/render-prompt.log"
sha256sum "$RUN/rendered-prompt-low.txt" > "$RUN/rendered-prompt-sha256.txt"
sha256sum target/release/memra-server target/release/tok_span \
  target/release/tok_chat_render > "$RUN/binary-sha256.txt"
printf 'prepare complete run=%s commit=%s\n' "$RUN_ID" "$(git rev-parse HEAD)"
