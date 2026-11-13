#!/usr/bin/env bash
# Runs ON the dev desk (memra#1244). Red arms (tools/redarm.py), each of which must fail the
# tests its docstring names, then green: all seven pass.
# env: RG_ARMS (the arms to run; default all), RG_TREE / RG_TARGET (a tree copy and its target
# dir, so two workers can split the arms), RG_GREEN (0 = skip the green run).
set -u
D=$HOME/.cache/memra-1244
export PATH="$HOME/.cargo/bin:/usr/local/cuda-13.1/bin:$PATH"
export MEMRA_NVCC=/usr/local/cuda-13.1/bin/nvcc MEMRA_CUDA_ARCH=120a
export CARGO_TARGET_DIR=${RG_TARGET:-$D/target}
for v in $(env | grep -o '^MEMRA_[A-Z0-9_]*' | grep -v '^MEMRA_NVCC$\|^MEMRA_CUDA_ARCH$'); do unset "$v"; done
TREE=${RG_TREE:-$D/tree}
cd "$TREE" || exit 1
W=crates/memra-server/src/worker.rs
GREEN=$TREE.worker.green.rs
cp "$W" "$GREEN"
TESTS="spec_round_yield_due_names_the_plain_sessions_behind_the_bursts spec_burst_ends_at_its_first_round_while_a_plain_session_waits spec_round_yield_is_wired spec_round_yield_weighs_interactive_plain_sessions_only plain_prime_runs_only_when_phase_b_primes_the_last_chunk carried_rider_models_follow_one_rule dark_work_keeps_its_burst_cadence_on_yielded_ticks"
ARMS=${RG_ARMS:-parent nohook nodue nowrite noride noridelane nolane nolanewire noheld nooutlook nowindows nochoice carriedA carriedB carriedC nomtp nocadence nocadencewire noreset nocadencerounds}
run() {
  local name=$1
  echo "== $name: $(date -Is)"
  for t in $TESTS; do
    nice -n 15 cargo test --release -p memra-server --lib -j "${RG_J:-12}" "$t" >> "$D/redgreen-$name.log" 2>&1
    echo "$name $t rc=$?"
  done
  grep -E "^test .*(ok|FAILED)$|panicked at|test result|^error" "$D/redgreen-$name.log" | head -n 30
}
for arm in $ARMS; do
  rm -f "$D/redgreen-red-$arm.log"
  python3 "$D/tools/redarm.py" "$arm" "$GREEN" "$W" || { echo "red arm $arm did not apply"; cp "$GREEN" "$W"; exit 1; }
  run "red-$arm"
done
cp "$GREEN" "$W"
if [ "${RG_GREEN:-1}" = 1 ]; then
  rm -f "$D/redgreen-green.log"
  run green
fi
echo "REDGREEN-DONE $(date -Is)"
