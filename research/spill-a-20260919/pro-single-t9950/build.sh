#!/usr/bin/env bash
# OWED item 3's owed half (DAY44.md section 1): design T's reading on a 9950X-class host, outside any hold, on the box's
# clone of this lane (/root/wt-a). First the host class: `lscpu`'s model name must name a Ryzen 9 9950X-class part
# (Zen 5 desktop, 16 cores, 32 threads); any other host writes HOST-NOT-9950X-CLASS and builds nothing (the reading
# stays owed). Then, from ONE clone, on the crates of <g4_sha> (the lane before S2's code: G4 and T, equal to main):
# ft (as built, design T), f1 (plus rtx5090-day39/f1-g4.patch, design F at one fill thread), hk (plus
# rtx5090-day39/hk-revert-tip.patch, the day-32 helper fill with design K), the fill probe (day39-fill-survey), each
# patched build's tree checked back at the tip. usage: build.sh <tip_sha> <g4_sha>
set -uo pipefail
R=/root/spill-receipts/a-t9950
export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
mkdir -p "$R/bins/ft" "$R/bins/f1" "$R/bins/hk"
L=$R/build-steps.log
{ lscpu | grep -E '^(Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz)'; free -g | head -2; nproc; } > "$R/host-shape.txt" 2>&1
if ! grep -qiE 'Model name:.*Ryzen 9 9950X' "$R/host-shape.txt"; then
  echo "HOST-NOT-9950X-CLASS $(grep -h 'Model name' "$R/host-shape.txt")" | tee "$R/HOST-NOT-9950X-CLASS"
  echo "rc=3 (host class)" >> "$R/build.log"; exit 3
fi
cd /root/wt-a || exit 1
TIP=$(git rev-parse HEAD)
[ "$TIP" = "$(git rev-parse "$1")" ] || { echo "rc=2 (the clone is not at the tip)" >> "$R/build.log"; exit 2; }
echo "$TIP" > "$R/tree-tip.sha"; echo "$2" > "$R/tree-arms.sha"
clean() { [ -z "$(git status --porcelain --untracked-files=no)" ] && [ "$(git rev-parse HEAD)" = "$TIP" ]; }
back() { git checkout -q "$TIP" -- crates docs tools; git reset -q; git clean -q -fd crates; clean; }
for arm in ft f1 hk; do
  echo "== $arm" >> "$L"
  git diff --binary "$TIP" "$2" -- crates | git apply >> "$L" 2>&1 || { back; echo "rc=2 ($arm crates)" >> "$R/build.log"; exit 2; }
  case $arm in
    f1) git apply research/spill-a-20260919/rtx5090-day39/f1-g4.patch >> "$L" 2>&1 || { back; echo "rc=2 (f1 patch)" >> "$R/build.log"; exit 2; } ;;
    hk) git apply -3 research/spill-a-20260919/rtx5090-day39/hk-revert-tip.patch >> "$L" 2>&1 || { back; echo "rc=2 (hk patch)" >> "$R/build.log"; exit 2; } ;;
  esac
  nice -n 5 cargo build --release -p memra-server >> "$L" 2>&1; brc=$?
  cp target/release/memra-server "$R/bins/$arm/memra-server"
  if [ $arm = ft ] && [ $brc -eq 0 ]; then
    # ft's test binaries (the unit cells of (c)), located from cargo's own artifact messages.
    for pkg in memra-server:memra_server memra-engine:memra_engine; do
      exe=$(nice -n 5 cargo test -p "${pkg%%:*}" --lib --no-run --message-format=json 2>>"$L" | python3 -c 'import sys,json
name=sys.argv[1]
for l in sys.stdin:
    try: m=json.loads(l)
    except Exception: continue
    if m.get("reason")=="compiler-artifact" and m.get("executable") and m["target"]["name"]==name: print(m["executable"])' "${pkg##*:}" | tail -1)
      [ -n "$exe" ] && cp "$exe" "$R/bins/ft/${pkg%%:*}-tests" || brc=1
    done
  fi
  back || { echo "rc=2 (tree after $arm)" >> "$R/build.log"; exit 2; }
  [ $brc -eq 0 ] || { echo "rc=1 ($arm)" >> "$R/build.log"; exit 1; }
done
(cd research/spill-a-20260919/day39-fill-survey && nice -n 5 cargo build --release --target-dir "$R/fill-target") >> "$L" 2>&1 || { echo "rc=1 (fill probe)" >> "$R/build.log"; exit 1; }
git checkout -q -- research/spill-a-20260919/day39-fill-survey 2>/dev/null; clean || { echo "rc=2 (tree at the end)" >> "$R/build.log"; exit 2; }
sha256sum "$R"/bins/*/memra-server "$R"/bins/ft/*-tests "$R/fill-target/release/day39-fill-survey" | tee "$R/binaries.sha256"
echo "rc=0" >> "$R/build.log"
