#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
mkdir -p "$scratch/bin" "$scratch/home"

cat > "$scratch/bin/curl" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
out=
url=${!#}
while [ "$#" -gt 0 ]; do
  if [ "$1" = -o ]; then out=$2; shift 2; else shift; fi
done
case "$url" in
  */SHA256SUMS)
    printf '%064d  memra-v0.123.0-linux-x86_64-glibc2.35-sm120a.tar.gz\n' 0 > "$out"
    ;;
  */SHA256SUMS.sigstore.json)
    exit 22
    ;;
  *)
    exit 23
    ;;
esac
SH
chmod +x "$scratch/bin/curl"

set +e
output=$(PATH="$scratch/bin:$PATH" HOME="$scratch/home" MEMRA_VERSION=v0.123.0 \
  MEMRA_CUDA_ARCH=120a sh "$here/tools/install.sh" 2>&1)
rc=$?
set -e
[ "$rc" -ne 0 ] || { echo "installer accepted a release with no signature bundle" >&2; exit 1; }
grep -q 'no signed checksum bundle' <<<"$output" \
  || { echo "installer failed for the wrong reason: $output" >&2; exit 1; }
grep -q -- '--certificate-identity "https://github.com/avifenesh/memra/.github/workflows/release.yml@refs/tags/$VERSION"' \
  "$here/tools/install.sh"
grep -q 'cosign sign-blob.*--bundle' "$here/.github/workflows/release.yml"
echo "install authenticity fixture: missing/unsigned release refused"
