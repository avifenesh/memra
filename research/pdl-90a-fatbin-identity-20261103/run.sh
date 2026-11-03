#!/usr/bin/env bash
# Byte-identity check for the PDL entry-wait change (fix(cuda): keep the PDL entry wait on
# portable sm_90a). Builds the kernels, qmatvec and flash_attn fatbins, plus the flash_attn
# kf8vf8 variant, with the flags crates/memra-engine/build.rs passes, at the commit and at its
# parent, for sm_120a, sm_100a and sm_90a. Every build runs from the same relative path, so the
# only input that differs is the source. sm_90a is the positive control: it must differ.
#
# usage: run.sh <repo checkout> <work dir> [commit]   (NVCC=<path> overrides the compiler)
# CPU only: nvcc needs no GPU.
set -euo pipefail
repo=$(cd -- "$1" && pwd)
root=$(mkdir -p -- "$2" && cd -- "$2" && pwd)
fix=${3:-7e8b3673d4}
nvcc=${NVCC:-nvcc}
cd "$root"
for rev in base fix; do
    mkdir -p "$rev/cu" "$rev/out"
    commit=$fix
    [[ $rev == base ]] && commit="$fix^"
    for f in kernels.cu qmatvec.cu flash_attn.cu; do
        git -C "$repo" show "$commit:crates/memra-engine/cu/$f" > "$rev/cu/$f"
    done
done
"$nvcc" --version | tail -1 > nvcc.txt

jobs_file=jobs.txt
: > "$jobs_file"
for rev in base fix; do
    for arch in 120a 100a 90a; do
        extra=""
        [[ $arch == 90a ]] && extra="-DMEMRA_PORTABLE_CUDA=1 -DMEMRA_HOPPER_MMA=1"
        # No trailing blank: `xargs -L 1` joins a line ending in one with the next.
        for unit in kernels qmatvec flash_attn; do
            echo "$rev $arch $unit $unit${extra:+ $extra}" >> "$jobs_file"
        done
        echo "$rev $arch flash_attn flash_attn_kf8vf8${extra:+ $extra} -DMEMRA_KV_KFMT=1 -DMEMRA_KV_VFMT=2" >> "$jobs_file"
    done
done

build_one() {
    local rev=$1 arch=$2 unit=$3 name=$4
    shift 4
    (cd "$root/$rev" && "$nvcc" -gencode "arch=compute_${arch},code=sm_${arch}" -O3 --fatbin \
        "$@" -o "out/${name}.sm_${arch}.fatbin" "cu/${unit}.cu") \
        > "$root/$rev/out/${name}.sm_${arch}.log" 2>&1
    echo "built $rev sm_$arch $name rc=$?"
}
export -f build_one
export root nvcc
# shellcheck disable=SC2016
xargs -P 3 -L 1 bash -c 'build_one "$@"' _ < "$jobs_file"

for arch in 120a 100a 90a; do
    for name in kernels qmatvec flash_attn flash_attn_kf8vf8; do
        b=base/out/$name.sm_$arch.fatbin
        f=fix/out/$name.sm_$arch.fatbin
        if cmp -s "$b" "$f"; then verdict=IDENTICAL; else verdict=DIFFERS; fi
        printf '%s sm_%s %s base=%s fix=%s bytes=%s/%s\n' "$verdict" "$arch" "$name" \
            "$(sha256sum "$b" | cut -d' ' -f1)" "$(sha256sum "$f" | cut -d' ' -f1)" \
            "$(stat -c %s "$b")" "$(stat -c %s "$f")"
    done
done > cmp.txt
cuobjdump=$(dirname -- "$(command -v "$nvcc")")/cuobjdump
for name in kernels qmatvec flash_attn flash_attn_kf8vf8; do
    printf 'sm_90a %s ACQBULK base=%s fix=%s\n' "$name" \
        "$("$cuobjdump" -sass "base/out/$name.sm_90a.fatbin" | grep -c ACQBULK || true)" \
        "$("$cuobjdump" -sass "fix/out/$name.sm_90a.fatbin" | grep -c ACQBULK || true)"
done > acqbulk.txt
cat cmp.txt acqbulk.txt
