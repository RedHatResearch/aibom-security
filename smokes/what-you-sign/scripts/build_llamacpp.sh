#!/usr/bin/env bash
# Reproducible builder for the what-you-sign smoke's llama.cpp variants
# (smokes/what-you-sign/, arXiv:2610.19498 · #26).
#
# Builds, at one pinned commit, the three llama-quantize variants the smoke
# compares (config.py:LLAMA_CPP_COMMIT / B9222_COMMIT):
#
#   quantize-default     arm64 / Apple clang, default FP contraction,
#                        native CPU features (GGML_NATIVE default ON)
#   quantize-nocontract  arm64 / Apple clang, PR #25353 file-scoped
#                        -ffp-contract=off on ggml-quants.c (patch below)
#   quantize-x86-default x86_64 under Rosetta 2, baseline SSE2
#                        (GGML_NATIVE=OFF, FMA/AVX/F16C/SSE42/BMI2 all OFF)
#
# All three share one CPU-only link profile: GGML_METAL/ACCELERATE/OPENMP/
# BLAS=OFF, LLAMA_OPENSSL=OFF (Apple clang lacks -fopenmp; the libomp
# arm64-only link issue is documented in the build session).
#
# Per-variant object-level assertions (llvm-objdump on ggml-quants.c.o):
#   nocontract: 0 FMA instructions (contraction really off)
#   x86:        0 FMA + 0 ymm registers (SSE2 baseline = uncontracted)
#
# Usage:
#   scripts/build_llamacpp.sh SRC_DIR OUT_DIR [--commit SHA] [--b9222] [--force]
#
#   SRC_DIR   clone/cache dir for the llama.cpp sources (created if absent;
#             the nocontract variant builds from a git worktree SRC_DIR-nocontract
#             so the CMake patch never touches the default tree)
#   OUT_DIR   receives quantize-default/ quantize-nocontract/
#             quantize-x86-default/ with llama-quantize inside
#
#   --commit SHA   pin to commit SHA (40 hex). Default: the smoke's tip pin
#                  (config.py LLAMA_CPP_COMMIT). --b9222 defaults to the
#                  paper's release tag pin (config.py B9222_COMMIT,
#                  9a532ae4bab1b164052ce60a738f78538b421c66 = b9222).
#   --b9222        build the §IV Table III variants at the b9222 pin
#   --force        rebuild even if the output binaries already exist
#
# Idempotent: existing OUT_DIR binaries are verified (object assertions on
# the build trees), not rebuilt, unless --force. Build trees live under
# SRC_DIR (build-default, build-x86-default) and the nocontract worktree
# (build-nocontract); the smoke locates the b9222 binaries either via
# --b9222-builds-root (this script's OUT_DIR layout) or by globbing the
# .scratch source trees.

set -euo pipefail

REPO_URL="https://github.com/ggml-org/llama.cpp"
# Pins mirror smokes/what-you-sign/config.py (single source: change both).
TIP_COMMIT="d8d92973c8c21bb4dd5d592f0fec26fe2cd1e654"
B9222_COMMIT="9a532ae4bab1b164052ce60a738f78538b421c66"
LLAMA_QUANTIZE_REL="bin/llama-quantize"
QUANTS_OBJ_REL="ggml/src/CMakeFiles/ggml-base.dir/ggml-quants.c.o"

# PR #25353 (file-scoped -ffp-contract=off on ggml-quants.c), applied to
# ggml/src/CMakeLists.txt after the add_library(ggml-base ...) block.
nocontract_patch() {
    local cmake="ggml/src/CMakeLists.txt"
    grep -q "ffp-contract=off" "$cmake" && return 0
    python3 - "$cmake" <<'PYEOF'
import sys

path = sys.argv[1]
with open(path) as f:
    text = f.read()
anchor = "set_target_properties(ggml-base PROPERTIES"
block = """# ggml-quants.c contains the reference quantization kernels used when producing
# quantized model files (ggml_quantize_chunk). The k-quant / i-quant scale
# search loops in this file are sensitive to floating point contraction (FMA):
# when the compiler is allowed to contract a*b+c, the selected scales - and
# therefore the quantized output bits - differ between compilers and
# architectures. Disabling contraction for this single translation unit makes
# quantization results bit-identical across compilers and architectures.
# A '#pragma STDC FP_CONTRACT OFF' in the source is not sufficient:
# GCC does not implement the pragma and ignores it with a warning.
# (PR #25353)
if (CMAKE_C_COMPILER_ID MATCHES "GNU" OR CMAKE_C_COMPILER_ID MATCHES "Clang")
    set_source_files_properties(ggml-quants.c PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
endif()

"""
if anchor not in text:
    sys.exit(f"anchor not found in {path}: {anchor!r}")
text = text.replace(anchor, block + anchor, 1)
with open(path, "w") as f:
    f.write(text)
PYEOF
}

die() { echo "build_llamacpp.sh: error: $*" >&2; exit 1; }
log() { echo "build_llamacpp.sh: $*" >&2; }

usage() {
    sed -n '2,44p' "$0" | sed 's/^# \{0,1\}//' >&2
    exit 0
}

[[ $# -ge 2 ]] || usage
SRC_DIR=$1
OUT_DIR=$2
shift 2

COMMIT=""
MODE="tip"
FORCE=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --commit) [[ $# -ge 2 ]] || die "--commit needs a SHA"; COMMIT=$2; shift 2 ;;
        --b9222) MODE="b9222"; shift ;;
        --force) FORCE=1; shift ;;
        --help|-h) usage ;;
        *) die "unknown argument: $1" ;;
    esac
done

if [[ -n "$COMMIT" ]]; then
    [[ "$COMMIT" =~ ^[0-9a-fA-F]{40}$ ]] || die "--commit must be a 40-hex SHA, got: $COMMIT"
elif [[ "$MODE" == "b9222" ]]; then
    COMMIT=$B9222_COMMIT
else
    COMMIT=$TIP_COMMIT
fi
COMMIT=$(printf '%s' "$COMMIT" | tr 'A-F' 'a-f')

# Resolve everything to absolute paths up front — the builds run after
# pushd into the source tree, where relative paths would land in the wrong
# place.
SRC_DIR=$(cd "$(dirname "$SRC_DIR")" && pwd)/$(basename "$SRC_DIR")
OUT_DIR=$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)
NOCONTRACT_DIR="${SRC_DIR}-nocontract"
DEFAULT_BUILD="$SRC_DIR/build-default"
NOCONTRACT_BUILD="$NOCONTRACT_DIR/build-nocontract"
X86_BUILD="$SRC_DIR/build-x86-default"

OBJDUMP=""
for cand in /Library/Developer/CommandLineTools/usr/bin/llvm-objdump llvm-objdump; do
    if command -v "$cand" >/dev/null 2>&1; then OBJDUMP=$cand; break; fi
done
[[ -n "$OBJDUMP" ]] || die "llvm-objdump not found (install Xcode Command Line Tools)"

count_fma() { # stdin disassembly -> number of FMA-family instructions
    awk '$3 ~ /^(fmadd|fmsub|fnmadd|fnmsub|fmla|fmls|vfma|vfms|vfmadd|vfmsub|vfnmadd|vfnmsub|fmaf|fmaddh|fmlal|fmlsl)/ {n++} END {print n+0}'
}
count_ymm() { awk '/ymm/ {n++} END {print n+0}'; }

# assert_object VARIANT OBJECT_PATH — object-level FMA/ymm assertions.
assert_object() {
    local variant=$1 obj=$2
    [[ -f "$obj" ]] || die "$variant: ggml-quants.c.o not found: $obj
  (binary exists but its build tree does not — rebuild with --force)"
    local dis fma ymm
    dis=$("$OBJDUMP" -d "$obj")
    fma=$(printf '%s' "$dis" | count_fma)
    ymm=$(printf '%s' "$dis" | count_ymm)
    case "$variant" in
        nocontract)
            [[ "$fma" == 0 ]] || die "$variant: expected 0 FMA in ggml-quants.c.o, found $fma"
            ;;
        x86)
            [[ "$fma" == 0 ]] || die "$variant: expected 0 FMA in ggml-quants.c.o, found $fma"
            [[ "$ymm" == 0 ]] || die "$variant: expected 0 ymm in ggml-quants.c.o, found $ymm"
            ;;
        default) ;;
    esac
    log "$variant: object assertions passed (FMA=$fma ymm=$ymm)"
}

# cmake_build BUILD_DIR EXTRA_CMAKE_FLAGS... — configures and builds only
# the llama-quantize target with the shared CPU-only link profile.
cmake_build() {
    local build_dir=$1
    shift
    rm -rf "$build_dir"
    cmake -S . -B "$build_dir" \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_METAL=OFF \
        -DGGML_ACCELERATE=OFF \
        -DGGML_OPENMP=OFF \
        -DGGML_BLAS=OFF \
        -DGGML_CCACHE=OFF \
        -DGGML_BUILD_EXAMPLES=OFF \
        -DGGML_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_EXAMPLES=OFF \
        -DLLAMA_BUILD_TESTS=OFF \
        -DLLAMA_BUILD_TOOLS=ON \
        -DLLAMA_OPENSSL=OFF \
        -DLLAMA_CURL=OFF \
        "$@" >/dev/null 2>&1
    cmake --build "$build_dir" --target llama-quantize -j "$(sysctl -n hw.ncpu)" \
        >/dev/null 2>&1
}

install_binary() { # BUILD_DIR VARIANT_DIRNAME
    local build_dir=$1 vdir=$2
    mkdir -p "$OUT_DIR/$vdir"
    cp "$build_dir/$LLAMA_QUANTIZE_REL" "$OUT_DIR/$vdir/llama-quantize"
}

# ---------------------------------------------------------------------------
# 1. Source tree at the pinned commit.
# ---------------------------------------------------------------------------
if ! git -C "$SRC_DIR" rev-parse --git-dir >/dev/null 2>&1; then
    if [[ -e "$SRC_DIR" && -n $(ls -A "$SRC_DIR" 2>/dev/null) ]]; then
        die "$SRC_DIR exists but is not a git clone/worktree (no .git); pass an empty or clone dir"
    fi
    log "cloning llama.cpp into $SRC_DIR"
    mkdir -p "$(dirname "$SRC_DIR")"
    git clone --filter=blob:none "$REPO_URL" "$SRC_DIR" >&2
fi

# The clone dir itself stays stock at the pinned commit (default + x86
# variants); the nocontract variant builds from a git worktree so the CMake
# patch never leaks into the other variants.
pushd "$SRC_DIR" >/dev/null
if ! git cat-file -e "$COMMIT^{commit}" 2>/dev/null; then
    log "fetching $COMMIT"
    git fetch --quiet origin >&2 || die "git fetch failed for $COMMIT"
fi
git checkout --quiet --detach "$COMMIT" || die "cannot check out $COMMIT in $SRC_DIR"
HEAD_OK=$(git rev-parse HEAD)
[[ "$HEAD_OK" == "$COMMIT" ]] || die "$SRC_DIR is at $HEAD_OK, expected $COMMIT"
log "source at $COMMIT ($MODE)"

# ---------------------------------------------------------------------------
# 2. Variants: verify existing outputs, build the missing ones.
# ---------------------------------------------------------------------------
variant_dir() { # v -> OUT_DIR subdirectory name
    case "$1" in
        x86) echo "quantize-x86-default" ;;
        *) echo "quantize-$1" ;;
    esac
}
variant_build() { # v -> build tree holding its ggml-quants.c.o
    case "$1" in
        default) echo "$DEFAULT_BUILD" ;;
        nocontract) echo "$NOCONTRACT_BUILD" ;;
        x86) echo "$X86_BUILD" ;;
    esac
}

need_default=0 need_nocontract=0 need_x86=0
for v in default nocontract x86; do
    bin="$OUT_DIR/$(variant_dir "$v")/llama-quantize"
    if [[ -x "$bin" && $FORCE -eq 0 ]]; then
        log "$(variant_dir "$v"): exists, verifying (use --force to rebuild)"
        assert_object "$v" "$(variant_build "$v")/$QUANTS_OBJ_REL"
    else
        case "$v" in
            default) need_default=1 ;;
            nocontract) need_nocontract=1 ;;
            x86) need_x86=1 ;;
        esac
    fi
done

if [[ $need_default -eq 1 ]]; then
    # arm64 default: stock tree, clang default FP contraction, native CPU
    # features (GGML_NATIVE left at its default ON, matching the recorded
    # builds — forcing it OFF changes codegen and the smoke's binaries).
    log "building quantize-default (arm64, default contraction)"
    cmake_build "$DEFAULT_BUILD"
    assert_object default "$DEFAULT_BUILD/$QUANTS_OBJ_REL"
    install_binary "$DEFAULT_BUILD" "$(variant_dir default)"
fi

if [[ $need_nocontract -eq 1 ]]; then
    # arm64 nocontract: PR #25353 file-scoped patch, built in a git worktree.
    if [[ ! -d "$NOCONTRACT_DIR/.git" ]]; then
        git worktree add --detach "$NOCONTRACT_DIR" "$COMMIT" >&2
    fi
    pushd "$NOCONTRACT_DIR" >/dev/null
    git checkout --quiet --detach "$COMMIT" || die "cannot check out $COMMIT in $NOCONTRACT_DIR"
    git checkout HEAD -- ggml/src/CMakeLists.txt 2>/dev/null || true
    nocontract_patch
    git diff --stat -- ggml/src/CMakeLists.txt | grep -q "CMakeLists.txt" \
        || die "nocontract patch did not apply to $NOCONTRACT_DIR"
    log "building quantize-nocontract (arm64, PR #25353 -ffp-contract=off)"
    cmake_build "$NOCONTRACT_BUILD"
    assert_object nocontract "$NOCONTRACT_BUILD/$QUANTS_OBJ_REL"
    install_binary "$NOCONTRACT_BUILD" "$(variant_dir nocontract)"
    popd >/dev/null
fi

if [[ $need_x86 -eq 1 ]]; then
    # x86_64 Rosetta baseline: CMAKE_OSX_ARCHITECTURES=x86_64 with every
    # SIMD flag off — baseline SSE2, the paper's uncontracted premise.
    log "building quantize-x86-default (x86_64 Rosetta, baseline SSE2)"
    cmake_build "$X86_BUILD" \
        -DCMAKE_OSX_ARCHITECTURES=x86_64 \
        -DGGML_NATIVE=OFF \
        -DGGML_FMA=OFF -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_F16C=OFF \
        -DGGML_SSE42=OFF -DGGML_BMI2=OFF
    assert_object x86 "$X86_BUILD/$QUANTS_OBJ_REL"
    install_binary "$X86_BUILD" "$(variant_dir x86)"
fi
popd >/dev/null

log "done: $OUT_DIR (commit $COMMIT, mode $MODE)"
