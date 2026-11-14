#!/bin/zsh
# Builds the inference engines from Bemly git submodules under vendor/:
#   vendor/llama.cpp              -> build-static/bin (llama-server, …)
#   vendor/audio.cpp              -> build-static/bin (audiocpp_*)
#   vendor/stable-diffusion.cpp   -> build-static/bin (sd-cli, …)
#
# No patch series: each submodule tip already carries the AMD Metal stack
# (ToshGEMM + FA-RDNA2) and the ToshLLM engine bits. Bump by updating the
# submodule commit (see scripts/bump-engine-submodules.sh / bump-engines.yml).
#
# Usage:
#   ./scripts/build-engines.sh                  # host architecture
#   ARCH=x86_64 ./scripts/build-engines.sh      # cross-compile (CI on arm64 runners)
#   ARCH=universal ./scripts/build-engines.sh   # x86_64 + arm64 fat binaries (experimental)
#   SKIP_LLAMA=1 SKIP_AUDIO=1 SKIP_IMAGE=1 …
set -e
cd "$(dirname "$0")/.."
ROOT="$PWD"

ARCH="${ARCH:-$(uname -m)}"
DEPLOYMENT_TARGET="14.0"        # same floor as the app (Package.swift)
if [ "$ARCH" = "universal" ]; then
    for slice in x86_64 arm64; do
        ARCH="$slice" "$0"
        for dir in vendor/llama.cpp vendor/audio.cpp; do
            [ -d "$dir/build-static/bin" ] || continue
            for tool in llama-server llama-bench llama-perplexity audiocpp_cli audiocpp_server; do
                mv "$dir/build-static/bin/$tool" "$dir/build-static/bin/$tool.$slice" 2>/dev/null || true
            done
        done
    done
    for dir in vendor/llama.cpp vendor/audio.cpp; do
        [ -d "$dir/build-static/bin" ] || continue
        for tool in llama-server llama-bench llama-perplexity audiocpp_cli audiocpp_server; do
            if [ -f "$dir/build-static/bin/$tool.x86_64" ] && [ -f "$dir/build-static/bin/$tool.arm64" ]; then
                lipo -create -output "$dir/build-static/bin/$tool" \
                    "$dir/build-static/bin/$tool.x86_64" "$dir/build-static/bin/$tool.arm64"
                rm "$dir/build-static/bin/$tool".{x86_64,arm64}
            fi
        done
    done
    echo "universal engines ready"
    exit 0
fi

# Ensure submodules are present (CI checkout --recurse-submodules, or first local build).
if [ ! -e vendor/llama.cpp/.git ] || [ ! -e vendor/audio.cpp/.git ] || [ ! -e vendor/stable-diffusion.cpp/.git ]; then
    echo "initializing engine submodules…"
    git submodule update --init --depth 1 vendor/llama.cpp vendor/audio.cpp vendor/stable-diffusion.cpp
fi

if [ -z "${DEVELOPER_DIR:-}" ] && [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
fi
if ! xcrun metal -v >/dev/null 2>&1 || ! xcrun -f metallib >/dev/null 2>&1; then
    xcodebuild -downloadComponent MetalToolchain >/dev/null 2>&1 || true
fi
resolve_metal_toolchain() {
    xcrun metal -v >/dev/null 2>&1 || return 1
    METAL_COMPILER="$(xcrun -f metal 2>/dev/null)" || return 1
    METALLIB_COMPILER="$(xcrun -f metallib 2>/dev/null)" || return 1
    [ -x "$METAL_COMPILER" ] && [ -x "$METALLIB_COMPILER" ]
}

if resolve_metal_toolchain; then
    METAL_PRECOMPILE=1
    echo "Metal compiler available — will precompile default.metallib (fast model load)"
else
    METAL_PRECOMPILE=0
    echo "Metal compiler unavailable — embedded source only (slower first load)"
fi

TOSH_VERSION="$(tr -d '[:space:]' < "$ROOT/VERSION")"

retry_git() {
    local attempt=1
    local max_attempts=4
    local delay
    while ! "$@"; do
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: git operation failed after $max_attempts attempts: $*" >&2
            return 1
        fi
        delay=$((attempt * 10))
        echo "Git network operation failed (attempt $attempt/$max_attempts); retrying in ${delay}s..." >&2
        sleep "$delay"
        attempt=$((attempt + 1))
    done
}

reset_stale_cmake_cache() {
    local cache="$1/CMakeCache.txt" key cached
    [ -f "$cache" ] || return 0
    for key in CMAKE_OSX_SYSROOT CMAKE_C_COMPILER CMAKE_CXX_COMPILER; do
        cached=$(sed -n "s/^$key:[A-Z]*=//p" "$cache")
        if [ -n "$cached" ] && [ "${cached#/}" != "$cached" ] && [ ! -e "$cached" ]; then
            echo "stale cmake cache in $1 ($key -> $cached); reconfiguring"
            rm -rf "$cache" "$1/CMakeFiles"
            return 0
        fi
    done
}

CMAKE_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DBUILD_SHARED_LIBS=OFF
    -DGGML_METAL=ON
    -DGGML_METAL_EMBED_LIBRARY=ON
    -DTOSH_ENABLE_DYNAMIC_MOE=ON
    -DGGML_NATIVE=OFF
    -DCMAKE_OSX_ARCHITECTURES="$ARCH"
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$DEPLOYMENT_TARGET"
    -DCMAKE_CXX_FLAGS="-DTOSH_VERSION=$TOSH_VERSION"
    -DLLAMA_OPENSSL=OFF
)

ISA_FLAGS=()
if [ "$ARCH" = "x86_64" ]; then
    if [ -z "${TOSH_NO_AVX2:-}" ]; then
        ISA_FLAGS=(-DGGML_SSE42=ON -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON
                   -DGGML_F16C=ON -DGGML_BMI2=ON -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF)
    else
        ISA_FLAGS=(-DGGML_SSE42=ON -DGGML_AVX=OFF -DGGML_AVX2=OFF -DGGML_FMA=OFF
                   -DGGML_F16C=OFF -DGGML_BMI2=OFF -DGGML_AVX_VNNI=OFF -DGGML_AVX512=OFF)
    fi
    CMAKE_FLAGS+=("${ISA_FLAGS[@]}")
fi

build_llama_engine() {
    local vendor="vendor/llama.cpp"
    cd "$ROOT/$vendor"
    echo "building llama @ $(git rev-parse --short HEAD) (FA/ToshGEMM from Bemly/llama.cpp)"

    reset_stale_cmake_cache build-static
    cmake -B build-static "${CMAKE_FLAGS[@]}"
    cmake --build build-static --config Release -j "${JOBS:-$(sysctl -n hw.ncpu)}" \
        -t llama-server llama-bench llama-perplexity test-backend-ops

    local embed_dir="build-static/ggml/src/ggml-metal/autogenerated"
    if [ "$METAL_PRECOMPILE" = 1 ] && [ -n "$(echo $embed_dir/ggml-metal-embed-*.metal(N))" ]; then
        local kernels="build-static/bin/kernels"
        rm -rf "$kernels"
        mkdir -p "$kernels"
        local ok=1
        if resolve_metal_toolchain; then
            for src in $embed_dir/ggml-metal-embed-*.metal; do
                local name="${${src:t:r}#ggml-metal-embed-}"
                "$METAL_COMPILER" -O3 -mmacosx-version-min=14.0 -std=metal3.1 \
                    -DGGML_METAL_EMBED_LIBRARY -DGGML_METAL_HAS_BF16 \
                    -I ggml/src -I ggml/src/ggml-metal \
                    -c "$src" -o "$kernels/$name.air" &&
                "$METALLIB_COMPILER" "$kernels/$name.air" -o "$kernels/$name.metallib" || { ok=0; break; }
                echo "$name $(shasum -a 256 "$src" | cut -d' ' -f1)" >> "$kernels/fingerprint"
            done
        else
            ok=0
        fi
        rm -f "$kernels"/*.air
        if [ "$ok" = 1 ]; then
            echo "precompiled $(ls "$kernels" | wc -l | tr -d ' ') metal libraries ($(du -sh "$kernels" | cut -f1))"
        else
            rm -rf "$kernels"
            echo "WARNING: metallib precompile failed — falling back to runtime compile"
            METAL_PRECOMPILE=0
        fi
    fi
    echo "engine ready at $PWD/build-static/bin (arch: $ARCH)"
    cd "$ROOT"
}

for engine_script in "$ROOT"/scripts/engines/*.sh(N); do
    source "$engine_script"
done

if [ -z "$SKIP_LLAMA" ]; then
    build_llama_engine
fi
if [ -z "$SKIP_AUDIO" ]; then
    build_audio_engine
fi
if [ -z "$SKIP_IMAGE" ]; then
    build_image_engine
fi
