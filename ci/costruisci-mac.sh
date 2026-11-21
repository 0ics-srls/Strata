#!/usr/bin/env bash
# 0ics: the Mac Studio build of the Metal engine, one script for the release (.github/workflows/rilascio-mac-0ics.yml) and
# for ubuntu-ai's esegui on the Mac. The steps are the release's, in its order: the pins, metal-cpp, ggml (llama.cpp) at the
# pinned commit, cmake, the engine starts, BUILD.json, the engine package (packaging/package_engine.py) and the Terminal
# archive (packaging/package_terminal.py), SHA256SUMS.
#   ci/costruisci-mac.sh <out> <cache>
#     <out>    build-metal/, pkg/ and dist/ go here; it must not exist (nothing is ever deleted: remove it by hand)
#     <cache>  metal-cpp and llama.cpp at their pins, downloaded only when missing
#   STRATA_SOURCE_COMMIT  the commit of the engine's sources (required: on the Mac the tree has no .git)
#   STRATA_TAG_COMMIT     the tagged commit (default: STRATA_SOURCE_COMMIT)
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1
[ $# -eq 2 ] || { echo "usage: ci/costruisci-mac.sh <out> <cache>" >&2; exit 2; }
out="$1"; cache="$2"
: "${STRATA_SOURCE_COMMIT:?the commit of the engine sources is required}"
tag_commit="${STRATA_TAG_COMMIT:-$STRATA_SOURCE_COMMIT}"
[ ! -e "$out" ] || { echo "costruisci-mac: $out exists already: remove it by hand" >&2; exit 1; }
cd "$(dirname "$0")/.."

# the pins, from where they live
versione=$(sed -n 's/^project(strata VERSION \([0-9.]*\).*/\1/p' CMakeLists.txt)
llama=$(sed -n 's/^LLAMA_CPP_COMMIT = "\([0-9a-f]*\)"$/\1/p' setup.py)
url=$(sed -n 's/^METAL_CPP_URL = "\(.*\)"$/\1/p' metal/setup_mac.py)
sha=$(sed -n 's/^METAL_CPP_SHA256 = "\([0-9a-f]*\)"$/\1/p' metal/setup_mac.py)
for v in "$versione" "$llama" "$url" "$sha"; do [ -n "$v" ] || { echo "costruisci-mac: a pin is missing" >&2; exit 1; }; done

mkdir -p "$cache" "$out/dist"
# metal-cpp, checked against its pinned sha256
mcpp="$cache/metal-cpp-$sha"
if [ ! -d "$mcpp/metal-cpp" ]; then
  curl -fsSL --max-time 300 "$url" -o "$cache/metal-cpp-$sha.zip"
  echo "$sha  $cache/metal-cpp-$sha.zip" | shasum -a 256 -c -
  unzip -q "$cache/metal-cpp-$sha.zip" -d "$mcpp"
fi
# llama.cpp (ggml) at the pinned commit
ggml="$cache/llama.cpp-$llama"
if [ ! -d "$ggml/.git" ]; then
  mkdir -p "$ggml"
  git -C "$ggml" init -q
  git -C "$ggml" remote add origin https://github.com/ggml-org/llama.cpp.git
  git -C "$ggml" fetch -q --depth 1 origin "$llama"
  git -C "$ggml" checkout -q FETCH_HEAD
fi

cmake -S metal -B "$out/build-metal" -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DMETAL_CPP_DIR="$mcpp/metal-cpp" -DSTRATA_GGML_DIR="$ggml"
cmake --build "$out/build-metal" --target strata
"$out/build-metal/strata" --help > /dev/null

jq -n --arg version "$versione" --arg commit "$STRATA_SOURCE_COMMIT" --arg tag_commit "$tag_commit" \
  --arg ggml "$llama" --arg metal_cpp_sha256 "$sha" \
  '{version: $version, commit: $commit, tag_commit: $tag_commit, ggml: $ggml,
    metal_cpp_sha256: $metal_cpp_sha256, backend: "metal"}' \
  > "$out/build-metal/BUILD.json"
cat "$out/build-metal/BUILD.json"

# the engine package: the engine compiles its Metal sources when it starts and finds them beside itself
# (metal/src/resource_path.h): metal/kernels, metal/engine-msl, metal/ggml-common.h, metal/data. Text only: the Mac's Orca
# reads no pictures
stage="$out/pkg/stage"
mkdir -p "$stage/metal/kernels" "$stage/metal/engine-msl" "$stage/metal/data"
cp "$out/build-metal/strata" "$out/build-metal/BUILD.json" "$stage/"
cp metal/kernels/*.metal "$stage/metal/kernels/"
cp metal/src/engine/msl/*.metal "$stage/metal/engine-msl/"
cp third_party/ggml/ggml-common.h "$stage/metal/"
cp data/expert-profile.bin data/draft_vocab.bin "$stage/metal/data/"
python3 packaging/package_engine.py --src "$stage" --backend metal --version "$versione" \
  --platform macos-arm64 --arch apple-gpu --build-id "${STRATA_SOURCE_COMMIT:0:12}" --no-picture-reader \
  --include 'metal/**' --out "$out/pkg/engine"
python3 packaging/package_engine.py --verify "$out/pkg/engine/$versione-macos-arm64-metal"
tar -czf "$out/dist/strata-metal-$versione-macos-arm64.tar.gz" -C "$out/pkg/engine" "$versione-macos-arm64-metal"

# the Terminal archive
python3 packaging/package_terminal.py --target darwin-arm64 --no-installer --version "$versione" \
  --commit "$STRATA_SOURCE_COMMIT" --out "$out/dist"

(cd "$out/dist" && shasum -a 256 -- *.tar.gz *.manifest.json) > "$out/dist/SHA256SUMS"
cat "$out/dist/SHA256SUMS"
