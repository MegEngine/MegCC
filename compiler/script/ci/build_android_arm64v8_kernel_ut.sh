#!/bin/bash -e

set -x

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <test_dir> <path/to/ndk-root>"
  exit 1
fi

test_dir="$(readlink -f $1)"
SRC_DIR="$(dirname $0)/../../test/kernel"
NDK_ROOT="$(readlink -f $2)"

mkdir -p "$test_dir"

cmake -GNinja \
  "-H$SRC_DIR" \
  "-B$test_dir" \
  -DCMAKE_TOOLCHAIN_FILE="$NDK_ROOT/build/cmake/android.toolchain.cmake" \
  -DANDROID_NDK="$NDK_ROOT" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_NATIVE_API_LEVEL=21 \
  -DCMAKE_BUILD_TYPE=Debug \
  -DMEGCC_COMPILER_KERNEL_ENABLE_FP16=ON \
  -DMEGCC_COMPILER_KERNEL_WITH_ASAN=ON

cmake --build "$test_dir" -j$(nproc)  --target megcc_test_run

