#!/bin/bash -e

set -x

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <test_dir> <llvm_dir>"
  exit 1
fi

test_dir="$1"
SRC_DIR="$(dirname $0)/../../test/kernel"
LLVM_DIR="$2"

mkdir -p "$test_dir"

cmake -GNinja \
  "-H$SRC_DIR" \
  "-B$test_dir" \
  "-DMEGCC_COMPILER_KERNEL_MLIR_AUTO=ON"  \
  "-DMEGCC_INSTALLED_MLIR_DIR=$LLVM_DIR/lib/cmake"

cmake --build "$test_dir" -j$(nproc)  --target megcc_test_run

cd "$test_dir" && ./megcc_test_run
