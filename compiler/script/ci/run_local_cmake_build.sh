#!/bin/bash -e

set -x

if [[ $# -ne 2 ]] ; then
  echo "Usage: $0 <test_dir> <path/to/llvm-project>"
  exit 1
fi

test_dir="$(readlink -f $1)"
SRC_DIR="$(dirname $(readlink -f $0))/../.."
LLVM_DIR="$(readlink -f $2)"
LIT_PATH="${LIT_PATH:-$(which lit)}"

if [ -z "$LIT_PATH" ] ; then
    echo "cannot find llvm-lit in system, please set llvm-lit path to LIT_PATH"
    exit 1
fi

mkdir -p "$test_dir"

cmake -GNinja \
  "-H$SRC_DIR" \
  "-B$test_dir" \
  -DMEGCC_INSTALLED_MLIR_DIR="$LLVM_DIR/lib/cmake" \
  -DLLVM_EXTERNAL_LIT="$LIT_PATH"

cmake --build "$test_dir" -j$(nproc)
