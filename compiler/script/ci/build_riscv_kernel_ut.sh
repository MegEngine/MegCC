#!/bin/bash -e

set -x

if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <test_dir>"
  exit 1
fi

test_dir="$(readlink -f $1)"
SRC_DIR="$(dirname $0)/../../test/kernel"
RVV_TOOLCHAIN_FILE="$(readlink -f $(dirname $0))/../../../runtime/toolchains/riscv64-rvv-linux-gnu.toolchain.cmake"

mkdir -p "$test_dir"

cmake -GNinja \
  "-H$SRC_DIR" \
  "-B$test_dir" \
  -DCMAKE_TOOLCHAIN_FILE="$RVV_TOOLCHAIN_FILE"

cd "$test_dir"
ninja
echo "!! $0 done " 

