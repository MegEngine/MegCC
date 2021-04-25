#!/bin/bash -e

set -x

SRC_DIR="$(dirname $0)/../test/"
test_dir=${SRC_DIR}/build

mkdir -p "$test_dir"

cmake -GNinja \
  "-H$SRC_DIR" \
  "-B$test_dir" \
  -DCMAKE_BUILD_TYPE=Debug \
  -DTINYNN_ENABLE_ASAN=1

cmake --build "$test_dir" -j$(nproc)  --target TinyNNTest

cd "$test_dir" && ./TinyNNTest
