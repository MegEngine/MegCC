#!/bin/bash -e

set -x

if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <build_dir>"
  exit 1
fi

BUILD_DIR="$(readlink -f $1)"
cmake --build "$BUILD_DIR" -j$(nproc)  --target megcc-test
