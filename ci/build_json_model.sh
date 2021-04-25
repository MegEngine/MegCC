#!/bin/bash -e

set -x
echo "build model from json"
date
DUMP_MODE="--arm64"
if [[ $# -lt 3 ]] ; then
  echo "Usage: $0 <compiler_build_dir> <output_dir> <path/to/json> <dumo_mode>"
  exit 1
fi
if [[ $# -ge 4 ]] ; then
  DUMP_MODE=$4
fi

PROJECT_PATH="$(dirname $(readlink -f $0))/.."
cd ${PROJECT_PATH}

# build mgb-to-tinynn and mgb-runner
COMPILER_BUILD_DIR="$(readlink -f $1)"
cmake --build "$COMPILER_BUILD_DIR" -j$(nproc) --target mgb-to-tinynn --target mgb-runner

# export tinynn model and generate kernels
OUTPUT_DIR="$(readlink -f $2)"
JSON_PATH="$(readlink -f $3)"

rm -fr "${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"
$COMPILER_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "--json=$JSON_PATH" "$DUMP_MODE" --dump="$OUTPUT_DIR"

# compile arm model
ARM_OUTPUT_DIR="${OUTPUT_DIR}/"
mkdir -p $ARM_OUTPUT_DIR
$PROJECT_PATH/runtime/scripts/runtime_build.py --cross_build --kernel_dir ${ARM_OUTPUT_DIR} --remove_old_build
if [[ "--arm64v7" == "$DUMP_MODE" ]]; then
  $PROJECT_PATH/runtime/scripts/runtime_build.py --cross_build --kernel_dir ${ARM_OUTPUT_DIR} --remove_old_build --cross_build_target_arch armv7-a
fi
