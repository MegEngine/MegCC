#!/bin/bash -e
set -x
echo "start integration test"
date

if [[ $# -lt 7 ]] ; then
  echo "Usage: $0 <compiler_build_dir> <output_dir> <path/to/test_model>  <dump_input_shape_str, like: data=(1,3,224,224):data=(1,3,112,112)>  <test_input_str, like: img_raw=input_0.bin;lm_raw=input_1.bin> <eps> <dump_mode> <gen_armv7_flag:1> <only_dump_arm:0> <arm_extra_dump:None>"
  exit 1
fi
ONLY_DUMP_ARM=0
GEN_ARMV7_FLAG=1
ARM_EXTRA_DUMP=""
if [[ $# -ge 8 ]] ; then
  GEN_ARMV7_FLAG=$8
fi
if [[ $# -ge 9 ]] ; then
  ONLY_DUMP_ARM=$9
fi
if [[ $# -ge 10 ]] ; then
  ARM_EXTRA_DUMP=${10}
fi
PROJECT_PATH="$(dirname $(readlink -f $0))/.."
cd ${PROJECT_PATH}
source ./ci/test_tools.sh
COMPILER_BUILD_DIR="$(readlink -f ${1})"
BUILD_OUTPUT_DIR="$(readlink -f ${2})"

cmake --build "$COMPILER_BUILD_DIR" -j$(nproc) --target mgb-to-tinynn --target mgb-runner

cmake --build "$COMPILER_BUILD_DIR" -j$(nproc) --target hako-to-mgb
# $3 <path/to/test_model>
# $4 <dump_input_shape_str>
# $5 <test_input_str>
# $6 <eps>
# $7 <dump_mode>
run_single_test $BUILD_OUTPUT_DIR $3 $4 $4 $5 $6 $7 $GEN_ARMV7_FLAG $ONLY_DUMP_ARM "${ARM_EXTRA_DUMP}"

echo "end integration test"
date
