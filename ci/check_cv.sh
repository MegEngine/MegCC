#!/bin/bash -e

set -x
echo "build model from json"
date
if [[ $# -lt 1 ]] ; then
  echo "Usage: $0 <cv_dir> "
  exit 1
fi
CV_DIR=$1
PROJECT_PATH="$(dirname $(readlink -f $0))/.."
cd ${PROJECT_PATH}
## check cv opr exist
$NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android21-clang ci/resource/many_model/test_cv.c ${PROJECT_PATH}/runtime/src/utils.c ${CV_DIR}/tinycv_*.c ${CV_DIR}/*internal_cvremap.c -I${PROJECT_PATH}/runtime/include -I${PROJECT_PATH}/runtime/src -I${PROJECT_PATH}/immigration/include -lm -o ${CV_DIR}/test_cv