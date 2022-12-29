#!/bin/bash
set -ex
PROJECT_PATH="$(dirname $(readlink -f $0))/"
KERNEL_DIR="${PROJECT_PATH}/kern/"
${PROJECT_PATH}/runtime/script/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --specify_build_dir ${PROJECT_PATH}/build $@
