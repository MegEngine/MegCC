#!/bin/bash
set -e
export PATH=/root/gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf/bin/:$PATH
export PATH=/root/gcc-arm-none-eabi-10.3-2021.07/bin:$PATH
PROJECT_PATH="$(dirname $(readlink -f $0))/.."
if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <path/to/llvm>"
  exit 1
fi
BUILD_DEMO_SCRIPT=${PROJECT_PATH}/script/build_and_test_not_standard_os.sh
LLVM_DIR=$1 ${BUILD_DEMO_SCRIPT}
