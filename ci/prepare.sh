#!/bin/bash -e

set -x
    
if [[ $# -ne 1 ]] ; then
  echo "Usage: $0 <path/to/llvm>"
  exit 1
fi

LLVM_DIR="$1" $(dirname $0)/../third_party/prepare.sh

# we use prebuilt llvm for ci instead of in-tree building
# git submodule update -f --init llvm-project
python3 -m pip install MegEngine --user -i https://pypi.megvii-inc.com/simple
