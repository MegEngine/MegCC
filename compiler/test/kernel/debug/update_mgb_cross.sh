#!/bin/bash
set -ex
SRC_DIR=$(readlink -f "`dirname $0`/../../../../")
MGB_DIR=$SRC_DIR/third_party/MegEngine/build_dir/android/arm64-v8a/Release/build
pwd=`pwd`
cd $MGB_DIR
ninja liblite_static_all_in_one.a
cp lite/liblite_static_all_in_one.a $pwd/MGB_CROSS/
echo "done"
