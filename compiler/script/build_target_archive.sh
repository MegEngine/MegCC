#!/bin/bash
set -ex
script_dir=$(dirname $(readlink -f $0))
gen_dir=$1
extra_cmd=$2
cp ${script_dir}/compile_target_cmake.cmake ${gen_dir}/CMakeLists.txt
mkdir -p ${gen_dir}/build
rm -fr ${gen_dir}/build
mkdir -p ${gen_dir}/build
cd ${gen_dir}/build
cmake .. -G Ninja $extra_cmd
ninja
