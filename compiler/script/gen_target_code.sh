#!/bin/bash
set -ex
gen_src_dir=$1
megcc_test_gen_path=$2
rm -fr ${gen_src_dir}/ 
mkdir -p ${gen_src_dir}
LD_LIBRARY_PATH="$3":${LD_LIBRARY_PATH} ${megcc_test_gen_path} ${gen_src_dir} ${extra_gtest_args}
