#!/bin/bash
set -x 
# clean the tmp file folder
rm -fr ./xxx_nchw44/*
rm -fr ./tiny_out/*
rm -fr ./mgb_out/*
rm -fr ./dump/*

# get the settings
filepath=$1
input_name=$2
input_shape=$3
input_data=$4
other_options=$5
input_shape_string="$2=$3"
input_data_string="$2=$4"

# transform mgb model to tiny model
./tools/mgb-to-tinynn/mgb-to-tinynn $filepath --input-shapes $input_shape_string xxx_nchw44 --arm64 $other_options
# build the tinynn test program and other tools # --help for more runtime_build.py args
../../runtime/scripts/runtime_build.py --cross_build --kernel_dir ./xxx_nchw44 --build_with_profile --remove_old_build

#run tinynn test and get log or other outputs
filename=${filepath##*/}
tinyname="${filename%.*}.tiny"
run2mi9 ./xxx_nchw44/$tinyname
run2mi9 $input_data
run2mi9 ./xxx_nchw44/runtime/tinynn_test_lite $tinyname tiny_out 0 $input_data #>& get log  megcc_log
cpmi9 tiny_out/
cpmi9 ./dump/

#run mgb test and get log
./tools/mgb-runner/mgb-runner $filepath  mgb_out --input-shapes $input_shape_string --input-data $input_data_string   $other_options --bin_dump #--verbose >& mgb_log #

#compare the outputs difference between tinynn and mgb
python3 ../../ci/compare_output_bin.py ./tiny_out ./mgb_out
