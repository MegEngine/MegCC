#!/bin/bash
set -eo pipefail

./ci/run_integration_test.sh  ./compiler/local_cmake_build ./regression_test_workdir ./ci/resource/mobilenet/mobilenet.cppmodel "data=(1,3,224,224)" \
"data=ci/resource/mobilenet/input.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build ./shufflenet_test_workdir ./ci/resource/shufflenetv2/shufflenetv2.mge "data=(1,3,224,224)" \
"data=ci/resource/mobilenet/input.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./tof_eyeoc_workdir ./ci/resource/tof_eyeoc/eyeoc.mdl  "data=(1,1,128,128)" \
"data=ci/resource/tof_eyeoc/input.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./tof_det_workdir ./ci/resource/tof_det/det.mdl  "data=(1,1,640,640)" \
"data=ci/resource/tof_det/input.bin" 1e-3 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./tof_pbnobn_workdir ./ci/resource/tof_pb_nobn/pb_nobn.mdl "img=(1,1,128,128)" \
"img=ci/resource/tof_pb_nobn/input.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./tof_atten_workdir  ./ci/resource/tof_atten/tof_attension.mdl "img_raw=(1,1,640,480);lm_raw=(1,162)" \
"img_raw=ci/resource/tof_atten/input_0.bin;lm_raw=ci/resource/tof_atten/input_1.bin"  1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./ir_e151_workdir ./ci/resource/ir_e151/ir_e151.mdl "data=(1,224,224,1);landmark=(1,81,2)" \
"data=ci/resource/ir_e151/input_0.bin;landmark=ci/resource/ir_e151/input_1.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./cpu_int8_workdir ./ci/resource/cpu_int8/cpu_int8.mdl "img0_comp_fullface=(1,3,128,128)" \
"img0_comp_fullface=ci/resource/cpu_int8/input.bin" 2e-2 nchw 0

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./eyelive_int8_workdir ci/resource/eyelive_int8/eyelive_int8.mdl "data=(1,224,224,1);landmark=(1,81,2)" \
"data=ci/resource/eyelive_int8/input_0.bin;landmark=ci/resource/eyelive_int8/input_1.bin"  1e-4  nchw44-dot 0

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./depth_fp32_workdir ./ci/resource/depth_fp32/depth_fp32.mdl "data=(1,256,192,1);landmark=(1,81,2)" \
"data=ci/resource/depth_fp32/depth_data.bin;landmark=ci/resource/depth_fp32/depth_lmk3.bin"  1e-4 nchw44

./ci/run_integration_test.sh ./compiler/local_cmake_build ./kyy_liveness_workdir ./ci/resource/kyy_liveness/kyy_liveness_warp.mdl "data=(1,640,480,3);landmark=(1,81,2)" \
"data=ci/resource/kyy_liveness/fmp_bgr.bin;landmark=ci/resource/kyy_liveness/fmp_lmk.bin"  5e-4 nchw44

./ci/run_integration_test.sh ./compiler/local_cmake_build ./kyy_det_workdir ./ci/resource/kyy_det/kyy_det_u8.mdl \
"data=(1,1,384,288):data=(1,1,288,384)" "data=ci/resource/kyy_det/input_1_1_384_288.bin:data=ci/resource/kyy_det/input_1_1_288_384.bin"  5e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./kyy_det_attr_workdir ./ci/resource/kyy_det/kyy_attr.mdl "img=(1,1,112,112)" \
"img=ci/resource/kyy_det/input_1_1_112_112.bin"  1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./kyy_det_pf_workdir ./ci/resource/kyy_det/kyy_pf_u8.mdl  "data=(1,1,112,112)" \
"data=ci/resource/kyy_det/input_1_1_112_112.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./hako_liveness_workdir ./ci/resource/hako_liveness/models/bf9c6df9.emod  "data=(1,640,480,3);landmark=(1,81,2)" \
"data=ci/resource/kyy_liveness/fmp_bgr.bin;landmark=ci/resource/kyy_liveness/fmp_lmk.bin"  5e-4 nchw44

./ci/run_integration_test.sh ./compiler/local_cmake_build ./hako_rgb_workdir ./ci/resource/hako_rgb/recog.emod "img0_comp_fullface=(1,3,128,128)" \
"img0_comp_fullface=resize_face1.bgr:img0_comp_fullface=resize_face3.bgr" None nchw44-dot 0 1 "--add_nhwc2nchw_to_input -hako 1"

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./csg_det_workdir ./ci/resource/csg_det/csg_det.bin  "data=(1,1,512,512)" \
"data=ci/resource/csg_det/input_1_1_512_512_fp32.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./csg_lmk_workdir ./ci/resource/csg_lmk/csg_lmk.mdl  "data=(1,1,112,112)" \
"data=ci/resource/csg_lmk/input_1_1_112_112_fp32.bin" 1e-4 nchw44
./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./aon_det_workdir ./ci/resource/aon_det/aon_det.mge  "data=(1,1,640,480)" \
"data=ci/resource/aon_det/det_1_1_640_480_input_uint8.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./aon_lmk_workdir ./ci/resource/aon_lmk/aon_lmk.mge  "data=(1,1,128,128)" \
"data=ci/resource/aon_lmk/lmk_1_1_128_128_input_float32.bin" 1e-4 nchw44

./ci/run_integration_test.sh  ./compiler/local_cmake_build  ./finger_liveness_workdir ./ci/resource/finger/liveness/fp_liveness_opt.mdl  "data=(1,2,160,160)" \
"data=ci/resource/finger/liveness/input_1_2_160_160_fp32" 1e-4 nchw44
