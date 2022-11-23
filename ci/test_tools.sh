#!/bin/bash -e
set -x
function build_dump(){
  PROJECT_PATH="$(readlink -f ${1})"
  COMPILER_PROJECT_PATH="$PROJECT_PATH/compiler"
  COMPILER_BUILD_DIR="$(readlink -f ${2})"
  
  cmake -H$COMPILER_PROJECT_PATH -B$COMPILER_BUILD_DIR -G Ninja
  ninja -C $COMPILER_BUILD_DIR
}

function dump_model_and_gen_kernel(){
  rm -rf "${OUTPUT_DIR}"
  mkdir -p "${OUTPUT_DIR}"
  $COMPILER_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$OUTPUT_DIR" --input-shapes="${INPUT_SHAPE_STR}"
}

function build_runtime(){
  RUNTIME_SRC_DIR="$PROJECT_PATH/runtime"
  rm -rf "${RUNTIME_BUILD_DIR}" 
  mkdir -p "${1}"
  cmake -G Ninja \
    "-H$RUNTIME_SRC_DIR" \
    "-B${1}" \
    "-DRUNTIME_KERNEL_DIR=$OUTPUT_DIR" \
    "-DCMAKE_BUILD_TYPE=Debug"
  cmake --build "$RUNTIME_BUILD_DIR" --target tinynn_test_lite
}

function compare_output_with_mgb(){ 
  INPUT_DATA_SHAPE_STR="${1}"
  INPUT_DATA_STR="${2}"
  EPS="${3}"
  TINYNN_OUTPUT_DIR="$OUTPUT_DIR/tinynn_out/"
  mkdir -p "${TINYNN_OUTPUT_DIR}"
  TINYMODEL_PATH=`find  ${OUTPUT_DIR} -name "*.tiny"`
  TINYNN_SHAPE_STR=`echo $INPUT_DATA_SHAPE_STR | sed 's/[()]//g'`
  $RUNTIME_BUILD_DIR/tinynn_test_lite -m ${TINYMODEL_PATH} -o "$TINYNN_OUTPUT_DIR" -l 0 -d $INPUT_DATA_STR -s ${TINYNN_SHAPE_STR}
  MGB_OUTPUT_DIR="$OUTPUT_DIR/mgb_out/"
  mkdir -p "${MGB_OUTPUT_DIR}"
  if [[ "$MODEL_PATH" == *".emod" ]];then
    HAKO_MODEL_PATH=$MODEL_PATH
    MODEL_PATH="${MODEL_PATH}.mdl"
    $COMPILER_BUILD_DIR/tools/hako-to-mgb/hako-to-mgb $HAKO_MODEL_PATH $MODEL_PATH
  fi
  $COMPILER_BUILD_DIR/tools/mgb-runner/mgb-runner "$MODEL_PATH" "$MGB_OUTPUT_DIR" --input-shapes="${INPUT_DATA_SHAPE_STR}" --input-data="${INPUT_DATA_STR}"
  python3 $PROJECT_PATH/ci/compare_output_bin.py $TINYNN_OUTPUT_DIR $MGB_OUTPUT_DIR --eps="$EPS"
}

# run asan model
function check_mem_leak_with_asan(){
  RUNTIME_BUILD_DIR_ASAN="$RUNTIME_BUILD_DIR/ASAN"
  INPUT_DATA_SHAPE_STR="${1}"
  TINYNN_SHAPE_STR=`echo $INPUT_DATA_SHAPE_STR | sed 's/[()]//g'`
  mkdir -p "$RUNTIME_BUILD_DIR_ASAN"
  cmake -G Ninja \
    "-H$RUNTIME_SRC_DIR" \
    "-B$RUNTIME_BUILD_DIR_ASAN" \
    "-DRUNTIME_KERNEL_DIR=$OUTPUT_DIR" \
    "-DCMAKE_BUILD_TYPE=Debug"  \
    "-DTINYNN_ENABLE_ASAN=ON"   \
    "-DTINYNN_SANITY_ALLOC=ON"  

  cmake --build "$RUNTIME_BUILD_DIR_ASAN" --target tinynn_test_lite
  TINYNN_OUTPUT_ASAN_DIR="$OUTPUT_DIR/tinynn_out_asan"
  mkdir -p ${TINYNN_OUTPUT_ASAN_DIR}  
  $RUNTIME_BUILD_DIR_ASAN/tinynn_test_lite -m ${TINYMODEL_PATH} -o "$TINYNN_OUTPUT_ASAN_DIR" -l 0 -d $INPUT_DATA_STR -s $TINYNN_SHAPE_STR
  python3 $PROJECT_PATH/ci/compare_output_bin.py $TINYNN_OUTPUT_ASAN_DIR $MGB_OUTPUT_DIR --eps="$EPS"
}

# compile arm model
function dump_and_build_arm_sdk(){
  DUMP_MODE="${1}"
  GEN_ARMV7_FLAG="${2}"
  ARM_EXTRA_DUMP="${3}"
  DUMP_OPT=""
  if [ "${DUMP_MODE}" == "nchw44" ];then
    DUMP_OPT="--enable_nchw44"
  elif [ "${DUMP_MODE}" == "nchw44-dot" ];then
    DUMP_OPT="--enable_nchw44_dot"
    GEN_ARMV7_FLAG=0
  fi
  ARM_OUTPUT_DIR="${OUTPUT_DIR}/arm/"
  mkdir -p $ARM_OUTPUT_DIR
  $COMPILER_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$ARM_OUTPUT_DIR" --input-shapes="${INPUT_SHAPE_STR}" ${DUMP_OPT} --arm64 ${ARM_EXTRA_DUMP}
  $PROJECT_PATH/runtime/scripts/runtime_build.py --cross_build --kernel_dir ${ARM_OUTPUT_DIR} --remove_old_build
  if [ "${GEN_ARMV7_FLAG}" == 1 ];then
    ARMV7_OUTPUT_DIR="${OUTPUT_DIR}/armv7/"
    mkdir -p $ARMV7_OUTPUT_DIR
    $COMPILER_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$ARMV7_OUTPUT_DIR" --input-shapes="${INPUT_SHAPE_STR}" ${DUMP_OPT} --armv7 ${ARM_EXTRA_DUMP}
    $PROJECT_PATH/runtime/scripts/runtime_build.py --cross_build --kernel_dir ${ARMV7_OUTPUT_DIR} --cross_build_target_arch armv7-a --remove_old_build
  fi
}

function run_single_test(){
  ONLY_DUMP_ARM=$9
  OUTPUT_DIR="$(readlink -f ${1})"
  MODEL_PATH="$(readlink -f ${2})"
  INPUT_SHAPE_STR="${3}"
  if [[ ${ONLY_DUMP_ARM} == 0 ]];then
    dump_model_and_gen_kernel
    RUNTIME_BUILD_DIR="$OUTPUT_DIR/runtime"
    build_runtime $RUNTIME_BUILD_DIR
    compare_output_with_mgb "${4}" "${5}" "${6}"
    check_mem_leak_with_asan "${4}"
  fi
  dump_and_build_arm_sdk "${7}" "${8}" "${10}"
}
