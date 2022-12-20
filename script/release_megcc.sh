#!/bin/bash -e

set -x

if [[ $# -lt 1 ]] ; then
  echo "Usage: $0 <out_dir> "
  exit 1
fi

PROJECT_PATH="$(dirname $(readlink -f $0))/.."
RUNTIME_PATH=${PROJECT_PATH}/runtime
COMPILER_PATH=${PROJECT_PATH}/compiler
OUT_DIR_RP=$1
mkdir -p ${OUT_DIR_RP}/
OUT_DIR=$(readlink -f $OUT_DIR_RP)
rm -fr ${OUT_DIR}/*
mkdir -p ${OUT_DIR}/build_host
mkdir -p ${OUT_DIR}/bin
mkdir -p ${OUT_DIR}/runtime
mkdir -p ${OUT_DIR}/immigration
mkdir -p ${OUT_DIR}/immigration/include
mkdir -p ${OUT_DIR}/example
mkdir -p ${OUT_DIR}/yolox_example 
cp -rf ${PROJECT_PATH}/yolox_example/* ${OUT_DIR}/yolox_example/ 
cp -rf ${PROJECT_PATH}/immigration/include/* ${OUT_DIR}/immigration/include/
mkdir -p ${OUT_DIR}/script
cp -a ${PROJECT_PATH}/script/{ppl_gen.sh,ppl_build.sh,test_model.py} ${OUT_DIR}/script/
cp -r ${PROJECT_PATH}/doc ${OUT_DIR}/doc
cp ${PROJECT_PATH}/README.md ${OUT_DIR}/
${PROJECT_PATH}/third_party/prepare.sh
pushd ${OUT_DIR}/build_host
    cmake ${COMPILER_PATH} -G Ninja
    ninja
    cp tools/mgb-to-tinynn/mgb-to-tinynn ${OUT_DIR}/bin/
    strip mgb-to-tinynn
    cp tools/mgb-runner/mgb-runner ${OUT_DIR}/bin/
    strip mgb-runner
    cp tools/mgb-importer/mgb-importer ${OUT_DIR}/bin/
    strip mgb-importer
    cp tools/kernel_exporter/kernel_exporter ${OUT_DIR}/bin/
    strip kernel_exporter
    cp tools/megcc-opt/megcc-opt ${OUT_DIR}/bin/
    strip megcc-opt
popd
pushd ${PROJECT_PATH}/compiler
    GIT_ID=`git rev-parse --short HEAD`
popd
cp -a $RUNTIME_PATH/{flatcc,include,schema,example,src,CMakeLists.txt,scripts} ${OUT_DIR}/runtime/
strip ${OUT_DIR}/bin/*
rm -fr ${OUT_DIR}/build_host
MEGCC_VER=`${OUT_DIR}/bin/mgb-to-tinynn --version | grep MegCC | awk '{print $3}'`
pushd ${PROJECT_PATH}
   tar -czf megcc_release_${MEGCC_VER}_${GIT_ID}.tar.gz "${OUT_DIR_RP}"
popd

