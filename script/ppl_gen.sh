#!/bin/bash -e

set -x

if [[ $# -lt 3 ]] ; then
  echo "Usage: $0 <path-to-dump-app> <path-to-json> <out_dir> <extra dump>"
  exit 1
fi

DUMP_APP=$1
JSON_PATH=$2
OUT_DIR=$3
EXTRA_DUMP_CMD=${@:4}
PROJECT_PATH="$(dirname $(readlink -f $0))/.."
RUNTIME_PATH=${PROJECT_PATH}/runtime
mkdir -p ${OUT_DIR}
KERN_DIR="${OUT_DIR}/kern/"
rm -fr ${OUT_DIR}/*
mkdir -p "${OUT_DIR}/runtime"
mkdir -p "${OUT_DIR}/model"
mkdir -p "${OUT_DIR}/model_info"
mkdir -p "${OUT_DIR}/runtime/script"
mkdir -p "${KERN_DIR}"
${DUMP_APP} --json="${JSON_PATH}" "${ARCH_SPECIFIC}" --dump="${KERN_DIR}" ${EXTRA_DUMP_CMD}
cp -r "${RUNTIME_PATH}/flatcc" "${OUT_DIR}/runtime/flatcc"
cp -r "${RUNTIME_PATH}/include" "${OUT_DIR}/runtime/include"
cp -r "${RUNTIME_PATH}/schema" "${OUT_DIR}/runtime/schema"
cp -r "${RUNTIME_PATH}/example" "${OUT_DIR}/runtime/example"
cp -r "${RUNTIME_PATH}/src" "${OUT_DIR}/runtime/src"
cp -r "${RUNTIME_PATH}/toolchains" "${OUT_DIR}/runtime/toolchains"
cp -r "${RUNTIME_PATH}/docker_env" "${OUT_DIR}/runtime/docker_env"
cp -r "${PROJECT_PATH}/immigration" "${OUT_DIR}/immigration"
cp "${RUNTIME_PATH}/CMakeLists.txt" "${OUT_DIR}/runtime/CMakeLists.txt"
cp "${RUNTIME_PATH}/version.ld" "${OUT_DIR}/runtime/version.ld"
MODEL_FILE=`find ${OUT_DIR}/kern/ -name "*.tiny"`
if [ ! -z "${MODEL_FILE}" ];then
  mv ${OUT_DIR}/kern/*.tiny "${OUT_DIR}/model"
  mv ${OUT_DIR}/kern/*.tiny.txt "${OUT_DIR}/model_info"
fi
MODEL_HEAD_FILE=`find ${OUT_DIR}/kern/ -name "*.tiny_head.bin"`
if [ ! -z "${MODEL_HEAD_FILE}" ];then
  mv ${OUT_DIR}/kern/*.tiny_head.bin "${OUT_DIR}/model"
fi
cp -a "${PROJECT_PATH}"/script/{ppl_build.sh,test_model.py} "${OUT_DIR}/"
cp "${RUNTIME_PATH}/scripts/runtime_build.py" "${OUT_DIR}/runtime/script/"
cp "${JSON_PATH}" "${OUT_DIR}/"
tar -czf megcc_ppl_gen.tar.gz "${OUT_DIR}"
