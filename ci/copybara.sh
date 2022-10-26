#!/bin/bash

set -bex

PROJECT_DIR=$(readlink -f "$(dirname "$0")/../")
MEGCC_WORKSPACE_DIR=${PROJECT_DIR}/ci/workdir
megcc_export_copybara_branch='copybara'

readonly MEGCC_URL="git@git-core.megvii-inc.com:brain-sdk/megcc.git"
readonly MEGCC_EXPORT_URL="https://github.com/MegEngine/MegCC.git"
readonly MEGCC_LOCAL_URL="file://${MEGCC_WORKSPACE_DIR}"
readonly COPYBARA_CONFIG="./migrate/copy.bara.sky"

function pull_megcc_export(){
    echo "Starting pull megcc_export"
    if [ -d "${MEGCC_WORKSPACE_DIR}/MegCC" ]; then
        rm -rf ${MEGCC_WORKSPACE_DIR}/MegCC
    fi
    if [ ! -d "${MEGCC_WORKSPACE_DIR}" ]; then
        mkdir -p ${MEGCC_WORKSPACE_DIR}
    fi
    pushd "${MEGCC_WORKSPACE_DIR}" > /dev/null
        git clone "${MEGCC_EXPORT_URL}"
    popd > /dev/null
    echo "Finished pulling megcc_export"
}

function do_copybara(){
    pushd "${PROJECT_DIR}" > /dev/null
        ./migrate/copybara ${COPYBARA_CONFIG} default \
            --git-destination-url=${MEGCC_LOCAL_URL}/MegCC \
            --git-destination-fetch="${megcc_export_copybara_branch}" \
            --git-destination-push="${megcc_export_copybara_branch}" \
            --init-history --force

    popd > /dev/null
}

eval $(curl -s http://deploy.i.brainpp.cn/httpproxy)
pull_megcc_export
do_copybara
