#!/bin/bash

set -bex
git --version
PROJECT_DIR=$(readlink -f "$(dirname "$0")/../")
MEGCC_WORKSPACE_DIR=${PROJECT_DIR}/ci/workdir
megcc_export_copybara_branch='copybara'

readonly MEGCC_URL="git@git-core.megvii-inc.com:brain-sdk/megcc.git"
readonly MEGCC_EXPORT_URL="https://github.com/MegEngine/MegCC.git"
readonly MEGCC_LOCAL_URL="file://${MEGCC_WORKSPACE_DIR}"
readonly COPYBARA_CONFIG="./migrate/copy.bara.sky"
readonly CURRENT_BRANCH="$(git symbolic-ref --short -q HEAD)"
function pull_megcc_export(){
    echo "Starting pull megcc_export"
    if [ -d "${MEGCC_WORKSPACE_DIR}/MegCC" ]; then
        rm -rf ${MEGCC_WORKSPACE_DIR}/MegCC
    fi
    if [ ! -d "${MEGCC_WORKSPACE_DIR}" ]; then
        mkdir -p ${MEGCC_WORKSPACE_DIR}
    fi
    pushd "${MEGCC_WORKSPACE_DIR}" > /dev/null
        git clone -b ${megcc_export_copybara_branch} "${MEGCC_EXPORT_URL}"
        git -C "${MEGCC_WORKSPACE_DIR}/MegCC" config receive.denyCurrentBranch ignore
    popd > /dev/null
    echo "Finished pulling megcc_export"
}
function recover_gitconfig(){
    if [ -e ~/.gitconfig_bak ];then
        mv ~/.gitconfig_bak ~/.gitconfig
    fi
    if [ -e /etc/gitconfig_bak ];then
        mv /etc/gitconfig_bak /etc/gitconfig
    fi
}

function backup_gitconfig(){
    if [ -e ~/.gitconfig ];then
        mv ~/.gitconfig ~/.gitconfig_bak
    fi
    if [ -e /etc/gitconfig ] && [ ${USER} = "root" ];then
        mv /etc/gitconfig /etc/gitconfig_bak
    fi
    git config --global user.email "CI_TEST@example.com"
    git config --global user.name "CI_TEST"
}

function do_copybara(){
    backup_gitconfig
    pushd "${PROJECT_DIR}" > /dev/null
        ./migrate/copybara ${COPYBARA_CONFIG} default \
            --git-destination-url=${MEGCC_LOCAL_URL}/MegCC \
            --git-destination-fetch="${megcc_export_copybara_branch}" \
            --git-destination-push="${megcc_export_copybara_branch}" \
            --init-history --force --verbose

    popd > /dev/null
    git -C "${MEGCC_WORKSPACE_DIR}/MegCC" checkout copybara
    git -C "${MEGCC_WORKSPACE_DIR}/MegCC" reset --hard HEAD 
    recover_gitconfig
}

pull_megcc_export
do_copybara


