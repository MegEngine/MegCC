#!/bin/bash
set -e
SCRIPT_ROOT=$(dirname `readlink -f $0`)
MEGCC_ROOT=`dirname ${SCRIPT_ROOT}`
echo 'remove megengine cache'
rm -fr ${MEGCC_ROOT}/third_party/MegEngine/build ${MEGCC_ROOT}/third_party/MegEngine/build_dir ${MEGCC_ROOT}/third_party/MegEngine/install 
echo 'if do not have megcc_manylinux2014 img, run "docker build script/docker -t megcc_manlylinux2014"'
docker_args="-it"
if [ -z "${CI_SERVER_NAME}" ]; then
    CI_SERVER_NAME="null"
fi
if [ ${CI_SERVER_NAME} = "GitLab" ];then
    docker_args="-i"
fi
echo "$CI_SERVER_NAME"
${MEGCC_ROOT}/third_party/prepare.sh no
docker run ${docker_args} -v ${MEGCC_ROOT}:/megcc_root megcc_manlylinux2014 /megcc_root/script/release_megcc.sh /megcc_root/release_megcc