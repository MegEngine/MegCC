#!/bin/bash
set -e
SCRIPT_ROOT=$(dirname `readlink -f $0`)
MEGCC_ROOT=`dirname ${SCRIPT_ROOT}`
echo 'remove megbrain cache'
rm -fr ${MEGCC_ROOT}/third_party/MegBrain/build ${MEGCC_ROOT}/third_party/MegBrain/build_dir ${MEGCC_ROOT}/third_party/MegBrain/install 
echo 'if do not have megcc_manylinux2014 img, run "docker build script/docker -t megcc_manlylinux2014"'
docker run -it -v ${MEGCC_ROOT}:/megcc_root megcc_manlylinux2014 /megcc_root/script/release_megcc.sh /megcc_root/release_megcc