#!/bin/bash -e
SRC_DIR=$(dirname $(readlink -f $0))
cd $SRC_DIR
git submodule sync
git submodule update -f --init flatcc
git submodule update -f --init googletest
git submodule update -f --init flatbuffers
if [ "$LLVM_DIR" == "" ];then
    echo "find LLVM_DIR = ${LLVM_DIR}, skip llvm init"
    git submodule update -f --init llvm-project
    pushd llvm-project >/dev/null
    git checkout a2361eb28160dc747b4f5a321faefb9c4cc15ba1
    git am ${SRC_DIR}/0001-feat-add-scale-to-int.patch
    popd >/dev/null
fi
git submodule update -f --init MegEngine

echo "Start downloading MegEngine git submodules"

function mge_git_submodule_update() {
    pushd MegEngine/third_party >/dev/null
    git submodule sync
    git submodule update -f --init midout
    git submodule update -f --init protobuf
    git submodule update -f --init flatbuffers
    git submodule update -f --init Json
    git submodule update -f --init range-v3
    git submodule update -f --init libzmq
    git submodule update -f --init cppzmq
    git submodule update -f --init cpuinfo
    git submodule update -f --init gtest
    git submodule update -f --init gflags
    git submodule update -f --init cpp_redis
    git submodule update -f --init tacopie
    popd >/dev/null
}

mge_git_submodule_update

function build_mge() {
    mkdir -p $2
    cd $2
    cmake -DBUILD_SHARED_LIBS=OFF\
        -DMGE_ARCH=fallback\
        -DMGE_ENABLE_CPUINFO=OFF\
        -DMGE_WITH_CUDA=OFF\
        -DMGE_BUILD_IMPERATIVE_RT=OFF\
        -DMGE_WITH_TEST=ON\
        -DMGE_BUILD_SDK=OFF\
        -DMGE_WITH_LITE=OFF\
        -DMGE_WITH_PYTHON_MODULE=OFF\
        -DMGE_ENABLE_RTTI=ON\
        -DMGE_INFERENCE_ONLY=ON\
        -DMGE_ENABLE_EXCEPTIONS=OFF\
        -DCMAKE_INSTALL_PREFIX=$3 \
        -G Ninja \
        $1

    ninja install
}

echo "begin to build megbrain"
mge_dir=$SRC_DIR/MegEngine
mge_build=$SRC_DIR/MegEngine/build
mge_install=$SRC_DIR/MegEngine/install
build_mge $mge_dir $mge_build $mge_install

function build_flatcc() {
    SOURCE_DIR=$1
    BUILD_DIR=$2

    rm -fr $BUILD_DIR
    mkdir -p $BUILD_DIR
    cd $BUILD_DIR
    cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DFLATCC_CXX_TEST=OFF \
        -DFLATCC_TEST=OFF \
        $SOURCE_DIR

    make -j$(nproc)
}

if [ ! -e $SRC_DIR/flatcc/bin/flatcc ];then
    echo "can not find $SRC_DIR/flatcc/bin/flatcc, build flatcc"
    flatcc_dir=$SRC_DIR/flatcc
    flatcc_build=$SRC_DIR/flatcc/build
    build_flatcc $flatcc_dir $flatcc_build
fi
