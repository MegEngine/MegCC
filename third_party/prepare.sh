#!/bin/bash -e
READLINK=readlink
OS=$(uname -s)
if [ $OS = "Darwin" ];then
    READLINK=greadlink
fi
SRC_DIR=$(dirname $(${READLINK} -f $0))
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
    git am ${SRC_DIR}/0001-fix-mlir-buildintype-fix-float16-invalid-for-arrayre.patch
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
    rm -rf $2
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

    make -j`expr $(nproc) - 2`
}

if [ ! -e $SRC_DIR/flatcc/bin/flatcc ];then
    echo "can not find $SRC_DIR/flatcc/bin/flatcc, build flatcc"
    flatcc_dir=$SRC_DIR/flatcc
    flatcc_build=$SRC_DIR/flatcc/build
    build_flatcc $flatcc_dir $flatcc_build
fi

function build_protobuf() {
    cd $1
    git checkout v3.20.2
    rm -rf $2
    mkdir -p $2
    cd $2
    cmake $1/cmake \
    -Dprotobuf_BUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$3 \
    -DCMAKE_INSTALL_SYSCONFDIR=/etc \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -Dprotobuf_BUILD_TESTS=OFF \
    -DCMAKE_BUILD_TYPE=Release
    make -j32
    make install
}

echo "begin to build protobuf"
cd $SRC_DIR
git submodule update -f --init protobuf
protobuf_dir=$SRC_DIR/protobuf
protobuf_build=$SRC_DIR/protobuf/build
protobuf_install=$SRC_DIR/protobuf/install
build_protobuf $protobuf_dir $protobuf_build $protobuf_install

export PATH=$SRC_DIR/protobuf/install/bin:$PATH
export CMAKE_PREFIX_PATH=$SRC_DIR/protobuf/install/lib:$CMAKE_PREFIX_PATH
python3_path=$(which "python3")
function build_onnx() {
    cd $1 
    git checkout 7f0a6331
    rm -rf $2
    mkdir -p $2
    cd $2
    cmake -DCMAKE_INSTALL_PREFIX=$3\
          -DONNX_USE_LITE_PROTO=ON\
          -DPYTHON_EXECUTABLE=$python3_path\
          -G Ninja \
        $1

    ninja install
}

echo "begin to build onnx"
cd $SRC_DIR
git submodule update -f --init onnx
onnx_dir=$SRC_DIR/onnx
onnx_build=$SRC_DIR/onnx/build
onnx_install=$SRC_DIR/onnx/install
build_onnx $onnx_dir $onnx_build $onnx_install
