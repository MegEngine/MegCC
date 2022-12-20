#!/bin/bash
set -ex
PROJECT_PATH="$(dirname $(readlink -f $0))/.."
if [[ $# -ne 0 ]] ; then
  echo "Usage: $0"
  exit 1
fi
LLVM_DIR="$LLVM_DIR" ${PROJECT_PATH}/third_party/prepare.sh
# build dir gen
COMPILER_BUILD_DIR="${PROJECT_PATH}/build_dir/"
rm -fr ${COMPILER_BUILD_DIR}
mkdir -p ${COMPILER_BUILD_DIR}

MEGCC_SOURCE_DIR=${PROJECT_PATH}/compiler
MEGCC_BUILD_DIR=${COMPILER_BUILD_DIR}/megcc_compiler
mkdir -p ${MEGCC_BUILD_DIR}
if [ "$LLVM_DIR" == "" ];then
    cmake -GNinja \
        "-H${MEGCC_SOURCE_DIR}" \
        "-B${MEGCC_BUILD_DIR}" 
else
    cmake -GNinja \
        "-H${MEGCC_SOURCE_DIR}" \
        "-B${MEGCC_BUILD_DIR}" \
        -DMEGCC_INSTALLED_MLIR_DIR="$LLVM_DIR/lib/cmake"
fi
cmake --build "$MEGCC_BUILD_DIR" -j$(nproc) --target mgb-to-tinynn --target mgb-runner

function check_key_words() {
    #elf self mangle words, we do not care!!
    white_list="@MEGW mgb1 5Mbg6 MGBi O:MgBnWk <mbG =MEG>Yr]< 4emUi0B >HMgE kMEG RmEg MbGV4 MEgIy @MEg mGe#S BMgb MGB( mBg: MBgr8C A&mGB mEg; mGb>/ mEg= .strtab .shstrtab A=MgE= mgb=g MGe= g=MgE <mgE= =Mgb> MGE<"
    elf_file=$1
    if [ ! -f ${elf_file} ];then
        echo "ERR: can not find ${elf_file}"
        exit -1
    fi
    find_words=`strings ${elf_file} |grep -E ".strtab|MGB|MBG|MGE|MEG|huawei|emui|hicloud|hisilicon|hisi|kirin|balong|harmony|senstime|Mge|Mgb"`

    echo “===========================================================”
    echo "find sensitive words: ${find_words}"
    echo “===========================================================”

    for find_word in ${find_words}
    do
        echo "check find word: ${find_word} now"
        IS_IN_WHITE_LIST="FALSE"

        for white_word in ${white_list}
        do
            echo "check find_word vs white_word : (${find_word} : ${white_word})"
            if [ ${find_word} = ${white_word} ]; then
                echo "hit ${find_word} in \"${white_list}\""
                IS_IN_WHITE_LIST="TRUE"
                break
            fi
        done

        if [ ${IS_IN_WHITE_LIST} = "FALSE" ]; then
            echo "ERROR: can not hit ${find_word} in \"${white_list}\", please check your mr"
            exit -1
        fi
    done
}

################################################################################ test bare aarch64 #############################################################
echo "test bare aarch64"
MODEL_PATH=${PROJECT_PATH}/ci/resource/mobilenet/mobilenet.cppmodel
MODEL_INPUT=${PROJECT_PATH}/ci/resource/mobilenet/input.bin
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}
$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,3,224,224)" --arm64  --enable_nchw44 --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
# verify debug lib
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --build_with_dump_tensor --build_with_profile --cross_build_target_arch aarch64 --build_for_debug --cross_build_target_os NOT_STANDARD_OS --remove_old_build
python3 runtime/scripts/check_tinynn_lib.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a

# verify release lib
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --build_with_dump_tensor --build_with_profile --cross_build_target_arch aarch64 --cross_build_target_os NOT_STANDARD_OS --remove_old_build
python3 runtime/scripts/check_tinynn_lib.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a

# run aarch64 qemu
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --build_for_debug --cross_build_target_os NOT_STANDARD_OS
cd ${PROJECT_PATH}/runtime/example/Nonstandard_OS/bare_board
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 test_bare_board_qemu.py --tinynn_lib_install_dir ${KERNEL_DIR}/runtime/install --test_arch aarch64

################################################################################## test bare aarch32 #############################################################
echo "test bare aarch32"
MODEL_PATH=${PROJECT_PATH}/ci/resource/kyy_det/kyy_det_u8.mdl
MODEL_INPUT=${PROJECT_PATH}/ci/resource/kyy_det/input_1_1_384_288.bin
cd ${PROJECT_PATH}
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}
# as qemu aarch32 do not support neon, so we have to test with --baremetal
$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,1,384,288)" --baremetal --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}/input.bin
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
# run aarch32 qemu
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch armv7-a-qemu --build_for_debug --cross_build_target_os NOT_STANDARD_OS
cd ${PROJECT_PATH}/runtime/example/Nonstandard_OS/bare_board
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 test_bare_board_qemu.py --tinynn_lib_install_dir ${KERNEL_DIR}/runtime/install --test_arch aarch32

################################################################################## test freeRTOS #############################################################
echo "test freeRTOS"
cd ${PROJECT_PATH}/ci/resource/minist/
python3 dump_model.py 
MODEL_PATH=${PROJECT_PATH}/ci/resource/minist/minist.mge
MODEL_INPUT=${PROJECT_PATH}/ci/resource/minist/input.bin
cd ${PROJECT_PATH}
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}
# as cortex-m do not support neon, so we have to test with --baremetal
$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,1,32,32)" --baremetal --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}/input.bin
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch cortex-m --cross_build_target_os NOT_STANDARD_OS
# run freeRTOS qemu
cd ${PROJECT_PATH}/runtime/example/Nonstandard_OS/freeRTOS
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 test_freertos.py --tinynn_lib_install_dir ${KERNEL_DIR}/runtime/install --free_rtos_repo_dir /root/FreeRTOS
# verify LINUX local build
cd ${PROJECT_PATH}
./runtime/scripts/runtime_build.py --kernel_dir ${KERNEL_DIR} --remove_old_build --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor --build_for_debug --build_with_callback_register
rm -rf $MODEL_PATH
################################################################################## test tee #############################################################
echo "test tee"
MODEL_PATH=${PROJECT_PATH}/ci/resource/kyy_det/kyy_det_u8.mdl
MODEL_INPUT=${PROJECT_PATH}/ci/resource/kyy_det/input_1_1_384_288.bin
cd ${PROJECT_PATH}
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}
$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,1,384,288)" --arm64  --enable_nchw44 --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}/input.bin
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --build_for_debug --cross_build_target_os NOT_STANDARD_OS
# run freeRTOS qemu
cd ${PROJECT_PATH}/runtime/example/Nonstandard_OS/tee
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 test_optee.py --tinynn_lib_install_dir ${KERNEL_DIR}/runtime/install --optee_repo_dir /root/optee

################################################################################## test riscv #############################################################
echo "test riscv"
cd ${PROJECT_PATH}/ci/resource/minist/
python3 dump_model.py 
MODEL_PATH=${PROJECT_PATH}/ci/resource/minist/minist.mge
MODEL_INPUT=${PROJECT_PATH}/ci/resource/minist/input.bin
cd ${PROJECT_PATH}
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}

$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,1,32,32)" --baremetal --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}/input.bin
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
# verify release lib
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --build_with_dump_tensor --build_with_profile --cross_build_target_arch rv64gcv0p7 --remove_old_build --cross_build_target_os LINUX
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a

#TODO： test RVV at remote after sshhub RVV device ready


rm -rf $MODEL_PATH
#############################  verify more build #####################################################
echo "verify more build"
MODEL_PATH=${PROJECT_PATH}/ci/resource/kyy_det/kyy_det_u8.mdl
MODEL_INPUT=${PROJECT_PATH}/ci/resource/kyy_det/input_1_1_384_288.bin
cd ${PROJECT_PATH}
KERNEL_DIR="${COMPILER_BUILD_DIR}/kernel_dir/"
rm -fr ${KERNEL_DIR}
mkdir -p ${KERNEL_DIR}
$MEGCC_BUILD_DIR/tools/mgb-to-tinynn/mgb-to-tinynn "$MODEL_PATH" "$KERNEL_DIR" --input-shapes="data=(1,1,384,288)" --arm64  --enable_nchw44 --save-model
cp ${MODEL_INPUT} ${KERNEL_DIR}/input.bin
cd ${KERNEL_DIR}
xxd -i input.bin >input.c
cd -

cd ${PROJECT_PATH}
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --cross_build_target_os NOT_STANDARD_OS --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor
python3 runtime/scripts/check_tinynn_lib.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
python3 ${PROJECT_PATH}/runtime/scripts/strip_and_mangling_static_tinynn.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --cross_build_target_os NOT_STANDARD_OS --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor --build_for_debug
python3 runtime/scripts/check_tinynn_lib.py --tinynn_lib ${KERNEL_DIR}/runtime/install/lib/libTinyNN.a
# verify STANDARD_OS(android) build
export NDK_ROOT=/root/android-ndk-r21
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --cross_build_target_os ANDROID --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor --build_for_debug --build_with_callback_register
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --cross_build_target_os ANDROID --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor --build_shared_library
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.so
./runtime/scripts/runtime_build.py --cross_build --kernel_dir ${KERNEL_DIR} --remove_old_build --cross_build_target_arch aarch64 --cross_build_target_os ANDROID --build_with_ninja_verbose --build_tensor_alloc_sanity --build_with_profile --build_with_dump_tensor --build_shared_library --build_with_callback_register
check_key_words ${KERNEL_DIR}/runtime/install/lib/libTinyNN.so
