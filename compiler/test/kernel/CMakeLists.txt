cmake_minimum_required(VERSION 3.15.2)
# target megcc_test_gen will gen c code to the dir file libcompile_target.a will archive
# c code with script build_target_archive.sh and device cmake file target megcc_test_run
# will link libcompile_target.a gtest
project(MegCCKernelTest)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
option(MEGCC_COMPILER_TARGET_TEST "Enable megcc target unit test run" ON)
option(MEGCC_COMPILER_KERNEL_WITH_ASAN "Enalbe asan check" OFF)
option(MEGCC_COMPILER_KERNEL_BENCHMARK "Enalbe kernel benchmark" OFF)
option(MEGCC_COMPILER_KERNEL_DISABLE_DOT "Disable ARM Dot kernel" OFF)
option(MEGCC_COMPILER_KERNEL_MLIR_AUTO "Enable MLIR gen auto kernel" OFF)

if(MEGCC_COMPILER_KERNEL_MLIR_AUTO)
  set(MEGCC_ENABLE_MLIR_KERN_GEN 1)
else()
  set(MEGCC_ENABLE_MLIR_KERN_GEN 0)
endif()

set(MEGCC_COMPILER_DIR ${PROJECT_SOURCE_DIR}/../../)
set(MEGCC_TARGET_ARCHIVE_DEP ${MEGCC_COMPILER_DIR}/script/compile_target_cmake.cmake
                             ${MEGCC_COMPILER_DIR}/script/build_target_archive.sh)
set(MEGCC_CONST_ARMV8_ABI "arm64-v8a")
set(MEGCC_CONST_ARMV7_ABI "armeabi-v7a")
set(MEGCC_CONST_RISCV_RVV "rv64gcv0p7")

if(CMAKE_TOOLCHAIN_FILE)
  set(MEGCC_CROSS_COMPILE ON)
  if("${ANDROID_ARCH_NAME}" STREQUAL "arm64")
    set(MEGCC_COMPILER_KERNEL_TEST_ARCH ${MEGCC_CONST_ARMV8_ABI})
  elseif("${ANDROID_ARCH_NAME}" STREQUAL "arm")
    set(MEGCC_COMPILER_KERNEL_TEST_ARCH ${MEGCC_CONST_ARMV7_ABI})
  elseif("${RISCV_CROSS_BUILD_ARCH}" STREQUAL "riscv64")
    set(MEGCC_COMPILER_KERNEL_TEST_ARCH ${MEGCC_CONST_RISCV_RVV})
  else()
    message(
      FATAL_ERROR
        "unsupport arch to cross compile for megcc (only arm64-v8a, armeabi-v7a, rv64gcv0p7 is supported)"
    )
  endif()
  if(NOT MGB_HOST_INSTALL_DIR)
    if("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV8_ABI}
       OR "${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV7_ABI})
      if("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV8_ABI})
        set(MGB_BUILD_LIBS
            ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/build_dir/android/${ANDROID_ABI}/Release/install/lite/lib/aarch64/liblite_static_all_in_one.a
        )
      elseif("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV7_ABI})
        set(MGB_BUILD_LIBS
            ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/build_dir/android/${ANDROID_ABI}/Release/install/lite/lib/armv7/liblite_static_all_in_one.a
        )
      endif()
      set(MGB_INCLUDE_DIR
          ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/build_dir/android/${ANDROID_ABI}/Release/install/include
      )
      set(MGB_BUILD_SCRIPT
          ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/scripts/cmake-build/cross_build_android_arm_inference.sh
      )
    elseif("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_RISCV_RVV})
      set(MGB_BUILD_LIBS
          ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/build_dir/riscv-linux/${MEGCC_CONST_RISCV_RVV}/Release/install/lite/lib/riscv64/liblite_static_all_in_one.a
      )
      set(MGB_INCLUDE_DIR
          ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/build_dir/riscv-linux/${MEGCC_CONST_RISCV_RVV}/Release/install/include
      )
      set(MGB_BUILD_SCRIPT
          ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/scripts/cmake-build/cross_build_linux_riscv_inference.sh
      )
    endif()
    set(MGB_INSTALL_LIBS
        ${CMAKE_CURRENT_BINARY_DIR}/MGB_CROSS/liblite_static_all_in_one.a)
    set(MGB_INCLUDE_COPY_DIR ${CMAKE_CURRENT_BINARY_DIR}/MGB_CROSS/)
    add_custom_command(
      OUTPUT ${MGB_INSTALL_LIBS}
      COMMAND unset EXTRA_CMAKE_ARGS
      COMMAND
        export
        EXTRA_CMAKE_ARGS=\""-DMGE_BUILD_IMPERATIVE_RT=OFF -DMGE_WITH_TEST=OFF -DMGE_BUILD_SDK=OFF -DMGE_WITH_LITE=ON -DMGE_WITH_PYTHON_MODULE=OFF -DMGE_ENABLE_RTTI=ON -DBUILD_SHARED_LIBS=OFF"\"
      COMMAND ${MGB_BUILD_SCRIPT} -a ${MEGCC_COMPILER_KERNEL_TEST_ARCH}
      COMMAND cp ${MGB_BUILD_LIBS} ${CMAKE_CURRENT_BINARY_DIR}/MGB_CROSS/
      COMMAND cp -r ${MGB_INCLUDE_DIR} ${MGB_INCLUDE_COPY_DIR}/
      COMMENT "Building megengine to static library."
      DEPENDS)
  endif()
else()
  set(MEGCC_CROSS_COMPILE OFF)
  set(MGB_INSTALL_LIBS
      ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/install/lib/libmegengine.a
      ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/install/lib/libflatbuffers.a)
  set(MGB_INCLUDE_COPY_DIR ${MEGCC_COMPILER_DIR}/../third_party/MegEngine/install)
endif()

message(STATUS "Kernel Test BUILD INFO:")
message(STATUS "\tSystem:                                 ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor:                              ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tBuild type:                             ${CMAKE_BUILD_TYPE}")
message(
  STATUS "\tBuild for arch:                         ${MEGCC_COMPILER_KERNEL_TEST_ARCH}")
message(
  STATUS "\tBuild with Benchmark:                   ${MEGCC_COMPILER_KERNEL_BENCHMARK}")
message(
  STATUS "\tBuild with Asan:                        ${MEGCC_COMPILER_KERNEL_WITH_ASAN}")
message(
  STATUS "\tBuild with Auto:                        ${MEGCC_COMPILER_KERNEL_MLIR_AUTO}")

add_library(mgb_imported STATIC ${MGB_INSTALL_LIBS})
target_link_libraries(mgb_imported PUBLIC ${MGB_INSTALL_LIBS})
target_include_directories(mgb_imported PUBLIC ${MGB_INCLUDE_COPY_DIR}/include)
set_target_properties(mgb_imported PROPERTIES LINKER_LANGUAGE CXX)

set(MEGCC_COMPILE_FLAG -DENABLE_ASSERT=1 -DENABLE_LOG=1 -Wall -Wno-class-memaccess)
if(MEGCC_COMPILER_KERNEL_BENCHMARK)
  set(MEGCC_COMPILE_FLAG ${MEGCC_COMPILE_FLAG} -DENABLE_KERNEL_BENCHMARK=1)
endif()
set(MEGCC_THIRD_PARTY_DIR ${MEGCC_COMPILER_DIR}/../third_party/)

set(MEGCC_LINK_FLAG "")
if(MEGCC_COMPILER_KERNEL_WITH_ASAN)
  if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS " -fsanitize=address -g -O0 -static-libsan ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS " -fsanitize=address -g -O0 -static-libsan ${CMAKE_C_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libsan ${CMAKE_LINK_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS " -fsanitize=address -g -O0 -static-libasan ${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS " -fsanitize=address -g -O0 -static-libasan ${CMAKE_C_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libasan ${CMAKE_LINK_FLAGS}")
  endif()
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    set(MEGCC_COMPILE_FLAG ${MEGCC_COMPILE_FLAG} -g -O0)
  else()
    set(MEGCC_COMPILE_FLAG ${MEGCC_COMPILE_FLAG} -O3)
  endif()
endif()

if(MEGCC_COMPILER_KERNEL_DISABLE_DOT)
  set(MEGCC_COMPILE_FLAG ${MEGCC_COMPILE_FLAG} -DMEGCC_WITHOUT_DOT=1)
endif()
set(TCC_INCLUDE_DIR ${MEGCC_COMPILER_DIR}/../third_party/tcc/include)
set(RUNTIME_SRC_DIR ${MEGCC_COMPILER_DIR}/../runtime/src)
set(RUNTIME_INCLUDE_DIR ${MEGCC_COMPILER_DIR}/../runtime/include)

set(IMMIGRATION_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../immigration/include)

add_subdirectory(../../lib/KernelGen ${PROJECT_BINARY_DIR}/kernelGen/)
add_subdirectory(../../lib/Common ${PROJECT_BINARY_DIR}/Common/)

add_subdirectory(${MEGCC_COMPILER_DIR}/../third_party/googletest
                 ${CMAKE_CURRENT_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
set(MEGCC_TESTGEN_FLAG_DIR ${CMAKE_CURRENT_BINARY_DIR}/genflag/megcc_test_gen/)
set(MEGCC_TESTRUN_FLAG_DIR ${CMAKE_CURRENT_BINARY_DIR}/genflag/megcc_test_run/)
file(GLOB_RECURSE TEST_COMMON_SRC ./common/*.cpp ./test_main.cpp ./tools/*.cpp)
file(GLOB_RECURSE TEST_NAIVE_SRC ./opr/naive/*.cpp)
file(GLOB_RECURSE TEST_AUTO_NAIVE_SRC ./opr/auto_naive/*.cpp)
file(GLOB_RECURSE TEST_ARM64_SRC ./opr/arm/*.cpp)
file(GLOB_RECURSE TEST_ARMV7_SRC ./opr/armv7/*.cpp)
file(GLOB_RECURSE TEST_GI_SRC ./opr/generalIntrinsic/*.cpp)
set(RUNTIME_UTILS_SRC ${MEGCC_COMPILER_DIR}/../runtime/src/utils.c)
list(APPEND TEST_GEN_SRC ${TEST_COMMON_SRC} ${TEST_NAIVE_SRC} ${TEST_GI_SRC}
     ${RUNTIME_UTILS_SRC})
list(APPEND TEST_RUN_SRC ${TEST_COMMON_SRC} ${TEST_NAIVE_SRC} ${TEST_GI_SRC}
     ${RUNTIME_UTILS_SRC})

if(MEGCC_COMPILER_KERNEL_MLIR_AUTO)
  list(APPEND TEST_GEN_SRC ${TEST_AUTO_NAIVE_SRC})
  list(APPEND TEST_RUN_SRC ${TEST_AUTO_NAIVE_SRC})
endif()
set(MEGCC_TEST_COMMON_INCLUDE
    ${MEGCC_COMPILER_DIR}/
    ${MEGCC_COMPILER_DIR}/../runtime/src/
    ${MEGCC_COMPILER_DIR}/../runtime/include
    ${MEGCC_COMPILER_DIR}/include
    ${MEGCC_COMPILER_DIR}/../immigration/include
    ${PROJECT_SOURCE_DIR}/prebuild_include
    ${MEGCC_COMPILER_DIR}/lib/KernelGen)
set(MEGCC_TEST_COMMON_LINKED gtest KernelGen mgb_imported Common)

set(MEGCC_TEST_GEN_SRC_DIR ${CMAKE_CURRENT_BINARY_DIR}/gentest_src/)
set(MEGCC_TEST_GEN_SRC_DONE ${MEGCC_TEST_GEN_SRC_DIR}/done)
set(MEGCC_TEST_GEN_SRC_LIB ${MEGCC_TEST_GEN_SRC_DIR}/build/libcompile_target.a)

if(MEGCC_COMPILER_KERNEL_TEST_ARCH STREQUAL ${MEGCC_CONST_ARMV8_ABI})
  list(APPEND TEST_GEN_SRC ${TEST_ARM64_SRC})
  list(APPEND TEST_RUN_SRC ${TEST_ARM64_SRC})
elseif(MEGCC_COMPILER_KERNEL_TEST_ARCH STREQUAL ${MEGCC_CONST_ARMV7_ABI})
  list(APPEND TEST_GEN_SRC ${TEST_ARMV7_SRC})
  list(APPEND TEST_RUN_SRC ${TEST_ARMV7_SRC})
endif()

if(NOT MEGCC_CROSS_COMPILE)
  # host mode compile test_gen
  add_custom_command(
    OUTPUT ${MEGCC_TESTGEN_FLAG_DIR}/megcc_test_config.h
    COMMAND mkdir -p ${MEGCC_TESTGEN_FLAG_DIR} && echo "#define MEGCC_TEST_GEN 1" >
            ${MEGCC_TESTGEN_FLAG_DIR}/megcc_test_config.h
    DEPENDS
    VERBATIM)
  get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
  add_executable(megcc_test_gen ${TEST_GEN_SRC}
                                ${MEGCC_TESTGEN_FLAG_DIR}/megcc_test_config.h)
  target_link_libraries(megcc_test_gen PUBLIC ${MEGCC_TEST_COMMON_LINKED}
                                              ${MEGCC_LINK_FLAG})
  target_link_libraries(megcc_test_gen
                        PUBLIC ${MEGCC_COMPILER_DIR}/../third_party/tcc/lib/libtcc.a dl)
  target_include_directories(megcc_test_gen PRIVATE ${MEGCC_TEST_COMMON_INCLUDE}
                                                    ${MEGCC_TESTGEN_FLAG_DIR})
  target_compile_options(megcc_test_gen PRIVATE ${MEGCC_COMPILE_FLAG})
  target_link_options(megcc_test_gen PRIVATE "-fuse-ld=gold")
  install(
    TARGETS megcc_test_gen
    EXPORT megcc_test_gen
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT megcc_test_gen)
  set(GEN_SRC_EXTRA_FLAG
      "-DMEGCC_COMPILER_DIR=${MEGCC_COMPILER_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMEGCC_COMPILER_KERNEL_WITH_ASAN=${MEGCC_COMPILER_KERNEL_WITH_ASAN}"
  )
  add_custom_command(
    OUTPUT ${MEGCC_TEST_GEN_SRC_LIB}
    COMMAND
      ${MEGCC_COMPILER_DIR}/script/gen_target_code.sh ${MEGCC_TEST_GEN_SRC_DIR}
      $<TARGET_FILE:megcc_test_gen> "" &&
      ${MEGCC_COMPILER_DIR}/script/build_target_archive.sh ${MEGCC_TEST_GEN_SRC_DIR}
      "${GEN_SRC_EXTRA_FLAG}"
    DEPENDS megcc_test_gen ${MEGCC_TARGET_ARCHIVE_DEP}
    VERBATIM)
else()
  # target mode depend test_gen compiled by host mode
  include(ExternalProject)
  set(MEGCC_HOST_BUILD ${PROJECT_BINARY_DIR}/host_megcc_test)
  set(MEGCC_HOST_BUILD_GEN ${MEGCC_HOST_BUILD}/src/host_megcc_test-build/megcc_test_gen)
  set(MEGCC_TEST_CORE_SRC ${CMAKE_CURRENT_BINARY_DIR}/gentest_src/target.c)
  set(MEGCC_TEST_HELPER_SRC ${CMAKE_CURRENT_BINARY_DIR}/gentest_src/target_helper.cpp)

  ExternalProject_Add(
    host_megcc_test
    SOURCE_DIR ${PROJECT_SOURCE_DIR}
    PREFIX ${MEGCC_HOST_BUILD}
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=${MEGCC_HOST_BUILD}
               -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
               -DMEGCC_COMPILER_TARGET_TEST=OFF
               -DMEGCC_COMPILER_KERNEL_TEST_ARCH=${MEGCC_COMPILER_KERNEL_TEST_ARCH}
               -DMEGCC_COMPILER_KERNEL_WITH_ASAN=${MEGCC_COMPILER_KERNEL_WITH_ASAN}
               -DMEGCC_COMPILER_KERNEL_BENCHMARK=${MEGCC_COMPILER_KERNEL_BENCHMARK}
    BUILD_BYPRODUCTS ${MEGCC_HOST_BUILD_GEN}
    BUILD_ALWAYS TRUE)
  add_executable(codegen IMPORTED)
  add_dependencies(codegen host_megcc_test)
  set_property(TARGET codegen PROPERTY IMPORTED_LOCATION ${MEGCC_HOST_BUILD_GEN})
  if("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV8_ABI}
     OR "${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV7_ABI})
    set(GEN_CROSS_SRC_EXTRA_FLAG
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DANDROID_NDK=${ANDROID_NDK} -DANDROID_ABI=${ANDROID_ABI}  -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL} -DMEGCC_COMPILER_DIR=${MEGCC_COMPILER_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMEGCC_COMPILER_KERNEL_WITH_ASAN=${MEGCC_COMPILER_KERNEL_WITH_ASAN}"
    )
  elseif("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_RISCV_RVV})
    set(GEN_CROSS_SRC_EXTRA_FLAG
        "-DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE} -DMEGCC_COMPILER_DIR=${MEGCC_COMPILER_DIR} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMEGCC_COMPILER_KERNEL_WITH_ASAN=${MEGCC_COMPILER_KERNEL_WITH_ASAN}"
    )
  endif()

  add_custom_command(
    OUTPUT ${MEGCC_TEST_GEN_SRC_LIB}
    COMMAND
      ${MEGCC_COMPILER_DIR}/script/gen_target_code.sh ${MEGCC_TEST_GEN_SRC_DIR}
      $<TARGET_FILE:codegen> && ${MEGCC_COMPILER_DIR}/script/build_target_archive.sh
      ${MEGCC_TEST_GEN_SRC_DIR} "${GEN_CROSS_SRC_EXTRA_FLAG}"
    DEPENDS codegen ${MEGCC_TARGET_ARCHIVE_DEP}
    VERBATIM)
endif()

if(MEGCC_COMPILER_TARGET_TEST)
  add_library(compile_target_lib STATIC IMPORTED)
  add_dependencies(compile_target_lib compile_target)
  set_property(TARGET compile_target_lib PROPERTY IMPORTED_LOCATION
                                                  ${MEGCC_TEST_GEN_SRC_LIB})
  add_custom_command(
    OUTPUT ${MEGCC_TESTRUN_FLAG_DIR}/megcc_test_config.h
    COMMAND mkdir -p ${MEGCC_TESTRUN_FLAG_DIR} && echo "#define MEGCC_TEST_GEN 0" >
            ${MEGCC_TESTRUN_FLAG_DIR}/megcc_test_config.h
    DEPENDS
    VERBATIM)
  add_executable(
    megcc_test_run ${TEST_RUN_SRC} ${MEGCC_TESTRUN_FLAG_DIR}/megcc_test_config.h
                   ${MEGCC_TEST_GEN_SRC_LIB})
  target_link_libraries(
    megcc_test_run PUBLIC ${MEGCC_TEST_COMMON_LINKED} ${MEGCC_TEST_GENLIB}
                          ${MEGCC_TEST_GEN_SRC_LIB} ${MEGCC_LINK_FLAG})
  target_include_directories(
    megcc_test_run PRIVATE ${MEGCC_TEST_COMMON_INCLUDE} ${MEGCC_TESTRUN_FLAG_DIR}
                           ${PROJECT_SOURCE_DIR}/include)
  target_compile_options(megcc_test_run PRIVATE ${MEGCC_COMPILE_FLAG})
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    # do not stripe the symbol when build for debug
  else()
    set_target_properties(megcc_test_run PROPERTIES LINK_FLAGS -s)
  endif()
  if(MEGCC_CROSS_COMPILE)
    if("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV8_ABI}
       OR "${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_ARMV7_ABI})
      target_link_libraries(megcc_test_run PRIVATE -llog)
    elseif("${MEGCC_COMPILER_KERNEL_TEST_ARCH}" STREQUAL ${MEGCC_CONST_RISCV_RVV})
      message(STATUS "link for rvv")
      target_link_libraries(megcc_test_run PUBLIC dl atomic)
    endif()
  endif()
endif()
