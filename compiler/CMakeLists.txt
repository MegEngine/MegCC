cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

project(MegCC)

# set llvm build options
set(LLVM_INSTALL_UTILS
    ON
    CACHE BOOL "" FORCE)
set(LLVM_ENABLE_LLD
    OFF
    CACHE BOOL "" FORCE)
set(LLVM_ENABLE_PROJECTS
    "mlir"
    CACHE STRING "" FORCE)
set(LLVM_TARGETS_TO_BUILD
    "X86"
    CACHE STRING "" FORCE)
set(LLVM_INCLUDE_TOOLS
    ON
    CACHE BOOL "" FORCE)
set(LLVM_BUILD_TOOLS
    OFF
    CACHE BOOL "" FORCE)
set(LLVM_INCLUDE_TESTS
    OFF
    CACHE BOOL "" FORCE)
set(LLVM_ENABLE_ASSERTION
    ON
    CACHE BOOL "" FORCE)
set(LLVM_STATIC_LINK_CXX_STDLIB
    ON
    CACHE BOOL "" FORCE)
set(LLVM_ENABLE_ZLIB
    OFF
    CACHE BOOL "" FORCE)

set(MEGCC_INSTALLED_MLIR_DIR
    ""
    CACHE STRING "path of installed mlir")

set(CMAKE_FIND_LIBRARY_SUFFIXES .a ${CMAKE_FIND_LIBRARY_SUFFIXES})

if(MEGCC_INSTALLED_MLIR_DIR)
  set(MLIR_DIR ${MEGCC_INSTALLED_MLIR_DIR}/mlir)
  set(LLVM_DIR ${MEGCC_INSTALLED_MLIR_DIR}/llvm)
  find_package(MLIR REQUIRED CONFIG)
  message(STATUS "Using MLIRConfig.cmake in : ${MLIR_DIR}")
  message(STATUS "Using LLVMConfig.cmake in : ${LLVM_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
else()
  set(_CMAKE_BUILD_TYPE "${CMAKE_BUILD_TYPE}")
  add_subdirectory(${PROJECT_SOURCE_DIR}/../third_party/llvm-project/llvm
                   ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm EXCLUDE_FROM_ALL)
  set(CMAKE_BUILD_TYPE
      "${_CMAKE_BUILD_TYPE}"
      CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")
  list(APPEND CMAKE_MODULE_PATH
       "${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm/")

  list(APPEND LLVM_INCLUDE_DIRS
       ${PROJECT_SOURCE_DIR}/../third_party/llvm-project/llvm/include
       ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include)
  list(APPEND MLIR_INCLUDE_DIRS
       ${PROJECT_SOURCE_DIR}/../third_party/llvm-project/mlir/include
       ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/tools/mlir/include)

  set(MLIR_TABLEGEN_EXE mlir-tblgen)
endif()

message(
  STATUS
    "llvm include dir: ${LLVM_INCLUDE_DIRS}, mlir include dir: ${MLIR_INCLUDE_DIRS}")

include(LLVMConfig)
include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

find_library(
  MGB_INSTALL_FLAT_LIB
  NAMES libflatbuffers.a
  PATHS ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/install/lib/
        ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/install/lib64/ REQUIRED)
find_library(
  MGB_INSTALL_MGE_LIB
  NAMES libmegengine.a
  PATHS ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/install/lib/
        ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/install/lib64/ REQUIRED)
set(MGB_INSTALL_LIB ${MGB_INSTALL_FLAT_LIB} ${MGB_INSTALL_MGE_LIB})

set(MGB_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/install/include)

add_library(mgb_imported STATIC ${MGB_INSTALL_LIB})
target_link_libraries(mgb_imported PUBLIC ${MGB_INSTALL_LIB})
target_include_directories(mgb_imported PUBLIC ${MGB_INCLUDE_DIR})
set_target_properties(mgb_imported PROPERTIES LINKER_LANGUAGE CXX)

set(TCC_INSTALL_LIB ${PROJECT_SOURCE_DIR}/../third_party/tcc/lib/libtcc.a)
set(TCC_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../third_party/tcc/include)

set(RUNTIME_SRC_DIR ${PROJECT_SOURCE_DIR}/../runtime/src)
set(RUNTIME_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/../runtime/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -g")

include(cmake/GenLiteSchema.cmake)
gen_lite_schema()

include(cmake/GitUtils.cmake)
# 获取当前的GIT_HASH
set(GIT_HASH "unknown")
get_git_hash(GIT_HASH)
message(STATUS "Git hash is ${GIT_HASH}")

# 获取当前的分支
set(GIT_BRANCH "unknown")
get_git_branch(GIT_BRANCH)
message(STATUS "Git branch is ${GIT_BRANCH}")

check_cxx_compiler_flag(-fuse-ld=gold CXX_SUPPORT_GOLD)
if(CXX_SUPPORT_GOLD
   AND NOT ANDROID
   AND NOT APPLE
   AND NOT MSVC
   AND NOT WIN32)
  message(STATUS "Using GNU gold linker.")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fuse-ld=gold")
endif()
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -ldl")
set(MEGCC_THIRD_PARTY_DIR ${PROJECT_SOURCE_DIR}/../third_party/)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/compiler/Common/Version.h.in
               ${CMAKE_BINARY_DIR}/generate/include/compiler/Common/Version.h @ONLY)
include_directories(${CMAKE_BINARY_DIR}/generate/include/)

add_subdirectory(tools/mgb-tblgen)

add_custom_target(megcc-test)

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)
