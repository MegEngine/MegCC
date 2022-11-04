set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
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

set(MEGCC_INSTALLED_MLIR_DIR
    ""
    CACHE STRING "path of installed mlir")

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
  add_subdirectory(${PROJECT_SOURCE_DIR}/../../../third_party/llvm-project/llvm
                   ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm EXCLUDE_FROM_ALL)
  set(CMAKE_BUILD_TYPE
      "${_CMAKE_BUILD_TYPE}"
      CACHE STRING "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)

  list(APPEND CMAKE_MODULE_PATH "${CMAKE_BINARY_DIR}/lib/cmake/mlir")
  list(APPEND CMAKE_MODULE_PATH
       "${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/lib/cmake/llvm/")

  list(APPEND LLVM_INCLUDE_DIRS
       ${PROJECT_SOURCE_DIR}/../../../third_party/llvm-project/llvm/include
       ${CMAKE_BINARY_DIR}/third_party/llvm-project/llvm/include)
  list(APPEND MLIR_INCLUDE_DIRS
       ${PROJECT_SOURCE_DIR}/../../../third_party/llvm-project/mlir/include
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
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

file(GLOB_RECURSE CODE_GEN_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_mlir_dialect_library(
  MegCCCodeGen
  ${CODE_GEN_SRC}
  LINK_LIBS
  PUBLIC
  MLIRLinalg
  MLIRLinalgTransforms
  MLIRTosa
  MLIRTosaTransforms
  MLIROptLib
  MLIREmitC
  MLIRTosaToLinalg
  MLIRSCFToStandard
  MLIRMemRefToLLVM
  MLIRReconcileUnrealizedCasts
  MLIRExecutionEngine)
