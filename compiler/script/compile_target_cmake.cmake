cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")
project(compile_target)
option(MEGCC_COMPILER_KERNEL_WITH_ASAN "Enable asan check" OFF)
file(GLOB_RECURSE TARGET_FILE ./*.c ./*.cpp ./*.o)
add_library(compile_target STATIC ${TARGET_FILE})
target_compile_options(
  compile_target
  PRIVATE -DENABLE_ASSERT=1 -DENABLE_LOG=1 -Werror -Wno-format-zero-length
          $<$<COMPILE_LANGUAGE:C>:-Werror=implicit-function-declaration>)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64)
  if(MEGCC_COMPILER_KERNEL_ENABLE_FP16)
    target_compile_options(compile_target PRIVATE -march=armv8.2-a+fp16+dotprod)
  endif()
endif()
if(MEGCC_COMPILER_KERNEL_WITH_ASAN)
  target_compile_options(compile_target PRIVATE -g -O0 -fsanitize=address)
  target_link_libraries(compile_target PRIVATE -fsanitize=address)
else()
  if("${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
    target_compile_options(compile_target PRIVATE -g -O0)
  else()
    if(ANDROID
       OR IOS
       OR RISCV_CROSS_BUILD_ARCH STREQUAL "riscv64")
      target_compile_options(compile_target PRIVATE -Ofast -fno-finite-math-only)
    else()
      target_compile_options(compile_target PRIVATE -O3 -fno-finite-math-only)
    endif()
  endif()
endif()

target_include_directories(
  compile_target
  PRIVATE ${MEGCC_COMPILER_DIR} ${MEGCC_COMPILER_DIR}/../runtime/src/
          ${MEGCC_COMPILER_DIR}/../runtime/include
          ${MEGCC_COMPILER_DIR}/../immigration/include)
install(
  TARGETS compile_target
  EXPORT compile_target
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} COMPONENT compile_target)
