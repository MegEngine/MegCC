cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

project(TinyNN LANGUAGES C ASM)

option(TINYNN_SHARED_ENABLE "enable TinyNN build with shared library." OFF)
option(TINYNN_ENABLE_ASAN "Build with asan to check memory issue." OFF)
option(TINYNN_ACHIEVE_ALL "Build with achieve all lic static." OFF)
option(TINYNN_SANITY_ALLOC "malloc all tensor dynamic for asan check." OFF)
option(TINYNN_ENABLE_MEMORY_MANAGEMENT "Build with memory management componenet" OFF)
option(TINYNN_DUMP_TENSOR "Build with dump tensor from every layer." OFF)
option(TINYNN_PROFILE_KERNEL "Build with profile kernel" OFF)
option(TINYNN_CALLBACK_ENABLE
       "enable register callback api, like malloc/free/file/time api" OFF)
option(TINYNN_BUILD_FOR_NOT_STANDARD_OS
       "Build deploy for Not standard os, like TEE, Bare_board, RTOS etc." OFF)

if(NOT DEFINED RUNTIME_KERNEL_DIR)
  message(FATAL_ERROR "build MegCC runtime RUNTIME_KERNEL_DIR kernel empty")
else()
  message(STATUS "build MegCC runtime with kernel dir ${RUNTIME_KERNEL_DIR}")
endif()

message(STATUS "TinyNN BUILD INFO:")
message(STATUS "\tSystem:                                 ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor:                              ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tEnable shared libarys:                  ${TINYNN_SHARED_ENABLE}")
message(STATUS "\tEnable asan check:                      ${TINYNN_ENABLE_ASAN}")
message(STATUS "\tEnabel achieve all staitc:              ${TINYNN_ACHIEVE_ALL}")
message(STATUS "\tEnabel alloc tensor dynamic:            ${TINYNN_SANITY_ALLOC}")
message(STATUS "\tEnabel dump tensor to dir:              ${TINYNN_DUMP_TENSOR}")
message(STATUS "\tBuild with register callback:           ${TINYNN_CALLBACK_ENABLE}")
message(
  STATUS "\tBuild with Not standard os:             ${TINYNN_BUILD_FOR_NOT_STANDARD_OS}"
)
message(STATUS "\tBuild kernel dir:                       ${RUNTIME_KERNEL_DIR}")
# as TINYNN_BUILD_FOR_NOT_STANDARD_OS will change some options, so put head of
# CMakeLists.txt as possible
if(TINYNN_BUILD_FOR_NOT_STANDARD_OS)
  message(
    WARNING
      "force disable (TINYNN_SHARED_ENABLE/TINYNN_ENABLE_ASAN/TINYNN_ACHIEVE_ALL) when build for Not standard os"
  )
  # can only build static lib, for example TEE env can not use shared library
  set(TINYNN_SHARED_ENABLE OFF)
  # clang do not build with asan when with --target do not worry about this, you can run
  # same model at linux or android env to test asan issue
  set(TINYNN_ENABLE_ASAN OFF)
  # when build for Not standard os, example binary will be disable have no chance to
  # static link any libc or libm
  set(TINYNN_ACHIEVE_ALL OFF)
  message(
    WARNING
      "force enable TINYNN_CALLBACK_ENABLE when build for TINYNN_BUILD_FOR_NOT_STANDARD_OS"
  )
  # enable TINYNN_CALLBACK_ENABLE to user register it others side, some TEE vendor
  # suggest use customized memory and logger api
  set(TINYNN_CALLBACK_ENABLE ON)
  set(MEGCC_ARCH
      NONE
      CACHE STRING "Architecture build for megcc NOT_STANDARD_OS runtime")
  set(MEGCC_ARCHS
      "aarch64"
      "armv7-a"
      "cortex-m"
      "x86_64"
      "i386"
      "armv7-a-qemu"
      "rv64gcv0p7"
      "rv64norvv")
  if(${MEGCC_ARCH} STREQUAL "NONE")
    message(FATAL_ERROR "need config MEGCC_ARCH, now support one of: ${MEGCC_ARCHS}")
  endif()
  if(${MEGCC_ARCH} IN_LIST MEGCC_ARCHS)
    message(STATUS "build megcc NOT_STANDARD_OS runtime core to: ${MEGCC_ARCH}")
    set(CMAKE_C_FLAGS
        "-DFLATCC_NO_ASSERT=1 -fPIC -fpie -nostdlib -nodefaultlibs ${CMAKE_C_FLAGS}")
  else()
    message(FATAL_ERROR "not support build megcc runtime core to: ${MEGCC_ARCH}")
  endif()
  if(${MEGCC_ARCH} STREQUAL "aarch64")
    message(STATUS "set --target to aarch64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=aarch64")
  elseif(${MEGCC_ARCH} STREQUAL "x86_64")
    message(STATUS "set --target to x86_64")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=x86_64")
  elseif(${MEGCC_ARCH} STREQUAL "rv64gcv0p7")
    # clang will support rvv soon, so we add this.
    message(STATUS "set --target to rv64gcv0p7")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=riscv64 -march=rv64gcv0p7 -mabi=lp64d")
  elseif(${MEGCC_ARCH} STREQUAL "rv64norvv")
    message(STATUS "set --target to rv64norvv")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=riscv64 -march=rv64gc -mabi=lp64d")
  elseif(${MEGCC_ARCH} STREQUAL "i386")
    message(STATUS "set --target to i386")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=i386")
  elseif(${MEGCC_ARCH} STREQUAL "armv7-a")
    message(
      STATUS
        "config more flags for armv7-a, WARNING, use -mfloat-abi=softfp  default, if board support -mfloat-abi=hard, need chang it"
    )
    message(
      STATUS
        "add -fshort-enums if target link show uses 32-bit enums yet the output is to use variable-size enums; use of enum values across objects may fail"
    )
    message(STATUS "set --target to armv7-a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=armv7-a")
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -ffreestanding -mfloat-abi=softfp -mfpu=neon -march=armv7-a")
  elseif(${MEGCC_ARCH} STREQUAL "armv7-a-qemu")
    # for qemu vexpress-a9 caused by it do not support NEON, please do no choose
    # armv7-a-qemu for real armv7 board caused by performance
    message(STATUS "set --target to armv7-a")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=armv7-a")
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -ffreestanding -fshort-enums -march=armv7-a -mcpu=cortex-a9")
  elseif(${MEGCC_ARCH} STREQUAL "cortex-m")
    message(STATUS "set --target to arm")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} --target=arm")
    # may add -mcpu=cortex-mx(-mcpu=cortex-m3) to performance
    set(CMAKE_C_FLAGS
        "${CMAKE_C_FLAGS} -fshort-enums -ffreestanding -mfloat-abi=softfp -mthumb")
  else()
    message(FATAL_ERROR "CMakeLists code issue happened!! please FIXME!!")
  endif()

  if(NOT ${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    message(FATAL_ERROR "only support clang when build for NOT_STANDARD_OS!!!")
  endif()
endif()

message(STATUS "Gen flatcc header.")
set(FLATCC_DIR ${PROJECT_SOURCE_DIR}/flatcc)
set(FLATCC_EXECUTABLE ${FLATCC_DIR}/bin/flatcc)
set(SCHEMA_SOURCE ${PROJECT_SOURCE_DIR}/schema/model.fbs)
set(SCHEMA_OUTPUT ${PROJECT_SOURCE_DIR}/src/cheader)
set(CMAKE_C_FLAGS "-fPIC -fpie ${CMAKE_C_FLAGS}")

add_custom_command(
  OUTPUT ${SCHEMA_OUTPUT}
  COMMAND mkdir -p ${SCHEMA_OUTPUT}
  COMMAND "${FLATCC_EXECUTABLE}" -cr --reader -o ${SCHEMA_OUTPUT} ${SCHEMA_SOURCE}
  COMMENT "Gen flatcc header."
  DEPENDS ${SCHEMA_SOURCE})

add_custom_target(flatcc_gen_header DEPENDS ${SCHEMA_OUTPUT})

message(STATUS "Get immigration kernel.")
set(IMMIGRATION_DIR ${PROJECT_SOURCE_DIR}/../immigration)

file(GLOB_RECURSE IMMIGRATION ${IMMIGRATION_DIR}/include/*.h)

file(GLOB_RECURSE SRC src/*.c src/*.h ${RUNTIME_KERNEL_DIR}/*.h
     ${RUNTIME_KERNEL_DIR}/*.c)

if(TINYNN_SHARED_ENABLE)
  add_library(TinyNN SHARED ${SRC} ${IMMIGRATION})
  set(_VER_FILE ${PROJECT_SOURCE_DIR}/version.ld)
  target_link_options(TinyNN PRIVATE -Wl,--version-script=${_VER_FILE})
  set_target_properties(TinyNN PROPERTIES LINK_DEPENDS ${_VER_FILE})
else()
  add_library(TinyNN STATIC ${SRC} ${IMMIGRATION})
  message(
    WARNING
      "build static tinynn, if you want strip and mangle sym, please run python3 runtime/scripts/strip_and_mangling_static_tinynn.py manually "
  )
endif()

add_dependencies(TinyNN flatcc_gen_header)
set_target_properties(TinyNN PROPERTIES LINKER_LANGUAGE C)

target_include_directories(
  TinyNN
  PUBLIC $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/include>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
         $<BUILD_INTERFACE:${RUNTIME_KERNEL_DIR}>
         $<BUILD_INTERFACE:${SCHEMA_OUTPUT}>
         $<BUILD_INTERFACE:${FLATCC_DIR}/include>
         $<BUILD_INTERFACE:${IMMIGRATION_DIR}/include>)

if(NOT TINYNN_BUILD_FOR_NOT_STANDARD_OS)
  add_executable(tinynn_test_lite example/standard_OS/lite_main.c)
  target_link_libraries(tinynn_test_lite TinyNN m)
endif()

if(TINYNN_ACHIEVE_ALL AND NOT TINYNN_BUILD_FOR_NOT_STANDARD_OS)
  target_link_libraries(tinynn_test_lite -static)
endif()

if(TINYNN_ENABLE_MEMORY_MANAGEMENT)
  set(CMAKE_C_FLAGS " -DENABLE_MEMORY_MANAGEMENT=1 ${CMAKE_C_FLAGS}")
endif()

if(TINYNN_CALLBACK_ENABLE)
  set(CMAKE_C_FLAGS " -DTINYNN_CALLBACK_ENABLE=1 ${CMAKE_C_FLAGS}")
endif()

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_C_FLAGS " -g -O0  ${CMAKE_C_FLAGS} -DDEBUG_MODE=1")
else()
  set(CMAKE_C_FLAGS "-Ofast -g ${CMAKE_C_FLAGS}")
endif()

if(TINYNN_ENABLE_ASAN)
  if(TINYNN_ACHIEVE_ALL)
    message(FATAL_ERROR "Build for TINYNN_ENABLE_ASAN, do not support static libc")
  endif()
  if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_C_FLAGS " -fsanitize=address -g -O0 -static-libsan ${CMAKE_C_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libsan ${CMAKE_LINK_FLAGS}")
  else()
    set(CMAKE_C_FLAGS " -fsanitize=address -g -O0 -static-libasan ${CMAKE_C_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libasan ${CMAKE_LINK_FLAGS}")
  endif()
endif()

if(TINYNN_SANITY_ALLOC)
  set(CMAKE_C_FLAGS " -DTINYNN_SANITY_ALLOC=1 ${CMAKE_C_FLAGS}")
endif()

if(TINYNN_DUMP_TENSOR)
  set(CMAKE_C_FLAGS " -DTINYNN_DUMP_TENSOR=1 ${CMAKE_C_FLAGS}")
endif()

if(TINYNN_PROFILE_KERNEL)
  set(CMAKE_C_FLAGS " -DTINYNN_PROFILE_KERNEL=1 ${CMAKE_C_FLAGS}")
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include DESTINATION ${CMAKE_INSTALL_PREFIX})
install(TARGETS TinyNN LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
if(NOT TINYNN_BUILD_FOR_NOT_STANDARD_OS)
  install(TARGETS tinynn_test_lite LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
endif()
