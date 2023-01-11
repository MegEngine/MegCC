cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

project(TinyNNTest C CXX)

message(${PROJECT_SOURCE_DIR})
message(${CMAKE_CURRENT_SOURCE_DIR})

option(TINYNN_ENABLE_ASAN "Build with asan to check memory issue." OFF)
message(STATUS "TinyNN BUILD INFO:")
message(STATUS "\tSystem:                                 ${CMAKE_SYSTEM_NAME}")
message(STATUS "\tProcessor:                              ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "\tEnable asan check:                      ${TINYNN_ENABLE_ASAN}")

message(STATUS "Gen flatcc header.")

set(FLATCC_DIR ${PROJECT_SOURCE_DIR}/../flatcc)
set(FLATCC_EXECUTABLE ${FLATCC_DIR}/bin/flatcc)
set(SCHEMA_SOURCE ${PROJECT_SOURCE_DIR}/../schema/model.fbs)
set(SCHEMA_OUTPUT ${PROJECT_BINARY_DIR}/schema/cheader)

add_custom_command(
  OUTPUT ${SCHEMA_OUTPUT}
  COMMAND mkdir -p ${SCHEMA_OUTPUT}
  COMMAND "${FLATCC_EXECUTABLE}" -cr --reader -o ${SCHEMA_OUTPUT} ${SCHEMA_SOURCE}
  COMMENT "Gen flatcc header."
  DEPENDS ${SCHEMA_SOURCE})

add_custom_target(flatcc_gen_header DEPENDS ${SCHEMA_OUTPUT})

add_subdirectory(${PROJECT_SOURCE_DIR}/../../third_party/googletest
                 ${CMAKE_CURRENT_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)

file(
  GLOB_RECURSE
  SRC
  ./../src/vm/*.c
  ./../src/vm/*.h
  ./../src/device.c
  ./../src/vm.c
  ./../src/init.c
  ./../src/utils.c
  ./../src/lite/global.c
  ./main.cpp
  common/*.cpp
  runtime/*.cpp
  runtime/*.h
  instruction/*.cpp
  instruction/*.h)

file(GLOB_RECURSE OP_FILE ./../src/vm/op.c)
# op.c include kernel.h which is generated
list(REMOVE_ITEM SRC ${OP_FILE})

message(STATUS "All source file is ${SRC}")

add_executable(TinyNNTest ${SRC})
add_dependencies(TinyNNTest flatcc_gen_header)

target_include_directories(
  TinyNNTest
  PUBLIC ${PROJECT_SOURCE_DIR}/../include
         ${PROJECT_SOURCE_DIR}/../src
         ${PROJECT_SOURCE_DIR}
         ${PROJECT_SOURCE_DIR}/instruction
         ./runtime/
         ${SCHEMA_OUTPUT}
         ${PROJECT_SOURCE_DIR}/../../third_party/flatcc/include
         ${PROJECT_SOURCE_DIR}/../../third_part/gtest/include
         ${PROJECT_SOURCE_DIR}/../../immigration/include)

target_link_libraries(TinyNNTest gtest)

if(CMAKE_BUILD_TYPE MATCHES "Debug")
  set(CMAKE_CXX_FLAGS " -g -O0 ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-Ofast -g ${CMAKE_CXX_FLAGS}")
endif()

if(TINYNN_ENABLE_ASAN)
  if(${CMAKE_C_COMPILER_ID} STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS " -fsanitize=address -g -O0 -static-libsan ${CMAKE_CXX_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libsan ${CMAKE_LINK_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS " -fsanitize=address -g -O0 -static-libasan ${CMAKE_CXX_FLAGS}")
    set(CMAKE_LINK_FLAGS "-fsanitize=address -static-libasan ${CMAKE_LINK_FLAGS}")
  endif()
endif()

install(TARGETS TinyNNTest LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
