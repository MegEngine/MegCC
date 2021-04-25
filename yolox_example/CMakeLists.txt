cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

project(YOLOX_EXAMPLE)
# find OpenCV
find_package(OpenCV REQUIRED)
message(STATUS "OpenCV library status:")
message(STATUS "    config: ${OpenCV_DIR}")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

if(NOT DEFINED RUNTIME_KERNEL_DIR)
  message(FATAL_ERROR "build MegCC runtime RUNTIME_KERNEL_DIR kernel empty")
else()
  message(STATUS "build MegCC runtime with kernel dir ${RUNTIME_KERNEL_DIR}")
endif()

add_library(TinyNN STATIC IMPORTED)
set_target_properties(TinyNN PROPERTIES
  IMPORTED_LOCATION "${RUNTIME_KERNEL_DIR}/runtime/install/lib/libTinyNN.a")

add_executable(yolox_test main.cpp)

target_include_directories(
  yolox_test
  PUBLIC $<BUILD_INTERFACE:${RUNTIME_KERNEL_DIR}/runtime/install/include>)

target_link_libraries(yolox_test TinyNN m ${OpenCV_LIBS})
install(TARGETS yolox_test LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
