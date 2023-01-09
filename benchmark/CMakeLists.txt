cmake_minimum_required(VERSION 3.15.2)
set(CMAKE_EXPORT_COMPILE_COMMANDS
    ON
    CACHE INTERNAL "")

project(Benchmarker)

option(ENABLE_MEGENGINE_FRAMEWORK "build benchmark for megengine" OFF)
configure_file(src/build_config.h.in
               ${CMAKE_CURRENT_BINARY_DIR}/genfiles/build_config.h)
# set megcc lib 
if(NOT DEFINED RUNTIME_KERNEL_DIR)
  message(FATAL_ERROR "build MegCC runtime kernel dir RUNTIME_KERNEL_DIR is empty, use -DRUNTIME_KERNEL_DIR=your_model_kernel_dir to set")
else()
  message(STATUS "build MegCC runtime with kernel dir ${RUNTIME_KERNEL_DIR}")
endif()

add_library(TinyNN STATIC IMPORTED)
set_target_properties(
  TinyNN PROPERTIES IMPORTED_LOCATION
                    "${RUNTIME_KERNEL_DIR}/runtime/install/lib/libTinyNN.a")
if(ENABLE_MEGENGINE_FRAMEWORK)
  message(STATUS "build benchmark with megengine ${ENABLE_MEGENGINE_FRAMEWORK}")
  option(X86_BACKEND "Build bechmarker with X86 megengine lib" ON)
  # set megengine lib
  if(NOT DEFINED MEGENGINE_INSTALL_DIR)
    message(FATAL_ERROR "MEGENGINE_INSTALL_DIR is empty use -DMEGENGINE_INSTALL_DIR=your_megengine_install_dir to set")
  else()
    message(STATUS "MEGENGINE_INSTALL_DIR is ${MEGENGINE_INSTALL_DIR}")
  endif()
  add_library(mgb_imported INTERFACE IMPORTED)

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lite/lib/aarch64/liblite_static_all_in_one.a)
    target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS})
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "armv7-a")
    set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lite/lib/armv7/liblite_static_all_in_one.a)
    target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS})
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv64")
    set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lite/lib/riscv64/liblite_static_all_in_one.a)
    target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS})
  else()
    if(X86_BACKEND)
      if(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "AMD64")
        set(MKL_LIBS
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/x86_64/lib/libmkl_core.a
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/x86_64/lib/libmkl_sequential.a
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/x86_64/lib/libmkl_intel_ilp64.a
        )
        set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lite/lib/x86_64/liblite_static_all_in_one.a)
        target_compile_definitions(mgb_imported INTERFACE -DMKL_ILP64)
      # WARNING: i386 is not test locally 
      elseif(${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i386" OR ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "i686")
        set(MKL_LIBS
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/i386/lib/libmkl_core.a
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/x86_64/lib/libmkl_sequential.a
          ${PROJECT_SOURCE_DIR}/../third_party/MegEngine/third_party/mkl/x86_64/lib/libmkl_intel_32.a
        )
        set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lite/lib/i386/liblite_static_all_in_one.a)
      endif()
      set(MKL_DNN_LIBS
        ${MEGENGINE_INSTALL_DIR}/lib/libdnnl.a
        ${MEGENGINE_INSTALL_DIR}/lib/libmkldnn.a
      )

      if(UNIX AND NOT APPLE)
        target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS} ${MKL_DNN_LIBS} -Wl,--start-group -ldl ${MKL_LIBS} -Wl,--end-group) 
      else()
        target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS} ${MKL_DNN_LIBS} ${MKL_LIBS}) 
      endif()
    else()
      set(MGE_INSTALL_LIBS ${MEGENGINE_INSTALL_DIR}/lib/libmegengine.a ${MEGENGINE_INSTALL_DIR}/lib/libflatbuffers.a)
      target_link_libraries(mgb_imported INTERFACE  ${MGE_INSTALL_LIBS})  
    endif()
    
  endif()

  target_include_directories(mgb_imported INTERFACE ${MEGENGINE_INSTALL_DIR}/include)
endif()
# benchmarker config
file(GLOB_RECURSE SOURCES main.cpp src/*.cpp src/*.h)
add_executable(benchmarker ${SOURCES})
target_include_directories(
    benchmarker PUBLIC $<BUILD_INTERFACE:${RUNTIME_KERNEL_DIR}/runtime/install/include> $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/genfiles>)
if(ENABLE_MEGENGINE_FRAMEWORK)
  target_link_libraries(benchmarker -pthread TinyNN mgb_imported)
else()
target_link_libraries(benchmarker -pthread TinyNN)
endif()
message(STATUS "${CMAKE_TOOLCHAIN_FILE}")
if(CMAKE_TOOLCHAIN_FILE)
  if(ANDROID)
    target_link_libraries(benchmarker log)
  endif()
endif()

install(TARGETS benchmarker LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX})
