# use xxd generate the LibJit.h file
add_custom_command(
  OUTPUT ${CMAKE_BINARY_DIR}/generate/LibJit.h
  COMMAND mkdir -p ${CMAKE_BINARY_DIR}/generate && cd ${MEGCC_THIRD_PARTY_DIR}/tcc/lib
          && xxd -i libtcc1.a ${CMAKE_BINARY_DIR}/generate/LibJit.h
  DEPENDS
  VERBATIM)

file(GLOB_RECURSE ARM_SOURCES_FILE ./Arm/*.cpp ./Arm/*.h)
file(GLOB_RECURSE BARE_SOURCES_FILE ./BareMetal/*.cpp ./BareMetal/*.h)
file(GLOB_RECURSE COMMON_SOURCES_FILE ./Common/*.cpp ./Common/*.h)
file(GLOB_RECURSE UTILS_SOURCES_FILE ./Utils/*.cpp ./Utils/*.h)
file(GLOB_RECURSE JIT_SOURCES_FILE ./Jit/*.cpp ./Jit/*.h)
file(GLOB_RECURSE AUTO_BARE_SOURCES_FILE ./AutoBareMetal/*.cpp ./AutoBareMetal/*.h)
file(GLOB_RECURSE WEBASM_SOURCES_FILE ./WebAssembly/*.cpp ./WebAssembly/*.h)
file(GLOB INTERFACE_SOURCE_FILE ./KernelGen.cpp)
file(GLOB_RECURSE GI_SOURCES_FILE ./GeneralIntrinsic/*.cpp ./GeneralIntrinsic/*.h)

add_library(KernelGenIface STATIC ${INTERFACE_SOURCE_FILE})
if(MEGCC_ENABLE_MLIR_KERN_GEN)
  target_compile_definitions(KernelGenIface PUBLIC MEGCC_ENABLE_MLIR_KERN_GEN=1)
endif()
target_include_directories(KernelGenIface
                           PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
target_include_directories(KernelGenIface PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)

list(
  APPEND
  STATIC_KERNEL
  ${ARM_SOURCES_FILE}
  ${INTERFACE_SOURCE_FILE}
  ${BARE_SOURCES_FILE}
  ${COMMON_SOURCES_FILE}
  ${UTILS_SOURCES_FILE}
  ${JIT_SOURCES_FILE}
  ${GI_SOURCES_FILE}
  ${WEBASM_SOURCES_FILE})
if(MEGCC_ENABLE_MLIR_KERN_GEN)
  list(APPEND STATIC_KERNEL ${AUTO_BARE_SOURCES_FILE})
endif()

add_library(KernelGen STATIC ${STATIC_KERNEL} ${CMAKE_BINARY_DIR}/generate/LibJit.h)
target_include_directories(KernelGen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/)
target_include_directories(KernelGen PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../include/)
target_include_directories(KernelGen PRIVATE ${CMAKE_BINARY_DIR}/generate/)
target_include_directories(KernelGen PRIVATE ${TCC_INCLUDE_DIR} ${RUNTIME_SRC_DIR}
                                             ${RUNTIME_INCLUDE_DIR})
target_compile_options(KernelGen PRIVATE -Werror -Wno-error=variadic-macros
                                         -Wno-variadic-macros)
target_link_libraries(KernelGen PUBLIC ${TCC_INSTALL_LIB})
target_link_libraries(KernelGen PRIVATE KernelGenIface)
if(MEGCC_ENABLE_MLIR_KERN_GEN)
  add_subdirectory("../CodeGen/" ${PROJECT_BINARY_DIR}/kernel_gen_code_gen/)
  target_link_libraries(KernelGen PRIVATE MegCCCodeGen)
  mlir_check_all_link_libraries(KernelGen)
endif()
