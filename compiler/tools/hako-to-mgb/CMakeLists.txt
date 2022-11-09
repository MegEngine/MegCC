get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
file(GLOB_RECURSE SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.h
     ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_llvm_executable(hako-to-mgb ${SRCS} NO_INSTALL_RPATH)
llvm_update_compile_flags(hako-to-mgb)
target_link_libraries(hako-to-mgb PRIVATE ${dialect_libs} Common HakoParse)
mlir_check_all_link_libraries(hako-to-mgb)
