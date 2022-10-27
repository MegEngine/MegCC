get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(translation_libs GLOBAL PROPERTY MLIR_TRANSLATION_LIBS)
# since it links with imported megengine library, it should use default RPATH settings or
# llvm would rewrite it but missing correct rpath of libmegengine
add_llvm_executable(megcc-translate megcc-translate.cpp NO_INSTALL_RPATH)
llvm_update_compile_flags(megcc-translate)
target_link_libraries(
  megcc-translate
  PRIVATE ${dialect_libs}
          ${translation_libs}
          MLIRIR
          MLIRParser
          MLIRPass
          MLIRTranslation
          MLIRSupport
          Common)
mlir_check_all_link_libraries(megcc-translate)
