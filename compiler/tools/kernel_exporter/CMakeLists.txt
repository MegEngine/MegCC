get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
add_llvm_executable(kernel_exporter exporter_imp.cpp config_attr.cpp utils.cpp
                    tinynn-exporter.cpp)
llvm_update_compile_flags(kernel_exporter)
target_link_libraries(kernel_exporter PRIVATE ${dialect_libs} KernelGen Common)
target_compile_options(Common PUBLIC -fexceptions)
target_compile_options(kernel_exporter PUBLIC -fexceptions)
mlir_check_all_link_libraries(kernel_exporter)
