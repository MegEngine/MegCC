set(LLVM_TARGET_DEFINITIONS KernelDialect.td)
mlir_tablegen(KernelDialect.h.inc -gen-op-decls)
mlir_tablegen(KernelDialect.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRKernelIncGen)
add_dependencies(mlir-headers MLIRKernelIncGen)

add_mlir_interface(KernelInterfaces)
