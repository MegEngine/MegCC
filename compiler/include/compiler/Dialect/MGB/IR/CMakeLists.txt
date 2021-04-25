set(LLVM_TARGET_DEFINITIONS MGBDialect.td)
mlir_tablegen(MGBDialect.h.inc -gen-op-decls)
mlir_tablegen(MGBDialect.cpp.inc -gen-op-defs)

mlir_tablegen(MGBDialect_attr.h.inc -gen-attrdef-decls)
mlir_tablegen(MGBDialect_attr.cpp.inc -gen-attrdef-defs)

mlir_tablegen(MGBDialect_type.h.inc -gen-typedef-decls)
mlir_tablegen(MGBDialect_type.cpp.inc -gen-typedef-defs)

add_public_tablegen_target(MLIRMGBIncGen)
add_dependencies(mlir-headers MLIRMGBIncGen)
