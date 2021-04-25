set(LLVM_TARGET_DEFINITIONS ${CMAKE_CURRENT_SOURCE_DIR}/IR/ops.td)
tablegen(MGB enum_reflection.h.inl "--gen-enum-reflection")
add_public_tablegen_target(MGBEnumReflectionGen)
