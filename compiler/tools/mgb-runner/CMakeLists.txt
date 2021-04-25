add_llvm_executable(mgb-runner main.cpp NO_INSTALL_RPATH)
target_link_libraries(mgb-runner PRIVATE mgb_imported LLVMSupport)
