#include <iostream>

#define EXPORT_ERR(msg)          \
    llvm::outs() << msg << "\n"; \
    __builtin_trap();

std::string ssprintf(const char* fmt, ...);
