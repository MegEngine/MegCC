/**
 * \file compiler/tools/kernel_exporter/utils.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include <iostream>

#define EXPORT_ERR(msg)          \
    llvm::outs() << msg << "\n"; \
    __builtin_trap();

std::string ssprintf(const char* fmt, ...);
