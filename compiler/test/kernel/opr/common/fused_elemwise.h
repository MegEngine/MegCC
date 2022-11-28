/**
 * \file
 * compiler/test/kernel/opr/commom/fused_elementwise.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <string>
#include "compiler/Common/TContext.h"
#include "megdnn/handle.h"
#include "test/kernel/common/cc_proxy.h"
#include "test/kernel/common/checker.h"
#include "test/kernel/common/dnn_helper.h"
#include "test/kernel/opr/common/elemwise.h"

namespace megcc {
namespace test {
void check_fuse_elemwise(TensorShapeArray shapes,
                         std::vector<std::string> modes,
                         megcc::KernelGen::Arch arch, const std::string& symbol,
                         float epsilon = 1e-3);
}
}  // namespace megcc

// vim: syntax=cpp.doxygen
