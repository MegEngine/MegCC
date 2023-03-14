/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Im2col/F16StrategyM8N8.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "../Im2colCommon.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
namespace GeneralIntrinsic {

class F32StrategyM4N8 : public Im2colStrategyBase {
public:
    // matmul call
    virtual KernelGen::InternalKernelFunc* GetInnerCtxMatmul(TContext* ctx) override;
    virtual std::string GetInnerCtxMatmulSym(TContext* ctx) override;

private:
    MatmulM4N8MK4Kernel m_inner_gemm;
};

}  // namespace GeneralIntrinsic
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen