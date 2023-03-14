/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Im2col/F32StrategyM4N12.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "F32StrategyM4N12.h"
#include <string>
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
GIMatmulInternal* F32StrategyM4N12::GetInnerMatmul(TContext* ctx) {
    auto fmt = ctx->getAttrStr("format");
    if (fmt == "MK4") {
        return &m_inner_mk4_gemm;
    } else {
        CC_ASSERT(fmt == "NCHW");
        return &m_inner_gemm;
    }
}