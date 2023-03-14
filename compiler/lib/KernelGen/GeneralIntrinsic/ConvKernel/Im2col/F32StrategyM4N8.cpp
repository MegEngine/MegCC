/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Im2col/F16StrategyM8N8.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "F32StrategyM4N8.h"
#include <string>
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
KernelGen::InternalKernelFunc* F32StrategyM4N8::GetInnerCtxMatmul(TContext*) {
    return &m_inner_gemm;
}

std::string F32StrategyM4N8::GetInnerCtxMatmulSym(TContext* ctx) {
    return m_inner_gemm.GetKernelSymbol(ctx) + "_naked";
}