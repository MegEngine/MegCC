/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Fp16lWinogradNchw44.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <memory>
#include "ConvKernel.h"
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/InternalKernel/InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
bool is_available(TContext* ctx) {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == 3 && ctx->getAttrUInt("kernel_w") == 3 &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            ctx->getAttrUInt("stride_h") == 1 && ctx->getAttrUInt("dilate_h") == 1 &&
            ctx->getAttrUInt("dilate_w") == 1;

    bool param_mode_ok = ctx->getAttrStr("sparse") == "DENSE" &&
                         ctx->getAttrStr("format") == "NCHW88" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";

    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = ctx->getAttrInt("nr_operands") >= 3 &&
                   ctx->getAttrOprand("operand:0").dtype == "f16" &&
                   ctx->getAttrOprand("operand:1").dtype == "f16" &&
                   ctx->getAttrOprand("operand:2").dtype == "f16";

    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 8;

    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok;
}
std::string get_header() {
    std::stringstream writer;
    writer << R"(
#include <math.h>
#include "gi_float.h"
#include "gi_float16.h"
#include "unroll_macro.h"

)";
    return writer.str();
}
}  // namespace
// 23
bool WinogradFp16F23NCHW88::IsAvailable(TContext* ctx) const {
    return is_available(ctx);
}

std::string WinogradFp16F23NCHW88::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << GenCommonRet() << " " << GetInitSignature(ctx) << "{\n";
    writer << m_framework.GenInitCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F23NCHW88::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream writer;
    writer << GenCommonRet() << " " << GetWorkspaceSignature(ctx) << "{\n";
    writer << m_framework.GenGetWorkSpaceCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F23NCHW88::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << "extern " << Fp16MatmulM8N8MK8Kernel().GetKernelSignature(ctx) << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    writer << m_framework.GenKernelBodyCode(ctx, &m_winograd_strategy);
    writer << "return TinyNN_SUCCESS;\n}";
    return writer.str();
}

std::vector<KernelObj> WinogradFp16F23NCHW88::GetDependInternalSymbol(TContext*) const {
    auto matmul = Fp16MatmulM8N8MK8Kernel();
    return {
            {matmul.GetKernelSymbol(nullptr), matmul.GetKernelBody(nullptr),
             matmul.GetBodyGuardBegin(nullptr), matmul.GetBodyGuardEnd(nullptr),
             matmul.GetDependInternalSymbol(nullptr)}};
}

std::string WinogradFp16F23NCHW88::GetKernelSymbol(TContext* context) const {
    auto symbol = GIConvImpl::GetKernelSymbol(context);
    return symbol + "_winograd_f23";
}

// 43
bool WinogradFp16F43NCHW88::IsAvailable(TContext* ctx) const {
    return is_available(ctx);
}

std::string WinogradFp16F43NCHW88::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << GenCommonRet() << " " << GetInitSignature(ctx) << "{\n";
    writer << m_framework.GenInitCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F43NCHW88::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream writer;
    writer << GenCommonRet() << " " << GetWorkspaceSignature(ctx) << "{\n";
    writer << m_framework.GenGetWorkSpaceCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F43NCHW88::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << "extern " << Fp16MatmulM8N8MK8Kernel().GetKernelSignature(nullptr)
           << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    writer << m_framework.GenKernelBodyCode(ctx, &m_winograd_strategy);
    writer << "return TinyNN_SUCCESS;\n}";
    return writer.str();
}

std::vector<KernelObj> WinogradFp16F43NCHW88::GetDependInternalSymbol(TContext*) const {
    auto matmul = Fp16MatmulM8N8MK8Kernel();
    return {
            {matmul.GetKernelSymbol(nullptr), matmul.GetKernelBody(nullptr),
             matmul.GetBodyGuardBegin(nullptr), matmul.GetBodyGuardEnd(nullptr),
             matmul.GetDependInternalSymbol(nullptr)}};
}

std::string WinogradFp16F43NCHW88::GetKernelSymbol(TContext* context) const {
    auto symbol = GIConvImpl::GetKernelSymbol(context);
    return symbol + "_winograd_f43_fp16";
}

// 63
bool WinogradFp16F63NCHW88::IsAvailable(TContext* ctx) const {
    return is_available(ctx);
}

std::string WinogradFp16F63NCHW88::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << GenCommonRet() << " " << GetInitSignature(ctx) << "{\n";
    writer << m_framework.GenInitCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F63NCHW88::GetWorkspaceBody(TContext* ctx) const {
    std::stringstream writer;
    writer << GenCommonRet() << " " << GetWorkspaceSignature(ctx) << "{\n";
    writer << m_framework.GenGetWorkSpaceCode(ctx, &m_winograd_strategy);
    writer << "\n}";
    return writer.str();
}

std::string WinogradFp16F63NCHW88::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << get_header();
    writer << "extern " << Fp16MatmulM8N8MK8Kernel().GetKernelSignature(nullptr)
           << ";\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    writer << m_framework.GenKernelBodyCode(ctx, &m_winograd_strategy);
    writer << "return TinyNN_SUCCESS;\n}";
    return writer.str();
}

std::vector<KernelObj> WinogradFp16F63NCHW88::GetDependInternalSymbol(TContext*) const {
    auto matmul = Fp16MatmulM8N8MK8Kernel();
    return {
            {matmul.GetKernelSymbol(nullptr), matmul.GetKernelBody(nullptr),
             matmul.GetBodyGuardBegin(nullptr), matmul.GetBodyGuardEnd(nullptr),
             matmul.GetDependInternalSymbol(nullptr)}};
}

std::string WinogradFp16F63NCHW88::GetKernelSymbol(TContext* context) const {
    auto symbol = GIConvImpl::GetKernelSymbol(context);
    return symbol + "_winograd_f63_fp16";
}
// vim: syntax=cpp.doxygen
