/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Elemwise/Elemwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Elemwise.h"
#include "../ElemwiseHelper/ElemwiseHelper.h"
#include "../InternalKernel.h"
#include "../NeonIntrinCompat.h"
#include "Utils/SymbolHelper.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

bool ElemwiseKernel::IsAvailable(TContext* ctx) const {
    //! TODO: now only support float type
    int nr_operands = ctx->getAttrInt("nr_operands");
    bool type_ok = true;
    for (int i = 0; i < nr_operands; i++) {
        type_ok &= (ctx->getAttrOprand("operand:" + std::to_string(i)).dtype == "f32");
    }
    auto mode = ctx->getAttrStr("mode");
    bool mode_ok = mode == "RELU" || mode == "EXP" || mode == "ADD" ||
                   mode == "H_SWISH" || mode == "SIGMOID" || mode == "SUB" ||
                   mode == "MUL" || mode == "TRUE_DIV" || mode == "FUSE_ADD_RELU" ||
                   mode == "FUSE_MUL_ADD3";
    if (mode == "FUSE_MUL_ADD3") {
        auto bcast_type = ElemwiseGenTernary::GetBcastType(
                ctx->getAttrOprand("operand:0"), ctx->getAttrOprand("operand:1"),
                ctx->getAttrOprand("operand:2"));
        mode_ok = mode_ok && ElemwiseGenTernary::is_available(bcast_type);
    }
    bool ok_input = nr_operands == 4 || nr_operands == 3 || nr_operands == 2;
    bool usable = type_ok && mode_ok && ok_input;
    return usable;
}

std::string ElemwiseKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "ArmCommon_kernel_elementwise";
    ss << "_" << context->getAttrStr("mode");
    int nr_operands = context->getAttrInt("nr_operands");
    if (nr_operands == 2) {
        ss << "_unary_vec_vec";
    } else if (nr_operands == 3) {
        ss << "_binary_";
        ss << ElemwiseHelperFunc::BcastType2String(ElemwiseGenBinary::GetBcastType(
                context->getAttrOprand("operand:0"),
                context->getAttrOprand("operand:1")));
    } else if (nr_operands == 4) {
        ss << "_ternary_";
        ss << ElemwiseHelperFunc::BcastType2String(ElemwiseGenTernary::GetBcastType(
                context->getAttrOprand("operand:0"),
                context->getAttrOprand("operand:1"),
                context->getAttrOprand("operand:2")));
    } else {
        //! Not implement ternary elemwise kernel
        ss << "_invalid_nr_operands_";
    }
    //! TODO: add ternary elemwise
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string ElemwiseKernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    int nr_operands = ctx->getAttrInt("nr_operands");
    auto mode = ctx->getAttrStr("mode");
    std::vector<CCOperand> operands;
    for (int i = 0; i < nr_operands; i++) {
        operands.push_back(ctx->getAttrOprand("operand:" + std::to_string(i)));
    }
    auto ElemwiseImpl = ElemwiseHelperFunc::CreateGenHelper(mode, operands);
    auto InternalKernelFunc = ExpNeonKernel();
    CC_ASSERT(ElemwiseImpl) << "ElemwiseHelper Create error!\n";
    writer << R"(
        #include <arm_neon.h>
        #include <math.h>
        #include <stdbool.h>
        #include "tensor_util.h"
    )";
    writer << gen_neon_intrin_compat();
    writer << "extern " << InternalKernelFunc.GetKernelSignature(ctx) << ";\n";
    writer << "\n\n";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    //! input + output = 2, unary case
    if (nr_operands == 2) {
        writer << R"(
        float* input_data0 = inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        float* output_data = outputs[0]->ptr;
        ${ElemwiseImpl(input_data0, output_data)};
        )";
    } else if (nr_operands == 3) {
        writer << R"(
        float* input_data0 = inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        float* input_data1 = inputs[1]->ptr;
        TINYNN_ASSERT(input_data1);
        float* output_data = outputs[0]->ptr;
        ${ElemwiseImpl(input_data0, input_data1, output_data)};
        )";
    } else if (nr_operands == 4) {
        writer << R"(
        float* input_data0 = inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        float* input_data1 = inputs[1]->ptr;
        TINYNN_ASSERT(input_data1);
        float* input_data2 = inputs[2]->ptr;
        TINYNN_ASSERT(input_data2);
        float* output_data = outputs[0]->ptr;
        ${ElemwiseImpl(input_data0, input_data1, input_data2, output_data)};
        )";
    } else {
        CC_ABORT << "not support ternary elemwise.\n";
    }
    writer << "\nreturn TinyNN_SUCCESS;\n}";

    std::stringstream ss;
    auto ImpleGen = [=](std::vector<std::string> strs) {
        return ElemwiseImpl->GenCodeBody(strs);
    };
    ss << StringTemplate::StringTemplateArgs()
                    .add("ElemwiseImpl", ImpleGen)
                    .render(writer.str());
    return ss.str();
}

std::vector<KernelObj> ElemwiseKernel::GetDependInternalSymbol(TContext* ctx) const {
    auto mode = ctx->getAttrStr("mode");
    std::vector<KernelObj> depends;
    if (mode == "EXP" || mode == "SIGMOID") {
        ExpNeonKernel kern;
        depends.emplace_back(
                kern.GetKernelSymbol(ctx), kern.GetKernelBody(ctx),
                kern.GetBodyGuardBegin(ctx), kern.GetBodyGuardEnd(ctx));
    }
    return depends;
}

// vim: syntax=cpp.doxygen
