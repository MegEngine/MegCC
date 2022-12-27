/**
 * \file
 * compiler/lib/KernelGen/Arm/Arm64/Elemwise/Elemwise.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Elemwise.h"
#include "../ElemwiseHelper/ElemwiseHelper.h"
#include "Arm/ArmCommon/InternalKernel.h"
#include "Utils/SymbolHelper.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;

bool ElemwiseKernel::IsAvailable(TContext* ctx) const {
    //! TODO: now only support float type
    int nr_operands = ctx->getAttrInt("nr_operands");
    bool type_ok = true;
    for (int i = 0; i < nr_operands; i++) {
        type_ok &= (ctx->getAttrOprand("operand:" + std::to_string(i)).dtype ==
                    "f32");
    }
    auto mode = ctx->getAttrStr("mode");
    bool mode_ok = mode == "SIGMOID";
    bool ok_input = nr_operands == 2;
    bool usable = type_ok && mode_ok && ok_input;
    return usable;
}

std::string ElemwiseKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "Arm64_kernel_elementwise";
    ss << "_" << context->getAttrStr("mode");
    int nr_operands = context->getAttrInt("nr_operands");
    if (nr_operands == 2) {
        ss << "_unary_vec_vec";
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
    auto InternalKernelFunc = ArmCommon::ExpNeonKernel();

    CC_ASSERT(ElemwiseImpl) << "ElemwiseHelper Create error!\n";
    writer << R"(
        #include <arm_neon.h>
        #include <math.h>
        #include <stdbool.h>
        #include "tensor_util.h"
    )";
    writer << "\n\n";
    writer << "extern " << InternalKernelFunc.GetKernelSignature(ctx) << ";\n";
    writer << R"(
            static const struct {
    float lower_range;
    float upper_range;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
} sigmoid_constants = {
        -18.0f,
        18.0f,
        4.37031012579801e-11f,
        1.15627324459942e-07f,
        6.08574864600143e-05f,
        8.51377133304701e-03f,
        2.48287947061529e-01f,
        6.10247389755681e-13f,
        5.76102136993427e-09f,
        6.29106785017040e-06f,
        1.70198817374094e-03f,
        1.16817656904453e-01f,
        9.93151921023180e-01f,
        0.5f,
};
        )";
    writer << GenCommonRet() << " " << GetKernelSignature(ctx) << "{\n";
    //! input + output = 2, unary case
    if (nr_operands == 2) {
        writer << R"(
        float* input_data0 = inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        float* output_data = outputs[0]->ptr;
        ${ElemwiseImpl(input_data0, output_data)};
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

std::vector<KernelObj> ElemwiseKernel::GetDependInternalSymbol(
        TContext* ctx) const {
    std::vector<KernelObj> depends;
    ArmCommon::ExpNeonKernel kern;
    depends.emplace_back(kern.GetKernelSymbol(ctx), kern.GetKernelBody(ctx),
                         kern.GetBodyGuardBegin(ctx),
                         kern.GetBodyGuardEnd(ctx));
    return depends;
}

// vim: syntax=cpp.doxygen
