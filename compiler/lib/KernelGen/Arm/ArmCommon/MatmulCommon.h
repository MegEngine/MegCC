/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/MatmulCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <sstream>
#include <string>
#include "ElemwiseHelper/ElemwiseHelper.h"
#include "InternalKernel.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {
namespace {

static inline std::pair<std::string, std::string> gen_postprocess_inline(
        TContext* ctx, bool need_postprocess = true) {
    std::string call_str;
    std::stringstream declare_ss;
    auto nonline_mode = ctx->haveAttr("nonlineMode")
                                ? ctx->getAttrStr("nonlineMode")
                                : "IDENTITY";
    if ((nonline_mode == "SIGMOID" || nonline_mode == "H_SWISH") &&
        need_postprocess) {
        std::vector<CCOperand> operands;
        operands.resize(2);
        auto dtype = ctx->getAttrStr("dtype");
        auto create_elem = [=](std::string src_dtype, std::string dst_dtype)
                -> std::shared_ptr<ElemwiseGenUnary> {
            if (nonline_mode == "SIGMOID") {
                return std::make_shared<ElemwiseGenUnarySigmoid>(
                        src_dtype, dst_dtype, true);
            } else {
                CC_ASSERT(nonline_mode == "H_SWISH");
                return std::make_shared<ElemwiseGenUnaryHswish>(
                        src_dtype, dst_dtype, true);
            }
        };

        std::shared_ptr<ElemwiseGenUnary> ElemwiseImpl =
                create_elem("f32", "f32");

        if (Utils::is_quant_dtype(dtype)) {
            ElemwiseImpl = create_elem("si32", "si8");
        }

        auto ImpleGen = [=](std::vector<std::string> strs) {
            return ElemwiseImpl->GenCodeBody(strs);
        };
        std::string post_process_temp;
        if (ctx->getAttrStr("format") == "NCHW") {
            post_process_temp = R"(
            if (LDC == N){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                for(int m_idx = 0; m_idx < M; ++m_idx){
                    ${ElemwiseImplName}(C + m_idx * LDC, C + m_idx * LDC, N);
                }
            }
        )";
        } else if (ctx->getAttrStr("format") == "MK4") {
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(C + cnt * LDC, C + cnt * LDC, 4 * N);
                    ++cnt;
                }
            }
        )";
        } else {
            CC_ASSERT(ctx->getAttrStr("format") == "MK4_DOT")
                    << ", but get " << ctx->getAttrStr("format") << "\n";
            post_process_temp = R"(
            if (LDC == (4 * N)){
                ${ElemwiseImplName}(gemm_output, C, M * N, temp_scale, dst_scale_inv);
            }else{
                int cnt = 0;
                for(int m_idx = 0; m_idx < M; m_idx += 4){
                    ${ElemwiseImplName}(gemm_output + cnt * LDC, C + cnt * LDC, 4 * N, temp_scale, dst_scale_inv);
                    ++cnt;
                }
            }
        )";
        }
        call_str =
                StringTemplate::StringTemplateArgs()
                        .add("ElemwiseImplName", ElemwiseImpl->GenInlineName())
                        .render(post_process_temp);
        auto InternalKernelFunc = ExpNeonKernel();
        declare_ss << "extern " << InternalKernelFunc.GetKernelSignature(ctx)
                   << ";\n";
        declare_ss << ElemwiseImpl->GenCodeBody({});
    }
    return {declare_ss.str(), call_str};
}

}  // namespace
}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc