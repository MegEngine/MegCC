/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/InternalKernel.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "InternalKernel.h"
#include "Utils/Utils.h"
using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;
const std::string MatmulInternal::m_packa_workspace_call =
        "(int y0, int ymax, int k0, int kmax)";
const std::string MatmulInternal::m_packb_workspace_call =
        "(int x0, int xmax, int k0, int kmax)";
const std::string MatmulInternal::m_workspace_call =
        "(int y0, int ymax, int x0, int xmax, int k0, int kmax)";

std::string MatmulInternal::GenNakedKernelCall(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_float_dtype(dtype)) {
        return R"((const float* pack_a, const float* pack_b, float* C,
            size_t LDC, size_t M, size_t N, size_t K, const float* bias_ptr))";
    } else if (Utils::is_quant_dtype(dtype, 8)) {
        std::string last_dtype = "si8";
        if (ctx->haveAttr("last_dtype")) {
            last_dtype = ctx->getAttrStr("last_dtype");
        }
        if (Utils::is_int_dtype(last_dtype, 32)) {
            return R"((const int8_t* pack_a, const int8_t* pack_b, int* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, void* workspace, float scale, float temp_scale, float dst_scale_inv))";
        } else {
            return R"((const int8_t* pack_a, const int8_t* pack_b, int8_t* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, void* workspace, float scale, float temp_scale, float dst_scale_inv))";
        }
    } else if (dtype == "8832") {
        return R"((const int8_t* pack_a, const int8_t* pack_b, int32_t* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, float scale))";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string MatmulInternal::GenKernelCall(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_float_dtype(dtype)) {
        return R"((const float* A, size_t LDA, const float* B, size_t LDB, float* C,
            size_t LDC, size_t M, size_t N, size_t K, const float* bias_ptr, void* workspace))";
    } else if (Utils::is_quant_dtype(dtype, 8)) {
        std::string last_dtype = "si8";
        if (ctx->haveAttr("last_dtype")) {
            last_dtype = ctx->getAttrStr("last_dtype");
        }
        if (Utils::is_int_dtype(last_dtype, 32)) {
            return R"((const int8_t* A, size_t LDA, const int8_t* B, size_t LDB, int* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, void* workspace, float scale, float temp_scale, float dst_scale_inv))";
        } else {
            return R"((const int8_t* A, size_t LDA, const int8_t* B, size_t LDB, int8_t* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, void* workspace, float scale, float temp_scale, float dst_scale_inv))";
        }
    } else if (dtype == "8832") {
        return R"((const int8_t* A, size_t LDA, const int8_t* B, size_t LDB, int32_t* C,
            size_t LDC, size_t M, size_t N, size_t K, const int32_t* bias_ptr, void* workspace, float scale))";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string MatmulInternal::GenPackACall(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_float_dtype(dtype)) {
        return "(float* outptr, const float* inptr, int ldin, int y0, int "
               "ymax, int k0, int kmax)";
    } else if (Utils::is_quant_dtype(dtype, 8) || dtype == "8832") {
        return "(int8_t* outptr, const int8_t* inptr, int ldin, int y0, int "
               "ymax, int k0, int kmax)";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}

std::string MatmulInternal::GenPackBCall(TContext* ctx) {
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_float_dtype(dtype)) {
        return "(float* outptr, const float* inptr, int ldin, int x0, int "
               "xmax, int k0, int kmax)";
    } else if (Utils::is_quant_dtype(dtype, 8) || dtype == "8832") {
        return "(int8_t* outptr, const int8_t* inptr, int ldin, int x0, int "
               "xmax, int k0, int kmax)";
    } else {
        CC_ABORT << "not support dtype " << dtype << "\n";
    }
    return "";
}
