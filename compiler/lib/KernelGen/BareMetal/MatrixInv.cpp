/**
 * \file
 * compiler/lib/KernelGen/BareMetal/MatrixInv.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "MatrixInv.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool MatrixInvKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32";
    auto src_shape = context->getAttrOprand("operand:0").shape;
    bool ok_shape =
            src_shape.size() >= 2 &&
            src_shape[src_shape.size() - 1] == src_shape[src_shape.size() - 2];
    return ok_dtype && ok_shape;
}

//! kernel gen
std::string MatrixInvKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_matrix_inv_f32";
    return ss.str();
}

std::string MatrixInvKernel::GetWorkspaceBody(TContext* context) const {
    std::stringstream ss;
    ss << GenCommonRet() << " " << GetWorkspaceSignature(context);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        const Layout in_layout = inputs[0]->layout;
        const uint32_t n = in_layout.dims[in_layout.nr_dim - 1];
        *workspace = 2 * n * n * sizeof(float);
        return TinyNN_SUCCESS;
    })";
    ss << workspace_temp;
    return ss.str();
}

std::string MatrixInvKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    ss << R"(
        #include <math.h>
        #include <string.h>
    )";
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    float* a_data = (float*)inputs[0]->ptr;
    float* c_data = (float*)outputs[0]->ptr;
    TINYNN_ASSERT(a_data);
    TINYNN_ASSERT(c_data);
    const Tensor* a_tensor = inputs[0];
    const Tensor* c_tensor = outputs[0];
    const Layout a_layout = a_tensor->layout;
    const int n = a_layout.dims[a_layout.nr_dim - 1];
    const int ld_buffer = 2 * n;
    int batch = 1;
    for (int i = 0; i < a_layout.nr_dim - 2; ++i) {
        batch *= a_layout.dims[i];
    }
    float* src_buffer = (float*)(workspace->ptr);
    for (int b_idx = 0; b_idx < batch; ++b_idx){
        float* batch_src = a_data + b_idx * n * n;
        float* batch_dst = c_data + b_idx * n * n;
        for (int row = 0; row < n; ++row){
            memcpy(&src_buffer[row * ld_buffer], batch_src + row * n, sizeof(float) * n);
            memset(&src_buffer[row * ld_buffer] + n, 0, sizeof(float) * n);
            src_buffer[row * ld_buffer + n + row] = 1;
        }

        for (int out_row = 0; out_row < n; ++out_row){
            float abs_max = 0.f;
            int select_row = out_row;
            for (int row = out_row; row < n; ++row){
                float abs_val = fabsf(src_buffer[row * ld_buffer + out_row]);
                if (abs_val > abs_max){
                    abs_max = abs_val;
                    select_row = row;
                }
            }
            TINYNN_ASSERT(abs_max > 1e-7);
            for(int col = 0; col < 2 * n; ++col){
                float temp = src_buffer[out_row * ld_buffer + col];
                src_buffer[out_row * ld_buffer + col] = src_buffer[select_row * ld_buffer + col];
                src_buffer[select_row * ld_buffer + col] = temp;
            }

            // substract pivot row from other rows
            float* pivot_row_ptr = &src_buffer[out_row * ld_buffer];
            for (int row = 0; row < n; ++row) {
                if (row == out_row) {
                    continue;
                }
                float inv_pivot = -src_buffer[row * ld_buffer + out_row] / pivot_row_ptr[out_row];
                for (int col = out_row; col < n * 2; ++col) {
                    src_buffer[row * ld_buffer + col] += pivot_row_ptr[col] * inv_pivot;
                }
            }
            // scale pivot row after subtracting it from other rows
            {
                float scale = 1.f / pivot_row_ptr[out_row];
                for (int col = out_row; col < n * 2; ++col) {
                    pivot_row_ptr[col] *= scale;
                }
            }
        }
        for (int row = 0; row < n; ++row) {
            memcpy(batch_dst + row * n, &src_buffer[row * ld_buffer] + n, sizeof(float) * n);
        }
    }

    return TinyNN_SUCCESS;
})";
    ss << body_temp;
    return ss.str();
}

// vim: syntax=cpp.doxygen
