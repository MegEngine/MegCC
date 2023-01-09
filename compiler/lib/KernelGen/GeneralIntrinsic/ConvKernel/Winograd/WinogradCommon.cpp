/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ConvKernel/Winograd/WinogradCommon.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "WinogradCommon.h"
#include <memory>
#include "Common/ConvKernel.h"
#include "GeneralIntrinsic/Activation.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

std::string WinogradFrameNchw44::GenGetWorkSpaceCode(
        TContext* context, WinogradStrategyBase* strategy) {
    CC_ASSERT(context->getAttrStr("format") == "NCHW44")
            << "format mismatch  now: " << context->getAttrStr("format")
            << ", expect: NCHW44\n";
    auto WeightShape = context->getAttrOprand("operand:1").shape;
    std::stringstream ss;
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        uint32_t PACK_C_SIZE = 4;
        uint32_t Align = 64;
        uint32_t KernelSize = ${KernelSize};
        uint32_t OutputBlockSize = ${OutputBlockSize};
        uint32_t Alpha = OutputBlockSize + KernelSize - 1;

        const Layout in_layout = inputs[1]->layout;

        size_t OC = in_layout.dims[0] * PACK_C_SIZE;
        size_t IC = in_layout.dims[1] * PACK_C_SIZE;
        if (in_layout.nr_dim == 7) {
            OC = in_layout.dims[1] * PACK_C_SIZE;
            IC = in_layout.dims[2] * PACK_C_SIZE;
        }

        //! input : (alpha, alpha, unit_tile_size, IC) or (alpha, alpha,
        //! ICB, unit_tile_size, IC_BLOCK_SIZE)
        size_t input_transform_buf_size =
                Alpha * Alpha * IC * ${tile_per_loop} * sizeof(float);
        input_transform_buf_size = 
                (input_transform_buf_size + Align -1) / Align * Align;

        //! output : (alpha, alpha, unit_tile_size, OC) or
        //! (alpha, alpha, OCB, unit_tile_size, OC_BLOCK_SIZE)
        size_t output_transform_buf_size =
                Alpha * Alpha * OC * ${tile_per_loop} * sizeof(float);
        output_transform_buf_size = 
                (output_transform_buf_size + Align -1) / Align * Align;

        size_t transform_mid_buf_size = 2 * Alpha * Alpha * sizeof(float) *
                PACK_C_SIZE;
        transform_mid_buf_size = (transform_mid_buf_size + Align -1) / Align * Align; 
        *workspace = input_transform_buf_size + output_transform_buf_size
        + transform_mid_buf_size;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("tile_per_loop", m_tile_per_loop)
                    .add("KernelSize", strategy->GetKernelSize())
                    .add("OutputBlockSize", strategy->GetOutputBlockSize())
                    .render(workspace_temp);
    return ss.str();
}

std::string WinogradFrameNchw44::GenInitCode(TContext*,
                                             WinogradStrategyBase* strategy) {
    uint32_t nr_out_weight = 1;
    std::string common_def = R"(
    int PACK_C_SIZE = 4;
    Tensor* in_weights = inputs[1];
    Layout in_layout = inputs[1]->layout;

    uint32_t OC, IC, Group;
    if(in_layout.nr_dim == 7){
        OC = in_layout.dims[1] * PACK_C_SIZE;
        IC = in_layout.dims[2] * PACK_C_SIZE;
        Group = in_layout.dims[0];
    } else {
        OC = in_layout.dims[0] * PACK_C_SIZE;
        IC = in_layout.dims[1] * PACK_C_SIZE;
        Group = 1;
    }
    uint32_t KernelSize = ${KernelSize};
    uint32_t OutputBlockSize = ${OutputBlockSize};
    uint32_t Alpha = OutputBlockSize + KernelSize - 1;)";

    std::stringstream common_writer;
    common_writer << StringTemplate::StringTemplateArgs()
                             .add("KernelSize", strategy->GetKernelSize())
                             .add("OutputBlockSize",
                                  strategy->GetOutputBlockSize())
                             .render(common_def);
    common_def = common_writer.str();

    std::string fill_weight_attr = R"(
    out_weights->layout.nr_dim = 4;
    out_weights->layout.dims[0] = Group;
    out_weights->layout.dims[1] = Alpha * Alpha;
    out_weights->layout.dims[2] = OC;
    out_weights->layout.dims[3] = IC;
    out_weights->layout.stride[0] = out_weights->layout.dims[1] * out_weights->layout.dims[2] * out_weights->layout.dims[3];
    out_weights->layout.stride[1] = out_weights->layout.dims[2] * out_weights->layout.dims[3];
    out_weights->layout.stride[2] = out_weights->layout.dims[3];
    out_weights->layout.stride[3] = 1;
    out_weights->dtype.type_enum=TinyNN_FLOAT;
    out_weights->name = in_weights->name;)";

    std::string fill_weight_transform = R"(
    float* outptr = out_weights->ptr;
    float* inptr = in_weights->ptr;
    {
    ${FilterTransform(inptr, outptr, OC, IC)}
    }
    )";
    std::stringstream transform_writer;
    transform_writer << StringTemplate::StringTemplateArgs()
                                .add("FilterTransform",
                                     [&](std::vector<std::string> strs) {
                                         return strategy->WeightTrans(strs);
                                     })
                                .render(fill_weight_transform);

    fill_weight_transform = transform_writer.str();

    std::stringstream ss;
    ss << StringTemplate::render_init_body(nr_out_weight, fill_weight_attr,
                                           fill_weight_transform, common_def);

    return ss.str();
}

std::string WinogradFrameNchw44::GenKernelBodyCode(
        TContext* ctx, WinogradStrategyBase* strategy) {
    std::stringstream writer;
    std::string framework = R"(
    //! weights is transformed
    Tensor* weight = inputs[1];
    Layout weight_layout = inputs[1]->layout;

    Tensor* input = inputs[0];
    Layout in_layout = inputs[0]->layout;

    Tensor* output = outputs[0];
    Layout out_layout = outputs[0]->layout;

    const uint32_t PACK_C_SIZE = 4;
    const uint32_t Align = 64;

    size_t N = out_layout.dims[0];
    size_t OC = out_layout.dims[1] * PACK_C_SIZE;
    size_t IC = in_layout.dims[1] * PACK_C_SIZE;
    size_t IH = in_layout.dims[2];
    size_t IW = in_layout.dims[3];
    size_t OH = out_layout.dims[2];
    size_t OW = out_layout.dims[3];
    size_t PH = ${pad_h};
    size_t PW = ${pad_w};

    uint32_t Group = 1;
    if(in_layout.nr_dim == 7){
        Group = weight_layout.dims[0];
    }
    uint32_t KernelSize = ${KernelSize};
    uint32_t OutputBlockSize = ${OutputBlockSize};
    uint32_t Alpha = OutputBlockSize + KernelSize - 1;

    uint32_t tiles_h = (OH + OutputBlockSize -1) / OutputBlockSize;
    uint32_t tiles_w = (OW + OutputBlockSize -1) / OutputBlockSize;
    uint32_t nr_tiles = tiles_h * tiles_w;
    uint32_t nr_tiles_per_loop = ${nr_tiles_per_loop};

    size_t input_transform_buf_size =
                Alpha * Alpha * IC * nr_tiles_per_loop * sizeof(float);
    input_transform_buf_size = 
                (input_transform_buf_size + Align -1) / Align * Align;
    
    size_t output_transform_buf_size =
                Alpha * Alpha * OC * nr_tiles_per_loop * sizeof(float);
    output_transform_buf_size = 
                (output_transform_buf_size + Align -1) / Align * Align;

    float* transform_input_ptr = workspace->ptr;
    float* transform_output_ptr = transform_input_ptr +
                        input_transform_buf_size / sizeof(float);
    
    float* transform_mid_ptr = transform_output_ptr +
                        output_transform_buf_size / sizeof(float);

    const float* input_ptr = input->ptr;
    const float* weight_ptr = weight->ptr;
    float* output_ptr = output->ptr;
    const float* bias_ptr = ${BiasPtr};

    size_t group_input_offset = IC * IH * IW;
    size_t group_weight_offset = Alpha * Alpha * OC * IC;
    size_t group_output_offset = OC * OH * OW;

    for(uint32_t n = 0; n < N; n++){
        for (uint32_t group = 0; group < Group; group++){
            const float* wptr = weight_ptr + group * group_weight_offset;
            const float* inptr = input_ptr + (n * Group + group) *
                                 group_input_offset;
            float* outptr = output_ptr + (n * Group + group)* group_output_offset;
            const float* bptr = NULL;
            if(bias_ptr) bptr = bias_ptr + group * OC;

            for(uint32_t tile_id = 0; tile_id < nr_tiles; tile_id += nr_tiles_per_loop) {
                    uint32_t nr_tiles_in_loop = nr_tiles_per_loop > nr_tiles -
                                tile_id? nr_tiles - tile_id : nr_tiles_per_loop;

                    //! input transform BTdB
                    {
                    ${InputTransform(inptr, transform_input_ptr, IH, IW, IC, PH, PW, tile_id, nr_tiles_in_loop)}
                    }

                    //! batched Matmul
                    const float* A_ptr = wptr;
                    float* B_ptr = transform_input_ptr;
                    float* C_ptr = transform_output_ptr;
                    uint32_t LDA = IC * PACK_C_SIZE;
                    uint32_t LDB = nr_tiles_in_loop * PACK_C_SIZE;
                    uint32_t LDC = nr_tiles_in_loop * PACK_C_SIZE;
                    {
                    ${BatchedMatmul(A_ptr, LDA, B_ptr, LDB, C_ptr, LDC, OC, IC, nr_tiles_in_loop)}
                    }

                    //! output transform: ATmA
                    {
                    ${OutputTransform(transform_output_ptr, outptr, bias_ptr, OH, OW, OC, tile_id, nr_tiles_in_loop)}
                    }

                }
        }
    })";
    std::string bias_ptr = ConvImpl::is_bias(ctx) ? "inputs[2]->ptr" : "NULL";
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add("KernelSize", strategy->GetKernelSize())
                      .add("OutputBlockSize", strategy->GetOutputBlockSize())
                      .add("nr_tiles_per_loop", m_tile_per_loop)
                      .add("BiasPtr", bias_ptr)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add("InputTransform",
                           [&](std::vector<std::string> strs) {
                               return strategy->InputFeatureTrans(strs);
                           })
                      .add("BatchedMatmul",
                           [&](std::vector<std::string> strs) {
                               return strategy->BatchedMatMul(strs);
                           })
                      .add("OutputTransform",
                           [&](std::vector<std::string> strs) {
                               return strategy->OutputFeatureTrans(strs, ctx);
                           })
                      .render(framework);
    return writer.str();
}

// vim: syntax=cpp.doxygen
