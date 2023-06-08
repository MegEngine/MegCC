#include "Arm/ArmCommon/ConvKernel/Int8/Winograd/WinogradCommon.h"
#include <memory>
#include "Arm/ArmCommon/Activation.h"
#include "Common/ConvKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace ArmCommon;

std::string WinogradFrameNchw44Int8::GenGetWorkSpaceCode(
        TContext* context, WinogradStrategyBase* strategy) {
    CC_ASSERT(context->getAttrStr("format") == "NCHW44")
            << "Now winograd only support NCHW44 format.\n";
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
        size_t Group = 1;
        if (in_layout.nr_dim == 7) {
            OC = in_layout.dims[1] * PACK_C_SIZE;
            IC = in_layout.dims[2] * PACK_C_SIZE;
            Group = in_layout.dims[0];
        }

        //! input : (alpha, alpha, unit_tile_size, IC) or (alpha, alpha,
        //! ICB, unit_tile_size, IC_BLOCK_SIZE)
        size_t input_transform_buf_size =
                Alpha * Alpha * IC * ${tile_per_loop} * sizeof(int16_t);
        input_transform_buf_size = 
                (input_transform_buf_size + Align -1) / Align * Align;

        //! output : (alpha, alpha, unit_tile_size, OC) or
        //! (alpha, alpha, OCB, unit_tile_size, OC_BLOCK_SIZE)
        size_t output_transform_buf_size =
                Alpha * Alpha * OC * ${tile_per_loop} * sizeof(int32_t);
        output_transform_buf_size = 
                (output_transform_buf_size + Align -1) / Align * Align;
        *workspace = input_transform_buf_size + output_transform_buf_size;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs()
                    .add("tile_per_loop", m_tile_per_loop)
                    .add("KernelSize", strategy->GetKernelSize())
                    .add("OutputBlockSize", strategy->GetOutputBlockSize())
                    .render(workspace_temp);
    return ss.str();
}

std::string WinogradFrameNchw44Int8::GenInitCode(
        TContext*, WinogradStrategyBase* strategy) {
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
                             .add("OutputBlockSize", strategy->GetOutputBlockSize())
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
    out_weights->dtype.type_enum = TinyNN_INT16;
    out_weights->dtype.param.scale = in_weights->dtype.param.scale;
    out_weights->name = in_weights->name;)";

    std::string fill_weight_transform = R"(
    for(size_t group = 0; group < Group; ++group){
        int16_t* outptr = (int16_t*)out_weights->ptr + group * out_weights->layout.stride[0];
        int8_t* inptr = (int8_t*)in_weights->ptr + group * OC * IC * ${KernelSize} * ${KernelSize};
        {
        ${FilterTransform(inptr, outptr, OC, IC)}
        }
    }
    )";
    std::stringstream transform_writer;
    transform_writer << StringTemplate::StringTemplateArgs()
                                .add("FilterTransform",
                                     [&](std::vector<std::string> strs) {
                                         return strategy->WeightTrans(strs);
                                     })
                                .add("KernelSize", strategy->GetKernelSize())
                                .render(fill_weight_transform);

    fill_weight_transform = transform_writer.str();

    std::stringstream ss;
    ss << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return ss.str();
}

std::string WinogradFrameNchw44Int8::GenKernelBodyCode(
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

    float bias_scale = input->dtype.param.scale * weight->dtype.param.scale * 0.25f;
    float dst_scale = 1.f / output->dtype.param.scale;

    const uint32_t PACK_C_SIZE = 4;
    const uint32_t MATMUL_PACK_SIZE = 8;
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

    uint32_t Group = weight_layout.dims[0];
    size_t OC_per_group = OC / Group;
    size_t IC_per_group = IC / Group;
    TINYNN_ASSERT(OC_per_group % 8 == 0 && IC_per_group % 8 == 0);
    uint32_t KernelSize = ${KernelSize};
    uint32_t OutputBlockSize = ${OutputBlockSize};
    uint32_t Alpha = OutputBlockSize + KernelSize - 1;

    uint32_t tiles_h = (OH + OutputBlockSize -1) / OutputBlockSize;
    uint32_t tiles_w = (OW + OutputBlockSize -1) / OutputBlockSize;
    uint32_t nr_tiles = tiles_h * tiles_w;
    uint32_t nr_tiles_per_loop = ${nr_tiles_per_loop};

    size_t input_transform_buf_size =
                Alpha * Alpha * IC_per_group * nr_tiles_per_loop * sizeof(int16_t);
    input_transform_buf_size = 
                (input_transform_buf_size + Align -1) / Align * Align;

    int16_t* transform_input_ptr = workspace->ptr;
    int32_t* transform_output_ptr = workspace->ptr + input_transform_buf_size;

    const int8_t* input_ptr = input->ptr;
    const int16_t* weight_ptr = weight->ptr;
    int8_t* output_ptr = output->ptr;
    const int32_t* bias_ptr = ${BiasPtr};

    size_t group_input_offset = IC_per_group * IH * IW;
    size_t group_weight_offset = Alpha * Alpha * OC_per_group * IC_per_group;
    size_t group_output_offset = OC_per_group * OH * OW;
    size_t n_input_offset = group_input_offset * Group;
    size_t n_output_offset = group_output_offset * Group;

    for(uint32_t n = 0; n < N; n++){
        for (uint32_t group = 0; group < Group; group++){
            const int16_t* wptr = weight_ptr + group * group_weight_offset;
            const int8_t* inptr = input_ptr + n * n_input_offset + group * group_input_offset;
            int8_t* outptr = output_ptr + n * n_output_offset + group * group_output_offset;
            const int32_t* bptr = NULL;
            if(bias_ptr) bptr = bias_ptr + group * OC_per_group;

            for(uint32_t tile_id = 0; tile_id < nr_tiles; tile_id += nr_tiles_per_loop) {
                    uint32_t nr_tiles_in_loop = nr_tiles_per_loop > nr_tiles -
                                tile_id? nr_tiles - tile_id : nr_tiles_per_loop;

                    //! input transform BTdB
                    {
                    ${InputTransform(inptr, transform_input_ptr, IH, IW, IC_per_group, PH, PW, tile_id, nr_tiles_in_loop)}
                    }
                    //! batched Matmul
                    const int16_t* A_ptr = wptr;
                    int16_t* B_ptr = transform_input_ptr;
                    int32_t* C_ptr = transform_output_ptr;
                    uint32_t LDA = IC_per_group * MATMUL_PACK_SIZE;
                    uint32_t LDB = nr_tiles_in_loop * MATMUL_PACK_SIZE;
                    uint32_t LDC = nr_tiles_in_loop * MATMUL_PACK_SIZE;
                    {
                    ${BatchedMatmul(A_ptr, LDA, B_ptr, LDB, C_ptr, LDC, OC_per_group, IC_per_group, nr_tiles_in_loop)}
                    }

                    //! output transform: ATmA
                    {
                    ${OutputTransform(transform_output_ptr, outptr, bptr, OH, OW, OC_per_group, tile_id, nr_tiles_in_loop)}
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
