#include <sstream>

#include "Common/ElemwiseCommon.h"
#include "ElemwiseKernel.h"
#include "FormatHelper.h"
#include "Fp16Common.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

namespace {

std::string gen_dep_func(std::string mode) {
    std::string res;
    if (mode == "H_SWISH") {
        res = R"(
            static inline float clip(float val, float min, float max){
                if(val < min){
                    val = min;
                }
                if(val > max){
                    val = max;
                }
                return val;
            }
        )";
    }
    return res;
}

std::string gen_unary(std::string mode) {
    if (mode == "RELU") {
        return "val > 0 ? val:0";
    } else if (mode == "EXP") {
        return "expf(val)";
    } else if (mode == "SIGMOID") {
        return " 1 / (1 + expf(-val))";
    } else if (mode == "NEGATE") {
        return "-val";
    } else if (mode == "ROUND") {
        return "roundf(val)";
    } else if (mode == "H_SWISH") {
        return "val * clip(val + 3, 0, 6) / 6";
    } else if (mode == "ABS") {
        return "val > 0? val:-val";
    } else if (mode == "LOG") {
        return "logf(val)";
    } else if (mode == "SILU") {
        return "val / (1 + expf(-val))";
    } else if (mode == "ERF") {
        return "erff(val)";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}

std::string gen_binary(std::string mode) {
    if (mode == "ADD") {
        return "val + val2";
    } else if (mode == "SUB") {
        return "val - val2";
    } else if (mode == "MUL") {
        return "val * val2";
    } else if (mode == "TRUE_DIV") {
        return "val / val2";
    } else if (mode == "FUSE_ADD_RELU") {
        return "(val + val2) > 0? (val + val2):0";
    } else if (mode == "FUSE_ADD_SIGMOID") {
        return "1.f/(1.f+ expf(-(val + val2)))";
    } else if (mode == "FUSE_ADD_TANH") {
        return "tanh(val + val2)";
    } else if (mode == "MAX") {
        return "(val > val2 ? val : val2)";
    } else if (mode == "MIN") {
        return "(val < val2 ? val : val2)";
    } else if (mode == "LT") {
        return "(val < val2)";
    } else if (mode == "LEQ") {
        return "(val <= val2)";
    } else if (mode == "EQ") {
        return "(val == val2)";
    } else if (mode == "FLOOR_DIV") {
        return "floorf(val / val2)";
    } else if (mode == "MOD") {
        //! WARNING: this is just for integer float please use fmod(x,y) in C
        //! and fmodf(x,y) in c++
        return R"(val % val2)";
    } else {
        CC_ABORT << "not support mode " << mode.c_str() << "\n";
    }
    return "";
}
// set stride of the dims with broadcast attribute into 0 stride[i]=0,if
// dims_in[i]!=dims_out[i] or i > nr_dims_in
std::string set_broadcast_stride(std::string in_layout, std::string out_layout) {
    std::string body = R"(
        // get the broadcast location in the given input layout
        size_t ${in_layout}_access_stride[MAX_DIM];
        bool ${in_layout}_access_valid = false;
        for (int i =0; i<${out_layout}.nr_dim; ++i) {
            bool non_broadcast = i<${in_layout}.nr_dim && ${in_layout}.dims[i]==${out_layout}.dims[i];
            size_t stride = i<${in_layout}.nr_dim?${in_layout}.stride[i]:1;
            ${in_layout}_access_stride[i] = non_broadcast?stride:0;  
            ${in_layout}_access_valid = ${in_layout}_access_valid || !non_broadcast;          
        }
    )";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("in_layout", in_layout)
                    .add("out_layout", out_layout)
                    .render(body);
    return ss.str();
}
// broadcast_offset = sum(idx[i]*stride_in[i]),i=0,1,...,nr_dim idx[i] =
// i_remain/out_stride[i]
std::string get_broadcast_offset(std::string in_layout, std::string out_layout) {
    std::string body = R"(
        // get the offset of broadcast layout with given offset of output (i)
        int ${in_layout}_broadcast_offset = 0, ${in_layout}_i_remain = i;
        if(!${in_layout}_access_valid){
            ${in_layout}_broadcast_offset = i;
        }else{
            for(size_t j = 0; j < ${out_layout}.nr_dim; ++j){
                size_t access_stride_val = ${in_layout}_access_stride[j];
                int idx = ${in_layout}_i_remain/${out_layout}.stride[j];
                ${in_layout}_broadcast_offset += idx*access_stride_val;
                ${in_layout}_i_remain %= ${out_layout}.stride[j];
            }
        }
    )";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("in_layout", in_layout)
                    .add("out_layout", out_layout)
                    .render(body);
    return ss.str();
}

/*
    kernel format:
    for(int idx_out = 0; idx_out< nr_elem_out;idx+=block_1){
        base_offset = get_all_offset_of_inputs_in_every_block_1();
        for(int idx_block_1 = 0; idx_block_1< block_1;idx_block_1+=block_0){
            get_all_offset_of_inputs_in_every_block_0();
            offset_block1 = base_offset + idx_block_1*stride_block_1;
            for(int idx_block_0 = 0; idx_block_0< block_0;++idx_block_0){
                offset_input = offset_block1 + idx_block_0*stride_block_0;
                offset_output  = idx_out + idx_block_1 + idx_block_0;
                get_calculate_and_save_the_result_with_given_offset();
            }
        }
    }
*/
std::string compute_tenary_body(TContext* ctx, std::string specifier) {
    auto mode = ctx->getAttrStr("mode");
    auto action = [&](std::vector<std::string> strs) {
        auto data0 = strs[0];
        auto data1 = strs[1];
        auto data2 = strs[2];
        std::stringstream writer;
        if (mode == "FUSE_MUL_ADD3") {
            writer << data0 << " * " << data1 << " + " << data2 << ";";
        } else {
            CC_ABORT << "Not support elemwise mode " << mode << "\n";
        }
        return writer.str();
    };

    std::string body = R"({
        ${specifier}* input_data0 = (${specifier}*)inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        ${specifier}* input_data1 = (${specifier}*)inputs[1]->ptr;
        TINYNN_ASSERT(input_data1);
        ${specifier}* input_data2 = (${specifier}*)inputs[2]->ptr;
        TINYNN_ASSERT(input_data2);
        ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;
        TINYNN_ASSERT(output_data);

        Layout in_layout0 = inputs[0]->layout;
        Layout in_layout1 = inputs[1]->layout;
        Layout in_layout2 = inputs[2]->layout;
        Layout out_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (size_t i = 0; i < out_layout.nr_dim; ++i) {
            nr_elem *= out_layout.dims[i];
        }
        // get the broadcast dims in different shape
        ${set_broadcast_stride(in_layout0, out_layout)}
        ${set_broadcast_stride(in_layout1, out_layout)}
        ${set_broadcast_stride(in_layout2, out_layout)}
        
        // combine the last two  part with the same broadcast validation
        size_t last_dim_idx = out_layout.nr_dim-1, last_dim =1;
        size_t broadcast_shape = 1, broadcast_idx = last_dim_idx;
        bool broadcast_ahead = in_layout0_access_stride[last_dim_idx]==0||in_layout1_access_stride[last_dim_idx]==0;
        broadcast_ahead = broadcast_ahead || in_layout2_access_stride[last_dim_idx]==0;
        bool next_valid =false;
        while(last_dim_idx>0){
            bool broadcast = in_layout0_access_stride[last_dim_idx]==0||in_layout1_access_stride[last_dim_idx]==0;
            broadcast = broadcast || in_layout2_access_stride[last_dim_idx]==0;
            if( broadcast != broadcast_ahead){
                if(next_valid){
                    break;
                }
                broadcast_shape = last_dim;
                broadcast_idx = last_dim_idx;
                broadcast_ahead = broadcast;
                next_valid = true;
            }
            last_dim *= out_layout.dims[last_dim_idx];
            last_dim_idx = last_dim_idx-1;
        }
        last_dim_idx = out_layout.nr_dim-1;
        size_t access_stride_val00 =in_layout0_access_stride[last_dim_idx];
        size_t access_stride_val10 =in_layout1_access_stride[last_dim_idx];
        size_t access_stride_val20 =in_layout2_access_stride[last_dim_idx];

        size_t access_stride_val01 =in_layout0_access_stride[broadcast_idx];
        size_t access_stride_val11 =in_layout1_access_stride[broadcast_idx];
        size_t access_stride_val21 =in_layout2_access_stride[broadcast_idx];
        // calculate the result
        for(size_t i = 0; i < nr_elem; i += last_dim){
            ${get_broadcast_offset(in_layout0, out_layout)}
            ${get_broadcast_offset(in_layout1, out_layout)}
            ${get_broadcast_offset(in_layout2, out_layout)}

            int in_layout0_offset, in_layout1_offset, in_layout2_offset;
            for(size_t k = 0; k< last_dim;k+=broadcast_shape){
                ${specifier} val0, val1, val2;
                if(broadcast_shape ==1){
                    // get the access offset for every element in diffrent input tensor
                    in_layout0_offset =in_layout0_broadcast_offset+k*access_stride_val01;
                    in_layout1_offset =in_layout1_broadcast_offset+k*access_stride_val11;
                    in_layout2_offset =in_layout2_broadcast_offset+k*access_stride_val21;
                    
                    val0 = input_data0[in_layout0_offset];
                    val1 = input_data1[in_layout1_offset];
                    val2 = input_data2[in_layout2_offset];
                    output_data[i+k] = ${act(val0, val1, val2)};
                }else{
                    int offset_k_01 = (k/broadcast_shape)*access_stride_val01;
                    int offset_k_11 = (k/broadcast_shape)*access_stride_val11;
                    int offset_k_21 = (k/broadcast_shape)*access_stride_val21;
                    for(size_t s = 0; s< broadcast_shape;++s){
                        in_layout0_offset =in_layout0_broadcast_offset+offset_k_01+s*access_stride_val00;
                        in_layout1_offset =in_layout1_broadcast_offset+offset_k_11+s*access_stride_val10;
                        in_layout2_offset =in_layout2_broadcast_offset+offset_k_21+s*access_stride_val20;
                        
                        val0 = input_data0[in_layout0_offset];
                        val1 = input_data1[in_layout1_offset];
                        val2 = input_data2[in_layout2_offset];
                        output_data[i+k+s] = ${act(val0, val1, val2)};
                    }
                }
                
            }
        }
        return TinyNN_SUCCESS;
    })";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .add("set_broadcast_stride", set_broadcast_stride)
                    .add("get_broadcast_offset", get_broadcast_offset)
                    .add("act", action)
                    .render(body);
    return ss.str();
}

std::string compute_quater_body(TContext* ctx, std::string specifier) {
    auto mode = ctx->getAttrStr("mode");
    auto action = [&](std::vector<std::string> strs) {
        auto data0 = strs[0];
        auto data1 = strs[1];
        auto data2 = strs[2];
        auto data3 = strs[3];
        std::stringstream writer;
        if (mode == "FUSE_MUL_ADD4") {
            writer << data0 << " * " << data1 << " + " << data2 << "*" << data3 << ";";
        } else {
            CC_ABORT << "Not support elemwise mode " << mode << "\n";
        }
        return writer.str();
    };

    std::string body = R"({
        ${specifier}* input_data0 = (${specifier}*)inputs[0]->ptr;
        TINYNN_ASSERT(input_data0);
        ${specifier}* input_data1 = (${specifier}*)inputs[1]->ptr;
        TINYNN_ASSERT(input_data1);
        ${specifier}* input_data2 = (${specifier}*)inputs[2]->ptr;
        TINYNN_ASSERT(input_data2);
        ${specifier}* input_data3 = (${specifier}*)inputs[3]->ptr;
        TINYNN_ASSERT(input_data3);
        ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;
        TINYNN_ASSERT(output_data);

        Layout in_layout0 = inputs[0]->layout;
        Layout in_layout1 = inputs[1]->layout;
        Layout in_layout2 = inputs[2]->layout;
        Layout in_layout3 = inputs[3]->layout;
        Layout out_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (size_t i = 0; i < out_layout.nr_dim; ++i) {
            nr_elem *= out_layout.dims[i];
        }
        // get the broadcast dims in different shape
        ${set_broadcast_stride(in_layout0, out_layout)}
        ${set_broadcast_stride(in_layout1, out_layout)}
        ${set_broadcast_stride(in_layout2, out_layout)}
        ${set_broadcast_stride(in_layout3, out_layout)}

        // combine the last two contigous part with the same broadcast validation
        size_t last_dim_idx = out_layout.nr_dim-1, last_dim =1;
        size_t broadcast_shape = 1, broadcast_idx = last_dim_idx;
        bool broadcast_ahead = in_layout0_access_stride[last_dim_idx]==0||in_layout1_access_stride[last_dim_idx]==0;
        broadcast_ahead = broadcast_ahead || in_layout2_access_stride[last_dim_idx]==0 || in_layout3_access_stride[last_dim_idx]==0;
        bool next_valid =false;
        while(last_dim_idx>0){
            bool broadcast = in_layout0_access_stride[last_dim_idx]==0||in_layout1_access_stride[last_dim_idx]==0;
            broadcast = broadcast || in_layout2_access_stride[last_dim_idx]==0 || in_layout3_access_stride[last_dim_idx]==0;
            if( broadcast != broadcast_ahead){
                if(next_valid){
                    break;
                }
                broadcast_shape = last_dim;
                broadcast_idx = last_dim_idx;
                broadcast_ahead = broadcast;
                next_valid = true;
            }
            last_dim *= out_layout.dims[last_dim_idx];
            last_dim_idx = last_dim_idx-1;
        }
        //get access stride in different tensor
        last_dim_idx = out_layout.nr_dim-1;
        size_t access_stride_val00 =in_layout0_access_stride[last_dim_idx];
        size_t access_stride_val10 =in_layout1_access_stride[last_dim_idx];
        size_t access_stride_val20 =in_layout2_access_stride[last_dim_idx];
        size_t access_stride_val30 =in_layout3_access_stride[last_dim_idx];

        size_t access_stride_val01 =in_layout0_access_stride[broadcast_idx];
        size_t access_stride_val11 =in_layout1_access_stride[broadcast_idx];
        size_t access_stride_val21 =in_layout2_access_stride[broadcast_idx];
        size_t access_stride_val31 =in_layout3_access_stride[broadcast_idx];

        // calculate the result
        for(size_t i = 0; i < nr_elem; i += last_dim){
            // get the block offset to access value in diffrent input tensor
            ${get_broadcast_offset(in_layout0, out_layout)}
            ${get_broadcast_offset(in_layout1, out_layout)}
            ${get_broadcast_offset(in_layout2, out_layout)}
            ${get_broadcast_offset(in_layout3, out_layout)}

            int in_layout0_offset, in_layout1_offset, in_layout2_offset,in_layout3_offset;
            for(size_t k = 0; k< last_dim;k+=broadcast_shape){
                ${specifier} val0, val1, val2, val3;
                if(broadcast_shape ==1){
                    in_layout0_offset =in_layout0_broadcast_offset+k*access_stride_val01;
                    in_layout1_offset =in_layout1_broadcast_offset+k*access_stride_val11;
                    in_layout2_offset =in_layout2_broadcast_offset+k*access_stride_val21;
                    in_layout3_offset =in_layout3_broadcast_offset+k*access_stride_val31;
                    
                    val0 = input_data0[in_layout0_offset];
                    val1 = input_data1[in_layout1_offset];
                    val2 = input_data2[in_layout2_offset];
                    val3 = input_data3[in_layout3_offset];
                    output_data[i+k] = ${act(val0, val1, val2, val3)};
                }else{
                    int offset_k_01 = (k/broadcast_shape)*access_stride_val01;
                    int offset_k_11 = (k/broadcast_shape)*access_stride_val11;
                    int offset_k_21 = (k/broadcast_shape)*access_stride_val21;
                    int offset_k_31 = (k/broadcast_shape)*access_stride_val31;
                    for(size_t s = 0; s< broadcast_shape;++s){
                        // get the access offset for every element in diffrent input tensor
                        in_layout0_offset =in_layout0_broadcast_offset+offset_k_01+s*access_stride_val00;
                        in_layout1_offset =in_layout1_broadcast_offset+offset_k_11+s*access_stride_val10;
                        in_layout2_offset =in_layout2_broadcast_offset+offset_k_21+s*access_stride_val20;
                        in_layout3_offset =in_layout3_broadcast_offset+offset_k_31+s*access_stride_val30;
                        
                        val0 = input_data0[in_layout0_offset];
                        val1 = input_data1[in_layout1_offset];
                        val2 = input_data2[in_layout2_offset];
                        val3 = input_data3[in_layout3_offset];
                        output_data[i+k+s] = ${act(val0, val1, val2, val3)};
                    }
                }
                
            }
        }
        return TinyNN_SUCCESS;
    })";
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("specifier", specifier)
                    .add("set_broadcast_stride", set_broadcast_stride)
                    .add("get_broadcast_offset", get_broadcast_offset)
                    .add("act", action)
                    .render(body);
    return ss.str();
}

}  // namespace

bool ElmwiseKernel::IsAvailable(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    auto nr_operands = context->getAttrInt("nr_operands");
    auto dtype = context->getAttrOprand("operand:0").dtype;
    bool nr_operands_ok = nr_operands >= 2 && nr_operands <= 5;
    bool mode_ok_unary = mode == "RELU" || mode == "SIGMOID" || mode == "EXP" ||
                         mode == "NEGATE" || mode == "ROUND" || mode == "ABS" ||
                         mode == "H_SWISH" || mode == "LOG" || mode == "SILU" ||
                         mode == "ERF";
    bool mode_ok_binary = mode == "ADD" || mode == "SUB" || mode == "MUL" ||
                          mode == "MAX" || mode == "MIN" || mode == "LEQ" ||
                          mode == "LT" || mode == "FLOOR_DIV" || mode == "EQ" ||
                          mode == "TRUE_DIV" || mode == "FUSE_ADD_RELU" ||
                          mode == "FUSE_ADD_SIGMOID" || mode == "FUSE_ADD_TANH" ||
                          (mode == "MOD" && (dtype == "i32" || dtype == "si32"));
    bool mode_ok_other = mode == "FUSE_MUL_ADD3" || mode == "FUSE_MUL_ADD4";
    return nr_operands_ok && (mode_ok_unary || mode_ok_binary || mode_ok_other);
}

std::string ElmwiseKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_elementwise";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}

std::string ElmwiseKernel::GetKernelBody(TContext* context) const {
    auto op0 = context->getAttrOprand("operand:0");
    auto src_dtype = op0.dtype;
    auto specifier = Utils::cvt_dtype_specifier(src_dtype);

    auto mode = context->getAttrStr("mode");
    std::stringstream writer;
    writer << "#include <math.h> \n";
    writer << "#include <stdbool.h> \n";
    if (specifier == "gi_float16_t")
        writer << gen_fp16_define();
    writer << gen_dep_func(mode);
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);
    if (context->getAttrInt("nr_operands") == 2) {
        std::string unary_str = R"({
                ${specifier}* input_data = (${specifier}*)inputs[0]->ptr;
                TINYNN_ASSERT(input_data);
                ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;
                TINYNN_ASSERT(output_data);

                Layout in_layout = inputs[0]->layout;
                size_t nr_elem = 1;
                for (size_t i = 0; i < in_layout.nr_dim; ++i) {
                    nr_elem *= in_layout.dims[i];
                }
                for(size_t i = 0; i < nr_elem; ++i){
                    ${specifier} val = input_data[i];
                    output_data[i] = ${act};
                }
                return TinyNN_SUCCESS;
                }
                )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("specifier", specifier)
                          .add("act", gen_unary(mode))
                          .render(unary_str);
    } else if (context->getAttrInt("nr_operands") == 3) {
        std::string binary_str = R"({
                ${specifier}* input_data = (${specifier}*)inputs[0]->ptr;
                TINYNN_ASSERT(input_data);
                ${specifier}* input_data_2 = (${specifier}*)inputs[1]->ptr;
                TINYNN_ASSERT(input_data_2);
                ${specifier}* output_data = (${specifier}*)outputs[0]->ptr;
                TINYNN_ASSERT(output_data);

                Layout in_layout_a = inputs[0]->layout;
                Layout in_layout_b = inputs[1]->layout;
                Layout out_layout = outputs[0]->layout;
                size_t nr_elem_a = 1, nr_elem_b = 1, nr_elem = 1;

                for (size_t i = 0; i < in_layout_a.nr_dim; ++i) {
                    nr_elem_a *= in_layout_a.dims[i];
                }
                for (size_t i = 0; i < in_layout_b.nr_dim; ++i) {
                    nr_elem_b *= in_layout_b.dims[i];
                }
                for (size_t i = 0; i < out_layout.nr_dim; ++i) {
                    nr_elem *= out_layout.dims[i];
                }
                if (nr_elem_a == nr_elem && nr_elem_b == nr_elem){
                    for(size_t i = 0; i < nr_elem; ++i){
                        ${specifier} val = input_data[i];
                        ${specifier} val2 = input_data_2[i];
                        output_data[i] = ${act};
                    }
                }else {
                    // get the broadcast dims in different shape 
                    // if dim[i] need broadcast, set stride[i]=0
                    ${set_broadcast_stride(in_layout_a, out_layout)}
                    ${set_broadcast_stride(in_layout_b, out_layout)}

                    // combine the last two part with the same broadcast validation
                    // combine those dims need to broadcast or not(with the same attribute) into one
                    // from the end of shape, do this process twice
                    size_t last_dim_idx = out_layout.nr_dim-1;
                    size_t last_dim = 1;
                    size_t access_stride_a0 = 1, access_stride_b0 = 1 ,access_stride_a1 = 1, access_stride_b1 = 1;
                    size_t broadcast_shape = 1, broadcast_idx = last_dim_idx;
                    bool broadcast_ahead = in_layout_a_access_stride[last_dim_idx]==0||in_layout_b_access_stride[last_dim_idx]==0;
                    bool next_valid =false;
                    while(last_dim_idx>0){
                        bool broadcast = in_layout_a_access_stride[last_dim_idx]==0||in_layout_b_access_stride[last_dim_idx]==0;
                        if( broadcast != broadcast_ahead){
                            if(next_valid){
                                break;
                            }
                            broadcast_shape = last_dim;
                            broadcast_idx = last_dim_idx;
                            broadcast_ahead = broadcast;
                            next_valid = true;
                        }
                        last_dim *= out_layout.dims[last_dim_idx];
                        last_dim_idx = last_dim_idx-1;
                    }
                    last_dim_idx = out_layout.nr_dim-1;
                    access_stride_a0 = in_layout_a_access_stride[last_dim_idx];
                    access_stride_b0 = in_layout_b_access_stride[last_dim_idx];
                    access_stride_a1 = in_layout_a_access_stride[broadcast_idx];
                    access_stride_b1 = in_layout_b_access_stride[broadcast_idx];

                    // calculate the result
                    for(size_t i = 0; i < nr_elem; i+=last_dim){
                        // get the block offset to access value in diffrent input tensor
                        ${get_broadcast_offset(in_layout_a, out_layout)}
                        ${get_broadcast_offset(in_layout_b, out_layout)}
                        int in_layout_a_offset, in_layout_b_offset;
                        for(size_t k = 0; k< last_dim;k+=broadcast_shape){
                            if(broadcast_shape ==1){
                                    in_layout_a_offset =in_layout_a_broadcast_offset+k*access_stride_a1;
                                    in_layout_b_offset =in_layout_b_broadcast_offset+k*access_stride_b1;
                                    ${specifier} val = input_data[in_layout_a_offset];
                                    ${specifier} val2 = input_data_2[in_layout_b_offset];
                                    output_data[i+k] = ${act};
                            }else{
                                int offset_k_a1 = (k/broadcast_shape)*access_stride_a1;
                                int offset_k_b1 = (k/broadcast_shape)*access_stride_b1;
                                int offset_s_a0 = 0, offset_s_b0 = 0;
                                for(size_t s = 0; s< broadcast_shape;++s){
                                    // get the access offset for every element in diffrent input tensor
                                    in_layout_a_offset =in_layout_a_broadcast_offset+offset_k_a1 + offset_s_a0;
                                    in_layout_b_offset =in_layout_b_broadcast_offset+offset_k_b1 + offset_s_b0;

                                    offset_s_a0 +=access_stride_a0;
                                    offset_s_b0 +=access_stride_b0;
                                    ${specifier} val = input_data[in_layout_a_offset];
                                    ${specifier} val2 = input_data_2[in_layout_b_offset];
                                    output_data[i+k+s] = ${act};
                                }
                            }
                        }
                    }
                }
                return TinyNN_SUCCESS;
                }
                )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("set_broadcast_stride", set_broadcast_stride)
                          .add("get_broadcast_offset", get_broadcast_offset)
                          .add("specifier", specifier)
                          .add("act", gen_binary(mode))
                          .render(binary_str);
    } else if (context->getAttrInt("nr_operands") == 4) {
        writer << compute_tenary_body(context, specifier);
    } else {
        CC_ASSERT(context->getAttrInt("nr_operands") == 5);
        writer << compute_quater_body(context, specifier);
    }
    return writer.str();
}

// vim: syntax=cpp.doxygen
