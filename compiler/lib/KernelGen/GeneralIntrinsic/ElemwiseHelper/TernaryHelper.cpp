/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ElemwiseHelper/TernaryHelper.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "ElemwiseHelper.h"
#include "Utils/SymbolHelper.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

namespace {
template <BcastType BT>
std::string TernaryCode();

template <>
std::string TernaryCode<VEC_VEC_VEC>() {
    std::string body = R"(
        size_t SIMD_WIDTH = ${simd_width};
        Layout layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < layout.nr_dim; ++i) {
            nr_elem *= layout.dims[i];
        }
        ${kernel_init()}
        const ${dtype_specifier} * src0 = ${source0};
        const ${dtype_specifier} * src1 = ${source1};
        const ${dtype_specifier} * src2 = ${source2};
        ${dtype_specifier} * dst = ${dst};
        size_t index = 0;
        for(; index + 2*SIMD_WIDTH-1 < nr_elem; index += 2*SIMD_WIDTH) {
            ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
            ${simd_dtype_specifier} vsrc0_1 = ${load_vec}(src0 + SIMD_WIDTH);
            ${simd_dtype_specifier} vsrc1_0 = ${load_vec}(src1);
            ${simd_dtype_specifier} vsrc1_1 = ${load_vec}(src1 + SIMD_WIDTH);
            ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
            ${simd_dtype_specifier} vsrc2_1 = ${load_vec}(src2 + SIMD_WIDTH);
            ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc2_0, vsrc0_1, vsrc1_1, vsrc2_1)}
            src0 += 2*SIMD_WIDTH;
            src1 += 2*SIMD_WIDTH;
            src2 += 2*SIMD_WIDTH;
            dst += 2*SIMD_WIDTH;
        }
        for(; index + SIMD_WIDTH-1 < nr_elem; index += SIMD_WIDTH) {
            ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
            ${simd_dtype_specifier} vsrc1_0 = ${load_vec}(src1);
            ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
            ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0, vsrc2_0)}
            src0 += SIMD_WIDTH;
            src1 += SIMD_WIDTH;
            src2 += SIMD_WIDTH;
            dst += SIMD_WIDTH;
        }
        for(; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, dst, src0, src1, src2)}
            src0 += 1;
            src1 += 1;
            src2 += 1;
            dst += 1;
        })";
    return body;
}

std::string bvbv_calc(
        std::string bvbv_vec_name, std::string src_layout, std::string dst_layout) {
    std::string body = R"(
        {
            int bvb_idx = 0;
            bool is_broadcast = true;
            ${bvbv_vec_name}[0] = 1;
            for (int i = 0; i < ${dst_layout}.nr_dim; ++i) {
                if(!is_broadcast){
                    if(${dst_layout}.dims[i] != ${src_layout}.dims[i]){
                        is_broadcast = true;
                        ++bvb_idx;
                    }
                }else{
                    if(${src_layout}.dims[i] != 1){
                        is_broadcast = false;
                        ++bvb_idx;
                    }
                }
                ${bvbv_vec_name}[bvb_idx] *= ${dst_layout}.dims[i];
            }
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("dst_layout", dst_layout)
            .add("bvbv_vec_name", bvbv_vec_name)
            .add("src_layout", src_layout)
            .render(body);
}

template <>
std::string TernaryCode<VEC_BCAST101_VEC>() {
    std::string body = R"(
        size_t SIMD_WIDTH = ${simd_width};
        Layout dst_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < dst_layout.nr_dim; ++i) {
            nr_elem *= dst_layout.dims[i];
        }
        Layout src_layout_0 = inputs[0]->layout;
        Layout src_layout_1 = inputs[1]->layout;
        Layout src_layout_2 = inputs[2]->layout;
        TINYNN_ASSERT(src_layout_0.nr_dim = src_layout_1.nr_dim);
        size_t bvbv[4] = {1, 1, 1, 1};
        )" + bvbv_calc("bvbv", "src_layout_1", "dst_layout") +
                       R"(
        size_t batch = bvbv[0];
        size_t channel = bvbv[1];
        size_t nr_elem_per_channel = bvbv[2];
        ${kernel_init()}
        const ${dtype_specifier} * src0 = ${source0};
        const ${dtype_specifier} * src1 = ${source1};
        const ${dtype_specifier} * src2 = ${source2};
        ${dtype_specifier} * dst = ${dst};
        for(size_t b=0; b<batch; b++){
            for(size_t c=0; c<channel; c++){
                ${simd_dtype_specifier} vsrc1_0 = ${broad_cast}(src1[c]);
                size_t index = 0;
                for(; index + 2*SIMD_WIDTH-1 < nr_elem_per_channel; index += 2*SIMD_WIDTH) {
                    ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
                    ${simd_dtype_specifier} vsrc0_1 = ${load_vec}(src0 + SIMD_WIDTH);
                    ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
                    ${simd_dtype_specifier} vsrc2_1 = ${load_vec}(src2 + SIMD_WIDTH);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc2_0, vsrc0_1, vsrc1_0, vsrc2_1)}
                    src0 += 2*SIMD_WIDTH;
                    src2 += 2*SIMD_WIDTH;
                    dst += 2*SIMD_WIDTH;
                }
                for(; index + SIMD_WIDTH-1 < nr_elem_per_channel; index += SIMD_WIDTH) {
                    ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
                    ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0, vsrc2_0)}
                    src0 += SIMD_WIDTH;
                    src2 += SIMD_WIDTH;
                    dst += SIMD_WIDTH;
                }
                for(; index < nr_elem_per_channel; index++) {
                    ${kernel_naive_unroll(1, dst, src0, src1+c, src2)}
                    src0 += 1;
                    src2 += 1;
                    dst += 1;
                }
            }
        }
        )";
    return body;
}

template <>
std::string TernaryCode<VEC_BCAST101xX_VEC>() {
    std::string body = R"(
        size_t SIMD_WIDTH = ${simd_width};
        Layout dst_layout = outputs[0]->layout;
        Layout src_layout_1 = inputs[1]->layout;
        
        size_t bvbv[4]={1, 1, 1, 1};
        )" + bvbv_calc("bvbv", "src_layout_1", "dst_layout") +
                       R"(
        size_t batch = bvbv[0];
        size_t channel = bvbv[1];
        size_t nr_elem_per_channel = bvbv[2] * bvbv[3];
        TINYNN_ASSERT(bvbv[3] == SIMD_WIDTH);

        ${kernel_init()}
        const ${dtype_specifier} * src0 = ${source0};
        const ${dtype_specifier} * src1 = ${source1};
        const ${dtype_specifier} * src2 = ${source2};
        ${dtype_specifier} * dst = ${dst};
        for(size_t b=0; b<batch; b++){
            for(size_t c=0; c<channel; c++){
                ${simd_dtype_specifier} vsrc1_0 = ${load_vec}(src1 + c * SIMD_WIDTH);
                size_t index = 0;
                for(; index + 2*SIMD_WIDTH-1 < nr_elem_per_channel; index += 2*SIMD_WIDTH) {
                    ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
                    ${simd_dtype_specifier} vsrc0_1 = ${load_vec}(src0 + SIMD_WIDTH);
                    ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
                    ${simd_dtype_specifier} vsrc2_1 = ${load_vec}(src2 + SIMD_WIDTH);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc2_0, vsrc0_1, vsrc1_0, vsrc2_1)}
                    src0 += 2*SIMD_WIDTH;
                    src2 += 2*SIMD_WIDTH;
                    dst += 2*SIMD_WIDTH;
                }
                for(; index + SIMD_WIDTH-1 < nr_elem_per_channel; index += SIMD_WIDTH) {
                    ${simd_dtype_specifier} vsrc0_0 = ${load_vec}(src0);
                    ${simd_dtype_specifier} vsrc2_0 = ${load_vec}(src2);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0, vsrc2_0)}
                    src0 += SIMD_WIDTH;
                    src2 += SIMD_WIDTH;
                    dst += SIMD_WIDTH;
                }
            }
        })";
    return body;
}
std::string get_bcast_template(BcastType bcast_type) {
    switch (bcast_type) {
        case VEC_VEC_VEC:
            return TernaryCode<VEC_VEC_VEC>();
        case VEC_BCAST101_VEC:
            return TernaryCode<VEC_BCAST101_VEC>();
        case VEC_BCAST101xX_VEC:
            return TernaryCode<VEC_BCAST101xX_VEC>();
        default:
            return "";
    }
    return "";
}
}  // namespace

std::string ElemwiseGenTernaryFuseMulAdd3::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenTernaryFuseMulAdd3::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (m_comp_type == Utils::DtypeEnum::float32) {
            writer << "\n GI_FLOAT32_t tmp" << i << " = GiMlaqFloat32("
                   << strs[str_id + 2] << "," << strs[str_id + 0] << ","
                   << strs[str_id + 1] << ");";
            writer << "\n GiStoreFloat32((" << dst << " + 4 * " << i << "), tmp" << i
                   << ");";
        } else if (m_comp_type == Utils::DtypeEnum::float16) {
            writer << "\n GI_FLOAT16_t tmp" << i << " = GiMlaqFloat16("
                   << strs[str_id + 2] << "," << strs[str_id + 0] << ","
                   << strs[str_id + 1] << ");";
            writer << "\n GiStoreFloat16((" << dst << " + 8 * " << i << "), tmp" << i
                   << ");";
        }

        str_id += 3;
    }
    return writer.str();
}

std::string ElemwiseGenTernaryFuseMulAdd3::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (m_comp_type == Utils::DtypeEnum::float32) {
            writer << "\n"
                   << "float tmp = (" << strs[str_id + 0] << ")[" << i << "] * ("
                   << strs[str_id + 1] << ")[" << i << "] + (" << strs[str_id + 2]
                   << ")[" << i << "];";
            writer << "\n"
                   << dst << "[" << i << "] = "
                   << " tmp;";
        } else if (m_comp_type == Utils::DtypeEnum::float16) {
            writer << "\n"
                   << "gi_float16_t tmp = (" << strs[str_id + 0] << ")[" << i << "] * ("
                   << strs[str_id + 1] << ")[" << i << "] + (" << strs[str_id + 2]
                   << ")[" << i << "];";
            writer << "\n"
                   << dst << "[" << i << "] = "
                   << " tmp;";
        }

        str_id += 3;
    }
    return writer.str();
}

std::string ElemwiseGenTernary::GenCodeBody(std::vector<std::string> strs) const {
    CC_ASSERT(strs.size() == 4);
    auto input0 = strs[0];
    auto input1 = strs[1];
    auto input2 = strs[2];
    auto output = strs[3];
    std::string body = get_bcast_template(m_bcast_type);
    CC_ASSERT(!body.empty()) << "unsupported broadcast type in elemwise\n";

    auto kernel_init = [this](std::vector<std::string> strs) {
        return GenKernelSimdInit(strs);
    };
    auto kernel_simd_unroll = [this](std::vector<std::string> strs) {
        return GenKernelSimdUnroll(strs);
    };
    auto kernel_naive_unroll = [this](std::vector<std::string> strs) {
        return GenKernelNaiveUnroll(strs);
    };
    std::stringstream ss;

    auto dtype = Utils::cvt_dtype_specifier(m_comp_type);
    auto simd_width = Utils::get_dtype_simd_length(m_comp_type);
    auto simd_dtype = Utils::get_dtype_gi_simd_type(m_comp_type);
    std::string gi_type_str = Utils::get_dtype_gi_type_str(m_comp_type);
    auto simd_load = "GiLoad" + gi_type_str;
    auto simd_broad = "GiBroadcast" + gi_type_str;
    ss << StringTemplate::StringTemplateArgs()
                    .add("source0", input0)
                    .add("source1", input1)
                    .add("source2", input2)
                    .add("dst", output)
                    .add("kernel_init", kernel_init)
                    .add("kernel_simd_unroll", kernel_simd_unroll)
                    .add("kernel_naive_unroll", kernel_naive_unroll)
                    .add("dtype_specifier", dtype)
                    .add("simd_dtype_specifier", simd_dtype)
                    .add("simd_width", (uint32_t)simd_width)
                    .add("load_vec", simd_load)
                    .add("broad_cast", simd_broad)
                    .render(body);

    return ss.str();
}

BcastType ElemwiseGenTernary::GetBcastType(
        const CCOperand& operand0, const CCOperand& operand1,
        const CCOperand& operand2) {
    return GetTernaryBcastType(operand0, operand1, operand2);
}

bool ElemwiseGenTernary::is_available(BcastType bcast_type) {
    return !get_bcast_template(bcast_type).empty();
}

// vim: syntax=cpp.doxygen
