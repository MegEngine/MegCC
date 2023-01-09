/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/ElemwiseHelper/BinaryHelper.cpp
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
std::string BinaryCode();

template <>
std::string BinaryCode<VEC_VEC>() {
    std::string body = R"(
        Layout layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < layout.nr_dim; ++i) {
                nr_elem *= layout.dims[i];
        }
        ${kernel_init()}
        const float * src0 = ${source0};
        const float * src1 = ${source1};
        float * dst = ${dst};
        size_t index = 0;
        for(; index + 7 < nr_elem; index += 8) {
            GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
            GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
            GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
            GI_FLOAT32_t vsrc1_1 = GiLoadFloat32(src1 + 4);
            ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_1)}
            src0 += 8;
            src1 += 8;
            dst += 8;
        }
        for(; index + 3 < nr_elem; index += 4) {
            GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
            GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
            ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
            src0 += 4;
            src1 += 4;
            dst += 4;
        }
        for(; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, dst, src0, src1)}
            src0 += 1;
            src1 += 1;
            dst += 1;
        })";
    return body;
}

template <>
std::string BinaryCode<VEC_SCALAR>() {
    std::string body = R"(
        Layout layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < layout.nr_dim; ++i) {
                nr_elem *= layout.dims[i];
        }
        ${kernel_init()}
        const float * src0 = ${source0};
        const float * src1 = ${source1};
        float * dst = ${dst};
        GI_FLOAT32_t vsrc1_0 = GiBroadcastFloat32(src1[0]);
        size_t index = 0;
        for(; index + 7 < nr_elem; index += 8) {
            GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
            GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
            ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_0)}
            src0 += 8;
            dst += 8;
        }
        for(; index + 3 < nr_elem; index += 4) {
            GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
            ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
            src0 += 4;
            dst += 4;
        }
        for(; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, dst, src0, src1)}
            src0 += 1;
            dst += 1;
        })";
    return body;
}

template <>
std::string BinaryCode<VEC_BCAST101>() {
    std::string body = R"(
        Layout dst_layout = outputs[0]->layout;
        size_t batch = dst_layout.dims[0];
        size_t channel = dst_layout.dims[1];
        size_t nr_elem_per_channel = dst_layout.dims[2] * dst_layout.dims[3];
        ${kernel_init()}
        const float * src0 = ${source0};
        const float * src1 = ${source1};
        float * dst = ${dst};
        for(size_t b=0; b<batch; b++){
            for(size_t c=0; c<channel; c++){
                GI_FLOAT32_t vsrc1_0 = GiBroadcastFloat32(src1[c]);
                size_t index = 0;
                for(; index + 7 < nr_elem_per_channel; index += 8) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_0)}
                    src0 += 8;
                    dst += 8;
                }
                for(; index + 3 < nr_elem_per_channel; index += 4) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                    src0 += 4;
                    dst += 4;
                }
                for(; index < nr_elem_per_channel; index++) {
                    ${kernel_naive_unroll(1, dst, src0, src1+c)}
                    src0 += 1;
                    dst += 1;
                }
            }
        }
        )";
    return body;
}

template <>
std::string BinaryCode<VEC_BCAST101x4>() {
    std::string body = R"(
        Layout dst_layout = outputs[0]->layout;
        size_t batch = dst_layout.dims[0];
        size_t channel = dst_layout.dims[1];
        size_t nr_elem_per_channel = dst_layout.dims[2] * dst_layout.dims[3] * dst_layout.dims[4];
        ${kernel_init()}
        const float * src0 = ${source0};
        const float * src1 = ${source1};
        float * dst = ${dst};
        for(size_t b=0; b<batch; b++){
            for(size_t c=0; c<channel; c++){
                GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1 + c * 4);
                size_t index = 0;
                for(; index + 7 < nr_elem_per_channel; index += 8) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_0)}
                    src0 += 8;
                    dst += 8;
                }
                for(; index + 3 < nr_elem_per_channel; index += 4) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                    src0 += 4;
                    dst += 4;
                }
            }
        })";
    return body;
}

template <>
std::string BinaryCode<VEC_BV>() {
    std::string body = R"(
        Layout dst_layout = outputs[0]->layout;
        Layout src_layout_0 = inputs[0]->layout;
        Layout src_layout_1 = inputs[1]->layout;
        Layout small_layout = ${reverse} == 1? src_layout_0:src_layout_1;
        //! b for broadcast, e for elemwise
        size_t batch = 1;
        size_t nr_elem_per_channel = 1;
        int be_idx = 0;
        for (int i = 0; i < dst_layout.nr_dim; ++i){
            if(small_layout.dims[i] == 1){
                batch *= dst_layout.dims[i];
            }else{
                nr_elem_per_channel *= dst_layout.dims[i];
            }
        }
        ${kernel_init()}
        const float * src0_base = ${source0};
        const float * src1_base = ${source1};
        float * dst = ${dst};
        for(size_t b=0; b<batch; b++){
            const float * src0 = src0_base + b * nr_elem_per_channel;
            const float * src1 = src1_base;
            size_t index = 0;
            for(; index + 7 < nr_elem_per_channel; index += 8) {
                GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
                GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                GI_FLOAT32_t vsrc1_1 = GiLoadFloat32(src1 + 4);
                ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_1)}
                src0 += 8;
                src1 += 8;
                dst += 8;
            }
            for(; index + 3 < nr_elem_per_channel; index += 4) {
                GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                src0 += 4;
                src1 += 4;
                dst += 4;
            }
            for(; index < nr_elem_per_channel; index++) {
                ${kernel_naive_unroll(1, dst, src0, src1)}
                src0 += 1;
                src1 += 1;
                dst += 1;
            }
        }
        )";
    return body;
}

template <>
std::string BinaryCode<NAIVE>() {
    std::string body = R"(
        Layout dst_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < dst_layout.nr_dim; ++i) {
            nr_elem *= dst_layout.dims[i];
        }
        Layout src_layout_0 = inputs[0]->layout;
        Layout src_layout_1 = inputs[1]->layout;
        ${kernel_init()}
        
        broadcast_layout(&src_layout_0, dst_layout);
        broadcast_layout(&src_layout_1, dst_layout);
        NoconIter src0_iter = init_iter(src_layout_0);
        NoconIter src1_iter = init_iter(src_layout_1);
        const float * src0 = inputs[0]->ptr;
        const float * src1 = inputs[1]->ptr;
        float * dst = outputs[0]->ptr;
        
        for(size_t index = 0; index < nr_elem; index++) {
            ${kernel_naive_unroll(1, dst, src0 + src0_iter.offset, src1 + src1_iter.offset)}
            inc_iter(src_layout_0, &src0_iter);
            inc_iter(src_layout_1, &src1_iter);
            dst++;
        }
        
    )";
    return body;
}

template <>
std::string BinaryCode<DYNAMIC_TYPE>() {
    std::string body = R"(
        Layout dst_layout = outputs[0]->layout;
        size_t nr_elem = 1;
        for (int i = 0; i < dst_layout.nr_dim; ++i) {
            nr_elem *= dst_layout.dims[i];
        }
        Layout src_layout_0 = inputs[0]->layout;
        Layout src_layout_1 = inputs[1]->layout;
        size_t nr_elem_in0 = 1;
        for (int i = 0; i < src_layout_0.nr_dim; ++i) {
            nr_elem_in0 *= src_layout_0.dims[i];
        }
        size_t nr_elem_in1 = 1;
        for (int i = 0; i < src_layout_1.nr_dim; ++i) {
            nr_elem_in1 *= src_layout_1.dims[i];
        }
        ${kernel_init()}
        if (nr_elem == nr_elem_in0 && nr_elem_in0 == nr_elem_in1){
            const float * src0 = inputs[0]->ptr;
            const float * src1 = inputs[1]->ptr;
            float * dst = outputs[0]->ptr;
            size_t index = 0;
            for(; index + 7 < nr_elem; index += 8) {
                GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
                GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                GI_FLOAT32_t vsrc1_1 = GiLoadFloat32(src1 + 4);
                ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_1)}
                src0 += 8;
                src1 += 8;
                dst += 8;
            }
            for(; index + 3 < nr_elem; index += 4) {
                GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                src0 += 4;
                src1 += 4;
                dst += 4;
            }
            for(; index < nr_elem; index++) {
                ${kernel_naive_unroll(1, dst, src0, src1)}
                src0 += 1;
                src1 += 1;
                dst += 1;
            }
        }else if(nr_elem_in0 == 1 || nr_elem_in1 == 1){
            if(nr_elem_in0 > nr_elem_in1){
                const float * src0 = inputs[0]->ptr;
                const float * src1 = inputs[1]->ptr;
                float * dst = outputs[0]->ptr;
                GI_FLOAT32_t vsrc1_0 = GiBroadcastFloat32(src1[0]);
                size_t index = 0;
                for(; index + 7 < nr_elem; index += 8) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    GI_FLOAT32_t vsrc0_1 = GiLoadFloat32(src0 + 4);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_1, vsrc1_0)}
                    src0 += 8;
                    dst += 8;
                }
                for(; index + 3 < nr_elem; index += 4) {
                    GI_FLOAT32_t vsrc0_0 = GiLoadFloat32(src0);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                    src0 += 4;
                    dst += 4;
                }
                for(; index < nr_elem; index++) {
                    ${kernel_naive_unroll(1, dst, src0, src1)}
                    src0 += 1;
                    dst += 1;
                }
            }else{
                const float * src0 = inputs[0]->ptr;
                const float * src1 = inputs[1]->ptr;
                float * dst = outputs[0]->ptr;
                GI_FLOAT32_t vsrc0_0 = GiBroadcastFloat32(src0[0]);
                size_t index = 0;
                for(; index + 7 < nr_elem; index += 8) {
                    GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                    GI_FLOAT32_t vsrc1_1 = GiLoadFloat32(src1 + 4);
                    ${kernel_simd_unroll(2, dst, vsrc0_0, vsrc1_0, vsrc0_0, vsrc1_1)}
                    src1 += 8;
                    dst += 8;
                }
                for(; index + 3 < nr_elem; index += 4) {
                    GI_FLOAT32_t vsrc1_0 = GiLoadFloat32(src1);
                    ${kernel_simd_unroll(1, dst, vsrc0_0, vsrc1_0)}
                    src1 += 4;
                    dst += 4;
                }
                for(; index < nr_elem; index++) {
                    ${kernel_naive_unroll(1, dst, src0, src1)}
                    src1 += 1;
                    dst += 1;
                }
            }
        }else{
            broadcast_layout(&src_layout_0, dst_layout);
            broadcast_layout(&src_layout_1, dst_layout);
            NoconIter src0_iter = init_iter(src_layout_0);
            NoconIter src1_iter = init_iter(src_layout_1);
            const float * src0 = inputs[0]->ptr;
            const float * src1 = inputs[1]->ptr;
            float * dst = outputs[0]->ptr;
            
            for(size_t index = 0; index < nr_elem; index++) {
                ${kernel_naive_unroll(1, dst, src0 + src0_iter.offset, src1 + src1_iter.offset)}
                inc_iter(src_layout_0, &src0_iter);
                inc_iter(src_layout_1, &src1_iter);
                dst++;
            }
        }
        
    )";
    return body;
}
}  // namespace

std::string ElemwiseGenBinaryAdd::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinaryAdd::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
               << ", GiAddFloat32(" << strs[str_id] << "," << strs[str_id + 1]
               << "));";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryAdd::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id] << ")["
               << i << "] + (" << strs[str_id + 1] << ")[" << i << "];";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinarySub::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinarySub::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (!m_should_reverse) {
            writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
                   << ", GiSubtractFloat32(" << strs[str_id] << ","
                   << strs[str_id + 1] << "));";
        } else {
            writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
                   << ", GiSubtractFloat32(" << strs[str_id + 1] << ","
                   << strs[str_id] << "));";
        }
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinarySub::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (!m_should_reverse) {
            writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id]
                   << ")[" << i << "] - (" << strs[str_id + 1] << ")[" << i
                   << "];";
        } else {
            writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id + 1]
                   << ")[" << i << "] - (" << strs[str_id] << ")[" << i << "];";
        }
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMul::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinaryMul::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
               << ", GiMultiplyFloat32(" << strs[str_id] << ","
               << strs[str_id + 1] << "));";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMul::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id] << ")["
               << i << "] * (" << strs[str_id + 1] << ")[" << i << "];";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryTrueDiv::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinaryTrueDiv::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (!m_should_reverse) {
            writer << "\n GiStoreFloat32((" << dst << "+4*" << i
                   << "), GiDivideFloat32(" << strs[str_id] << " , "
                   << strs[str_id + 1] << "));";
        } else {
            writer << "\n GiStoreFloat32((" << dst << "+4*" << i
                   << "), GiDivideFloat32(" << strs[str_id + 1] << " , "
                   << strs[str_id] << "));";
        }
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryTrueDiv::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        if (!m_should_reverse) {
            writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id]
                   << ")[" << i << "] / (" << strs[str_id + 1] << ")[" << i
                   << "];";
        } else {
            writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id + 1]
                   << ")[" << i << "] / (" << strs[str_id] << ")[" << i << "];";
        }
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryFuseAddRelu::GenKernelSimdInit(
        std::vector<std::string>) const {
    std::stringstream writer;
    writer << "\nGI_FLOAT32_t vzero = GiBroadcastFloat32(0.f);";
    return writer.str();
}

std::string ElemwiseGenBinaryFuseAddRelu::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n GI_FLOAT32_t tmp" << i << " = GiAddFloat32("
               << strs[str_id] << "," << strs[str_id + 1] << ");";
        writer << "\n GiStoreFloat32((" << dst << " +4*" << i
               << "), GiMaximumFloat32(tmp" << i << ", vzero));";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryFuseAddRelu::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n"
               << "float tmp = (" << strs[str_id] << ")[" << i << "] + ("
               << strs[str_id + 1] << ")[" << i << "];";
        writer << "\n"
               << dst << "[" << i << "] = "
               << " tmp > 0 ? tmp : 0.0f;";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMax::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinaryMax::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
               << ", GiMaximumFloat32(" << strs[str_id] << "," << strs[str_id + 1]
               << "));";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMax::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id] << ")["
               << i << "] > (" << strs[str_id + 1] << ")[" << i << "] ?(" << strs[str_id] << ")[" << i <<"]:(" << strs[str_id + 1] << ")["
               << i << "] ;";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMin::GenKernelSimdInit(
        std::vector<std::string>) const {
    return "";
}

std::string ElemwiseGenBinaryMin::GenKernelSimdUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
    for (int i = 0; i < unroll; i++) {
        writer << "\n GiStoreFloat32((" << dst << ") + 4 * " << i
               << ", GiMinimumFloat32(" << strs[str_id] << "," << strs[str_id + 1]
               << "));";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinaryMin::GenKernelNaiveUnroll(
        std::vector<std::string> strs) const {
    int unroll = std::stoi(strs[0]);
    auto dst = strs[1];
    std::stringstream writer;
    int str_id = 2;
   for (int i = 0; i < unroll; i++) {
        writer << "\n(" << dst << ")[" << i << "] = (" << strs[str_id] << ")["
               << i << "] < (" << strs[str_id + 1] << ")[" << i << "] ?(" << strs[str_id] << ")[" << i <<"]:(" << strs[str_id + 1] << ")["
               << i << "] ;";
        str_id += 2;
    }
    return writer.str();
}

std::string ElemwiseGenBinary::GenCodeBody(
        std::vector<std::string> strs) const {
    auto input0 = strs[0];
    auto input1 = strs[1];
    auto output = strs[2];
    std::string body;
    switch (m_bcast_type) {
        case VEC_VEC:
            body = BinaryCode<VEC_VEC>();
            break;
        case VEC_BCAST101:
        case BCAST101_VEC:
            body = BinaryCode<VEC_BCAST101>();
            break;
        case VEC_BCAST101x4:
        case BCAST101x4_VEC:
            body = BinaryCode<VEC_BCAST101x4>();
            break;
        case VEC_SCALAR:
        case SCALAR_VEC:
            body = BinaryCode<VEC_SCALAR>();
            break;
        case VEC_BV:
        case BV_VEC:
            body = BinaryCode<VEC_BV>();
            break;
        case NAIVE:
            body = BinaryCode<NAIVE>();
            break;
        case DYNAMIC_TYPE:
            body = BinaryCode<DYNAMIC_TYPE>();
            break;
        default:
            CC_ABORT << "unsupport broadcast type in elemwise\n";
    }

    auto kernel_init = [this](std::vector<std::string> strs) {
        return GenKernelSimdInit(strs);
    };
    auto kernel_simd_unroll = [this](std::vector<std::string> strs) {
        return GenKernelSimdUnroll(strs);
    };
    auto kernel_naive_unroll = [this](std::vector<std::string> strs) {
        return GenKernelNaiveUnroll(strs);
    };
    if (m_should_reverse) {
        input0 = strs[1];
        input1 = strs[0];
    }
    int reverse_flag = m_should_reverse ? 1 : 0;
    std::stringstream ss;
    ss << StringTemplate::StringTemplateArgs()
                    .add("source0", input0)
                    .add("source1", input1)
                    .add("reverse", reverse_flag)
                    .add("dst", output)
                    .add("kernel_init", kernel_init)
                    .add("kernel_simd_unroll", kernel_simd_unroll)
                    .add("kernel_naive_unroll", kernel_naive_unroll)
                    .render(body);

    return ss.str();
}

BcastType ElemwiseGenBinary::GetBcastType(const CCOperand& operand0,
                                          const CCOperand& operand1) {
    return GetBinaryBcastType(operand0, operand1);
}

bool ElemwiseGenBinary::WhetherShouldReverse(const CCOperand& operand0,
                                             const CCOperand& operand1) {
    auto shape0 = operand0.shape;
    auto shape1 = operand1.shape;
    size_t nr_elem0 = 1;
    size_t nr_elem1 = 1;
    for (size_t i = 0; i < shape0.size(); i++) {
        nr_elem0 *= shape0[i];
    }
    for (size_t i = 0; i < shape1.size(); i++) {
        nr_elem1 *= shape1[i];
    }
    if (Utils::is_shape_dynamic(shape0) || Utils::is_shape_dynamic(shape1)) {
        return false;
    }
    if (nr_elem0 < nr_elem1) {
        return true;
    } else {
        return false;
    }
}

// vim: syntax=cpp.doxygen
