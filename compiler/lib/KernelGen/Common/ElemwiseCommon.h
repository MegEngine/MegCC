/**
 * \file
 * compiler/lib/KernelGen/Common/ElemwiseCommon.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {

enum TensorType {
    VECTOR = 0,
    SCALAR = 1,
    BCAST101 = 2,
    BCAST101x4 = 3,
    UNKNOWN_TENSOR_TYPE = 4,
    NR_TENSOR_TYPE = 5
};
#define CAL_BCAST3(a, b, c) (a * 100 + b * 10 + c)
/*!
 * \brief broadcast type
 * BCAST_x[0]x[1]...: x[i] == !stride[i]
 */
enum BcastType {
    VEC = 0,
    VEC_VEC,
    VEC_BCAST101,
    VEC_BCAST101x4,
    VEC_SCALAR,
    SCALAR_VEC,
    BCAST101_VEC,
    BCAST101x4_VEC,
    BV_VEC,  //!  B=broadcast, V=elemwise(no broadcast)
    VEC_BV,
    BCAST_VEC,  //! VEC_BCAST is same enum
    NAIVE,
    BINARY_END = 100,
    //! Ternary type is 9 + TensorType0:TensorType1:TensorType2
    VEC_VEC_VEC = BINARY_END + CAL_BCAST3(VECTOR, VECTOR, VECTOR),
    VEC_VEC_SCALAR = BINARY_END + CAL_BCAST3(VECTOR, VECTOR, SCALAR),
    BCAST101_VEC_BCAST101 = BINARY_END + CAL_BCAST3(BCAST101, VECTOR, BCAST101),
    BCAST101x4_VEC_BCAST101x4 =
            BINARY_END + CAL_BCAST3(BCAST101x4, VECTOR, BCAST101x4),
    VEC_BCAST101_VEC = BINARY_END + CAL_BCAST3(VECTOR, BCAST101, VECTOR),
    VEC_BCAST101x4_VEC = BINARY_END + CAL_BCAST3(VECTOR, BCAST101x4, VECTOR),
    VEC_SCALAR_VEC = BINARY_END + CAL_BCAST3(VECTOR, SCALAR, VECTOR),
    VEC_SCALAR_SCALAR = BINARY_END + CAL_BCAST3(VECTOR, SCALAR, SCALAR),
    UNKNOWN_BCAST_TYPE = BINARY_END + 1000,
    DYNAMIC_TYPE = BINARY_END + 1001
};
#undef CAL_BCAST3
//! get the TensorType type of operand when the dst provided
TensorType GetOperandTensorType(const CCOperand& dst, const CCOperand& operand);

BcastType GetBinaryBcastType(const CCOperand& operand0,
                             const CCOperand& operand1);

BcastType GetTernaryBcastType(const CCOperand& operand0,
                              const CCOperand& operand1,
                              const CCOperand& operand2);

std::vector<TensorType> DecodeTernaryBcastType(const BcastType bc_type);

std::vector<TensorType> GetQuaterBcastType(const CCOperand& operand0,
                                           const CCOperand& operand1,
                                           const CCOperand& operand2,
                                           const CCOperand& operand3);
std::vector<size_t> CalBroadcastElemwise(const std::vector<size_t>& src,
                                         const std::vector<size_t>& dst);

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
