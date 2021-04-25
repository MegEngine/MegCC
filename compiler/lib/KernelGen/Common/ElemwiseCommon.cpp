/**
 * \file
 * compiler/lib/KernelGen/Common/ElemwiseCommon.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "./ElemwiseCommon.h"
#include "Utils/Utils.h"

namespace megcc {
namespace KernelGen {
BcastType GetBinaryBcastType(const CCOperand& operand0,
                             const CCOperand& operand1) {
    auto shape0 = operand0.shape;
    auto shape1 = operand1.shape;
    if (Utils::is_shape_dynamic(shape0) || Utils::is_shape_dynamic(shape1)) {
        return BcastType::DYNAMIC_TYPE;
    }
    size_t nr_elem0 = 1;
    size_t nr_elem1 = 1;
    for (size_t i = 0; i < shape0.size(); i++) {
        nr_elem0 *= shape0[i];
    }
    for (size_t i = 0; i < shape1.size(); i++) {
        nr_elem1 *= shape1[i];
    }
    if (nr_elem0 == nr_elem1) {
        return BcastType::VEC_VEC;
    } else if (nr_elem0 == 1 || nr_elem1 == 1) {
        if (nr_elem0 == 1) {
            return BcastType::SCALAR_VEC;
        } else {
            return BcastType::VEC_SCALAR;
        }
    } else {
        CC_ASSERT(shape0.size() == shape1.size())
                << shape0.size() << "==" << shape1.size() << "\n";
        auto small_shape = nr_elem0 < nr_elem1 ? shape0 : shape1;
        auto big_shape = nr_elem0 < nr_elem1 ? shape1 : shape0;
        bool bcast_vec = true;
        for (size_t i = 0; i < big_shape.size(); ++i) {
            if (big_shape[i] == 1) {
                bcast_vec &= small_shape[i] == 1;
            }
        }
        if (bcast_vec) {
            if (small_shape.size() == 4 && small_shape[0] == 1 &&
                small_shape[2] == 1 && small_shape[3] == 1) {
                if (nr_elem0 < nr_elem1) {
                    return BCAST101_VEC;
                } else {
                    return VEC_BCAST101;
                }

            } else if (small_shape.size() == 5 && small_shape[0] == 1 &&
                       small_shape[2] == 1 && small_shape[3] == 1 &&
                       small_shape[4] == 4) {
                if (nr_elem0 < nr_elem1) {
                    return BCAST101x4_VEC;
                } else {
                    return VEC_BCAST101x4;
                }
            }
            auto bv_array = CalBroadcastElemwise(small_shape, big_shape);
            if (bv_array.size() == 2) {
                if (nr_elem0 < nr_elem1) {
                    return BV_VEC;
                } else {
                    return VEC_BV;
                }
            }
        }
        return NAIVE;
    }
}

BcastType GetTernaryBcastType(const CCOperand& operand0,
                              const CCOperand& operand1,
                              const CCOperand& operand2) {
    auto shape0 = operand0.shape;
    auto shape1 = operand1.shape;
    auto shape2 = operand2.shape;
    if (Utils::is_shape_dynamic(shape0) || Utils::is_shape_dynamic(shape1) ||
        Utils::is_shape_dynamic(shape2)) {
        return BcastType::DYNAMIC_TYPE;
    }
    size_t nr_elem0 = 1;
    size_t nr_elem1 = 1;
    size_t nr_elem2 = 1;
    size_t max_shape_size =
            std::max(std::max(shape0.size(), shape1.size()), shape2.size());
    std::vector<size_t> dst_shape;
    for (size_t i = 0; i < max_shape_size; ++i) {
        dst_shape.push_back(1);
    }
    for (size_t i = 0; i < shape0.size(); i++) {
        nr_elem0 *= shape0[i];
        dst_shape[i] = std::max(dst_shape[i], shape0[i]);
    }
    for (size_t i = 0; i < shape1.size(); i++) {
        nr_elem1 *= shape1[i];
        dst_shape[i] = std::max(dst_shape[i], shape1[i]);
    }
    for (size_t i = 0; i < shape2.size(); i++) {
        nr_elem2 *= shape2[i];
        dst_shape[i] = std::max(dst_shape[i], shape2[i]);
    }
    size_t max_elemwise = std::max(std::max(nr_elem0, nr_elem1), nr_elem2);
    auto get_tensor_type = [&](size_t nr_elem, std::vector<size_t>& shape,
                               std::vector<size_t>& dst_shape) {
        if (nr_elem == max_elemwise) {
            return VECTOR;
        } else if (nr_elem == 1) {
            return SCALAR;
        } else {
            auto bvb_vec = CalBroadcastElemwise(shape, dst_shape);
            if (bvb_vec.size() == 4 && bvb_vec[3] == 4) {
                return BCAST101x4;
            } else if (bvb_vec.size() == 3) {
                return BCAST101;
            } else {
                return UNKNOWN_TENSOR_TYPE;
            }
        }
    };
    TensorType input0 = get_tensor_type(nr_elem0, shape0, dst_shape);
    TensorType input1 = get_tensor_type(nr_elem1, shape1, dst_shape);
    TensorType input2 = get_tensor_type(nr_elem2, shape2, dst_shape);
    return static_cast<BcastType>(input0 * 100 + input1 * 10 + input2 +
                                  VEC_VEC_VEC);
}

std::vector<TensorType> GetQuaterBcastType(const CCOperand& operand0,
                                           const CCOperand& operand1,
                                           const CCOperand& operand2,
                                           const CCOperand& operand3) {
    auto shape0 = operand0.shape;
    auto shape1 = operand1.shape;
    auto shape2 = operand2.shape;
    auto shape3 = operand3.shape;

    auto get_nr_elem = [](const std::vector<size_t>& shape) {
        size_t nr_elem = 1;
        for (size_t i = 0; i < shape.size(); i++) {
            nr_elem *= shape[i];
        }
        return nr_elem;
    };
    size_t nr_elem0 = get_nr_elem(shape0);
    size_t nr_elem1 = get_nr_elem(shape1);
    size_t nr_elem2 = get_nr_elem(shape2);
    size_t nr_elem3 = get_nr_elem(shape3);
    size_t max_elemwise = std::max(std::max(nr_elem0, nr_elem1),
                                   std::max(nr_elem2, nr_elem3));
    auto get_tensor_type = [&](size_t nr_elem, std::vector<size_t>& shape) {
        if (nr_elem == 1) {
            return SCALAR;
        } else if (nr_elem == max_elemwise) {
            return VECTOR;
        } else {
            if (shape[shape.size() - 1] != 4) {
                return BCAST101;
            } else {
                return BCAST101x4;
            }
        }
    };
    std::vector<TensorType> ret;
    ret.push_back(get_tensor_type(nr_elem0, shape0));
    ret.push_back(get_tensor_type(nr_elem1, shape1));
    ret.push_back(get_tensor_type(nr_elem2, shape2));
    ret.push_back(get_tensor_type(nr_elem3, shape3));
    return ret;
}

std::vector<TensorType> DecodeTernaryBcastType(const BcastType bct_type) {
    std::vector<TensorType> input_type;
    input_type.push_back(
            static_cast<TensorType>((bct_type - VEC_VEC_VEC) / 100));
    input_type.push_back(
            static_cast<TensorType>(((bct_type - VEC_VEC_VEC) % 100) / 10));
    input_type.push_back(
            static_cast<TensorType>((bct_type - VEC_VEC_VEC) % 10));
    return input_type;
}

std::vector<size_t> CalBroadcastElemwise(const std::vector<size_t>& src,
                                         const std::vector<size_t>& dst) {
    std::vector<size_t> res;
    res.push_back(1);
    bool is_broadcast = true;
    for (size_t i = 0; i < dst.size(); ++i) {
        if (is_broadcast == false) {
            //! elemwise
            if (src[i] != dst[i]) {
                is_broadcast = true;
                res.push_back(1);
            }
        } else {
            //! broadcast
            if (src[i] != 1) {
                is_broadcast = false;
                res.push_back(1);
            }
        }
        res.back() *= dst[i];
    }
    return res;
}

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
