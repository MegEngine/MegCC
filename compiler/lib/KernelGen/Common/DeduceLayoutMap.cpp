/**
 * \file
 * compiler/lib/KernelGen/Common/DeduceLayoutMap.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "./DeduceLayoutMap.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"
#include "compiler/Common/TContext.h"
using namespace megcc;
using namespace KernelGen;
namespace {
class IndexingDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        std::stringstream ss;
        ss << "DeduceFunc_indexingmultiaxisvex";
        int nr_operand = context->getAttrInt("nr_operands");
        for (int i = 0; i < nr_operand - 2; ++i) {
            ss << "_" << context->getAttrInt("axis:" + std::to_string(i));
        }
        return ss.str();
    }

    std::string GetDeduceBody(TContext* context) const override {
        std::stringstream writer;
        writer << R"(
            #include "tensor_util.h"
        )";
        std::stringstream axis_vec;
        axis_vec << "{";
        int nr_operand = context->getAttrInt("nr_operands");
        for (int i = 0; i < nr_operand - 2; ++i) {
            axis_vec << context->getAttrInt("axis:" + std::to_string(i));
            if (i < nr_operand - 3) {
                axis_vec << ",";
            }
        }
        axis_vec << "}";
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";

        std::string body = R"(
                uint32_t axis_vec[] = ${axis_vec};
                Tensor* output = outputs[0];
                output->layout = inputs[0]->layout;
                output->dtype = inputs[0]->dtype;
                int new_shape = inputs[1]->layout.dims[0];
                output->layout.dims[axis_vec[0]] = new_shape;
                //! shrink dims
                for (int in_idx = 1; in_idx < nr_input - 1; ++in_idx) {
                    TINYNN_ASSERT_MSG(inputs[in_idx + 1]->layout.dims[0] == new_shape,
                                "unsupport input\n");
                    int axis = axis_vec[in_idx];
                    int dim_offset = in_idx - 1;
                    for (int dim = axis - dim_offset; dim < output->layout.nr_dim - 1; ++dim) {
                        output->layout.dims[dim] = output->layout.dims[dim + 1];
                    }
                    output->layout.nr_dim--;
                }
                force_layout_contiguous(&(output->layout));
                return TinyNN_SUCCESS;
            }
        )";
        writer << StringTemplate::StringTemplateArgs()
                          .add("axis_vec", axis_vec.str())
                          .render(body);
        return writer.str();
    }
};
class ArgSortDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        return "DeduceFunc_ArgSort";
    }
    std::string GetDeduceBody(TContext* context) const override {
        std::stringstream writer;
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";
        writer << R"(
                outputs[0]->layout = inputs[0]->layout;
                outputs[1]->layout = inputs[0]->layout;
                return TinyNN_SUCCESS;
            }
        )";
        return writer.str();
    }
};

class ConcatDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        int axis = context->getAttrInt("axis");
        return "DeduceFunc_concat_axis_" + std::to_string(axis);
    }
    std::string GetDeduceBody(TContext* context) const override {
        std::stringstream writer;
        writer << R"(
            #include "tensor_util.h"
        )";
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";

        std::string body = R"(
                uint32_t axis = ${axis};
                Tensor* output = outputs[0];
                output->layout = inputs[0]->layout;
                output->dtype = inputs[0]->dtype;
                for (uint32_t i = 1; i < nr_input; i++) {
                    output->layout.dims[axis] += inputs[i]->layout.dims[axis];    
                }
                force_layout_contiguous(&(output->layout));
                return TinyNN_SUCCESS;
            }
        )";
        int axis = context->getAttrInt("axis");
        writer << StringTemplate::StringTemplateArgs()
                          .add("axis", axis)
                          .render(body);
        return writer.str();
    }
};
class ElemwiseDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        return "DeduceFunc_Elemwise";
    }
    std::string GetDeduceBody(TContext* context) const override {
        std::stringstream writer;
        writer << R"(
            #include <math.h>
            #include <stdbool.h>
            #include "tensor_util.h"
            static inline bool is_scalar(Layout layout){
                return layout.nr_dim == 1 && layout.dims[0] == 1;
            }
            #define Max(a, b) (a) > (b) ? (a) : (b)
        )";
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";
        writer << R"(
                Layout dst;
                memset(&dst, 0, sizeof(Layout));
                
                for(int i = 0; i < nr_input; ++i){
                    Layout cur = inputs[i]->layout;
                    if (!cur.nr_dim)
                        TINYNN_ASSERT(0);
                    if (!dst.nr_dim || is_scalar(dst))
                        dst = cur;
                    else if (!is_scalar(cur)) {
                        int max_ndim = Max(cur.nr_dim, dst.nr_dim);
                        for (int i = 0; i < max_ndim; ++i) {
                            int cur_idx = cur.nr_dim - i - 1;
                            int dst_idx = dst.nr_dim - i - 1;
                            if (cur_idx >= 0 && dst_idx >= 0) {
                                size_t v0 = dst.dims[dst_idx], v1 = cur.dims[cur_idx];
                                if (v0 != v1) {
                                    if (v0 > 1 && v1 > 1)
                                        TINYNN_ASSERT(0);
                                }
                                int final_idx = Max(cur_idx, dst_idx);
                                dst.dims[final_idx] = (v0 != 0 && v1 != 0) ? Max(v0, v1) : 0;
                            } else {
                                if (dst_idx < 0) {
                                    dst.dims[cur_idx] = cur.dims[cur_idx];
                                }
                            }
                        }
                        dst.nr_dim = max_ndim;
                    }
                }
                force_layout_contiguous(&dst);
                outputs[0]->layout = dst;
                return TinyNN_SUCCESS;
            }
        )";
        return writer.str();
    }
};

class TypecvtDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        return "DeduceFunc_Typecvt";
    }
    std::string GetDeduceBody(TContext* context) const override {
        std::stringstream writer;
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";
        writer << R"(
                outputs[0]->layout = inputs[0]->layout;
                return TinyNN_SUCCESS;
            }
        )";
        return writer.str();
    }
};

class WarpPerspectiveDeduceLayout : public DeduceFunc {
public:
    std::string GetDeduceSymbol(TContext* context) const override {
        auto fmt = context->getAttrStr("format");
        return "DeduceFunc_WarpPerspective_" + fmt;
    }
    std::string GetDeduceBody(TContext* context) const override {
        auto fmt = context->getAttrStr("format");
        std::stringstream writer;
        writer << R"(
            #include "tensor_util.h"
        )";
        writer << GenCommonRet() << " " << GetDeduceSig(context) << "{\n";
        CC_ASSERT(fmt == "NCHW");
        writer << R"(
                Tensor* output = outputs[0];
                Tensor* out_shape;
                if (nr_input == 3) {
                    out_shape = inputs[2];
                } else {
                    TINYNN_ASSERT(nr_input == 4);
                    out_shape = inputs[3];
                }

                output->layout.nr_dim = 4;
                //! batch is equal to mat
                output->layout.dims[0] = inputs[1]->layout.dims[0];
                output->layout.dims[1] = inputs[0]->layout.dims[1];
                output->layout.dims[2] = get_tensor_value(out_shape, 0);
                output->layout.dims[3] = get_tensor_value(out_shape, 1);
                force_layout_contiguous(&(output->layout));
                return TinyNN_SUCCESS;
            }
        )";

        return writer.str();
    }
};

}  // namespace

DeduceLayoutMap::DeduceLayoutMap() {
    map[KernelPack::KernType::IndexingMultiAxisKernel] =
            std::make_shared<IndexingDeduceLayout>();
    map[KernelPack::KernType::ArgSortKernel] =
            std::make_shared<ArgSortDeduceLayout>();
    map[KernelPack::KernType::ConcatKernel] =
            std::make_shared<ConcatDeduceLayout>();
    map[KernelPack::KernType::ElemwiseKernel] =
            std::make_shared<ElemwiseDeduceLayout>();
    map[KernelPack::KernType::TypeCvtKernel] =
            std::make_shared<TypecvtDeduceLayout>();
    map[KernelPack::KernType::WarpPerspectiveKernel] =
            std::make_shared<WarpPerspectiveDeduceLayout>();
}