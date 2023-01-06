/**
 * \file
 * compiler/lib/KernelGen/Common/ConvKernel.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"

namespace megcc {
namespace KernelGen {
class ConvImpl : public KernelFunc {
public:
    static bool is_bias(TContext* ctx) {
        return ctx->getAttrInt("nr_operands") > 3 &&
               ctx->getAttrOprand("operand:2").shape.size() > 0;
    }
    static bool is_no_pad(TContext* ctx) {
        auto pad_h = ctx->getAttrInt("pad_h");
        auto pad_w = ctx->getAttrInt("pad_w");
        return pad_h == 0 && pad_w == 0;
    }
    std::string GetKernelSymbol(TContext* context) const override;

    static bool is_qint8_conv_dtype(TContext* ctx,
                                    bool is_dst_support_si32 = false) {
        bool type_ok = ctx->getAttrInt("nr_operands") >= 3;
        auto dst_dtype = Utils::get_last_operand(ctx).dtype;
        type_ok = type_ok && Utils::is_quant_dtype(
                                     ctx->getAttrOprand("operand:0").dtype, 8);
        type_ok = type_ok && Utils::is_quant_dtype(
                                     ctx->getAttrOprand("operand:1").dtype, 8);
        if (is_dst_support_si32) {
            type_ok = type_ok && (Utils::is_quant_dtype(dst_dtype, 8) ||
                                  Utils::is_quant_dtype(dst_dtype, 32));
        } else {
            type_ok = type_ok && Utils::is_quant_dtype(dst_dtype, 8);
        }
        if (is_bias(ctx)) {
            type_ok = type_ok &&
                      Utils::is_quant_dtype(
                              ctx->getAttrOprand("operand:2").dtype, 32);
        }
        return type_ok;
    }
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
