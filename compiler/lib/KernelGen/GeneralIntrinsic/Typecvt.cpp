/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Typecvt.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <sstream>

#include "Typecvt.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool TypecvtKernel::IsAvailable(TContext* context) const {
    auto src_dtype = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:0").dtype);
    auto dst_dtype = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:1").dtype);
    bool ok_type =
            (Utils::is_quant_dtype(src_dtype, 8) &&
             Utils::is_quant_dtype(dst_dtype, 8)) ||
            (Utils::is_quant_dtype(src_dtype, 8) &&
             Utils::is_float_dtype(dst_dtype)) ||
            (Utils::is_float_dtype(src_dtype) &&
             Utils::is_quant_dtype(dst_dtype, 8)) ||
            (Utils::get_dtype_enum(src_dtype) == Utils::DtypeEnum::uint8 &&
             Utils::is_float_dtype(dst_dtype));
    if (Utils::is_quant_dtype(src_dtype)) {
        CC_ASSERT(context->getAttrOprand("operand:0").scale > 0);
    }
    if (Utils::is_quant_dtype(dst_dtype)) {
        CC_ASSERT(context->getAttrOprand("operand:1").scale > 0);
    }
    return ok_type;
}

//! kernel gen
std::string TypecvtKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_kernel_typecvt_" << SymbolHelper::gen_io_str(context);
    return ss.str();
}
namespace {
std::string init_declare(const std::string& src_dtype,
                         const std::string& dst_dtype) {
    std::string body_temp = R"(
        float scale;
        GI_FLOAT32_t vscale;
        const size_t SIMD_WIDTH = 16;
    )";
    return body_temp;
}

std::string gen_scale(const std::string& src_dtype,
                      const std::string& dst_dtype) {
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype)) {
        body_temp += R"(
            src_scale = 1;
         )";
    }
    if (Utils::is_float_dtype(dst_dtype)) {
        body_temp += R"(
           dst_scale = 1;
        )";
    }
    return body_temp;
}

std::string gen_cvt(const std::string& src_dtype,
                    const std::string& dst_dtype) {
    auto src_dtype_enum = Utils::get_dtype_enum(src_dtype);
    auto dst_dtype_enum = Utils::get_dtype_enum(dst_dtype);
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype) &&
        Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            GI_FLOAT32_t vitem0 = GiMultiplyFloat32(GiLoadFloat32(src), vscale);
            GI_FLOAT32_t vitem1 = GiMultiplyFloat32(GiLoadFloat32(src + 4), vscale);
            GI_FLOAT32_t vitem2 = GiMultiplyFloat32(GiLoadFloat32(src + 8), vscale);
            GI_FLOAT32_t vitem3 = GiMultiplyFloat32(GiLoadFloat32(src + 12), vscale);

            GI_FLOAT32_V4_t vitem;
            GiSetSubVectorFloat32V4(vitem, 0, vitem0);
            GiSetSubVectorFloat32V4(vitem, 1, vitem1);
            GiSetSubVectorFloat32V4(vitem, 2, vitem2);
            GiSetSubVectorFloat32V4(vitem, 3, vitem3);
            GI_INT8_t ans = GiCvtFromFloat32V4ToInt8(vitem);
            GiStoreInt8(dst, ans);
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_float_dtype(dst_dtype)) {
        body_temp = R"(
            GI_INT8_t src_reg = GiLoadInt8(src);
            GI_INT16_t vsrc0 = GiMoveLowLongInt8(src_reg);
            GI_INT16_t vsrc1 = GiMoveHighLongInt8(src_reg);
            GiStoreFloat32(dst, GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vsrc0)), vscale));
            GiStoreFloat32(dst+4, GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vsrc0)), vscale));
            GiStoreFloat32(dst+8, GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vsrc1)), vscale));
            GiStoreFloat32(dst+12, GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vsrc1)), vscale));

         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            GI_INT8_t src_reg = GiLoadInt8(src);
            GI_INT16_t vsrc0 = GiMoveLowLongInt8(src_reg);
            GI_INT16_t vsrc1 = GiMoveHighLongInt8(src_reg);
            GI_FLOAT32_t vitem0 =
            GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vsrc0)), vscale);
            GI_FLOAT32_t vitem1 =
            GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vsrc0)), vscale);

            GI_FLOAT32_t vitem2 =
            GiMultiplyFloat32(GiCastToFloat32(GiMoveLowLongInt16(vsrc1)), vscale);
            GI_FLOAT32_t vitem3 =
            GiMultiplyFloat32(GiCastToFloat32(GiMoveHighLongInt16(vsrc1)), vscale); 

            GI_FLOAT32_V4_t vitem;
            GiSetSubVectorFloat32V4(vitem, 0, vitem0);
            GiSetSubVectorFloat32V4(vitem, 1, vitem1);
            GiSetSubVectorFloat32V4(vitem, 2, vitem2);
            GiSetSubVectorFloat32V4(vitem, 3, vitem3);
            GI_INT8_t ans = GiCvtFromFloat32V4ToInt8(vitem);
            GiStoreInt8(dst, ans);
         )";
    } else if (src_dtype_enum == Utils::DtypeEnum::uint8 &&
               dst_dtype_enum == Utils::DtypeEnum::float32) {
        //! TODO: GI uint8 API is not integral this implement is little slow
        //! than using uint8 API directly, when GI uint8 API is updated, please
        //! optimize this implemnet
        body_temp = R"(
            GI_INT8_t u8_src = GiLoadInt8(src);
            GI_INT16_t vsrc0 = GiMoveLowLongInt8(u8_src);
            GI_INT16_t vsrc1 = GiMoveHighLongInt8(u8_src);
            GI_INT32_t vuint_mask = GiBroadcastInt32(0x000000ff);

            GI_FLOAT32_t vitem0 = GiCastToFloat32(GiAndInt32(GiMoveLowLongInt16(vsrc0), vuint_mask));
            GI_FLOAT32_t vitem1 = GiCastToFloat32(GiAndInt32(GiMoveHighLongInt16(vsrc0), vuint_mask));
            GI_FLOAT32_t vitem2 = GiCastToFloat32(GiAndInt32(GiMoveLowLongInt16(vsrc1), vuint_mask));
            GI_FLOAT32_t vitem3 = GiCastToFloat32(GiAndInt32(GiMoveHighLongInt16(vsrc1),vuint_mask));
            GiStoreFloat32(dst + 0 * 4, vitem0);
            GiStoreFloat32(dst + 1 * 4, vitem1);
            GiStoreFloat32(dst + 2 * 4, vitem2);
            GiStoreFloat32(dst + 3 * 4, vitem3);
         )";
    } else {
        CC_ABORT << "General Intrinsic not support optimise cvt " << src_dtype
                 << "->" << dst_dtype << "\n";
    }
    return body_temp;
}

std::string gen_cvt_remain(const std::string& src_dtype,
                           const std::string& dst_dtype) {
    auto src_dtype_enum = Utils::get_dtype_enum(src_dtype);
    auto dst_dtype_enum = Utils::get_dtype_enum(dst_dtype);
    std::string body_temp;
    if (Utils::is_float_dtype(src_dtype) &&
        Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
                float val = (*src)*scale;
                int dst_val = roundf(val);
                dst_val = dst_val >= 127? 127:dst_val;
                dst_val = dst_val <= -128? -128:dst_val;
                *dst = dst_val;
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_float_dtype(dst_dtype)) {
        body_temp = R"(
            *dst = (*src)*scale;
         )";
    } else if (Utils::is_quant_dtype(src_dtype, 8) &&
               Utils::is_quant_dtype(dst_dtype, 8)) {
        body_temp = R"(
            float val = (*src)*scale;
            int dst_val = roundf(val);
            dst_val = dst_val >= 127? 127:dst_val;
            dst_val = dst_val <= -128? -128:dst_val;
            *dst = dst_val;
         )";
    } else if (src_dtype_enum == Utils::DtypeEnum::uint8 &&
               dst_dtype_enum == Utils::DtypeEnum::float32) {
        body_temp = R"(
            *dst = (float)*src;
         )";
    } else {
        CC_ABORT << "General Intrinsic not support optimise cvt " << src_dtype << "->"
                 << dst_dtype << "\n";
    }
    return body_temp;
}

}  // namespace

std::string TypecvtKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    auto src_dtype_str = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:0").dtype);
    auto dst_dtype_str = SymbolHelper::gen_valid_dtype(
            context->getAttrOprand("operand:1").dtype);
    std::string src_specifier = Utils::cvt_dtype_specifier(src_dtype_str);
    std::string dst_specifier = Utils::cvt_dtype_specifier(dst_dtype_str);
    ss << R"(
    #include "gi_float.h"
    #include "gi_int.h"
    )";
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    ${init_declare_str}
    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    ${src_specifier}* src = (${src_specifier}*)(src_tensor->ptr);
    ${dst_specifier}* dst = (${dst_specifier}*)(dst_tensor->ptr);
    TINYNN_ASSERT(src);
    TINYNN_ASSERT(dst);
    
    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;
    float src_scale = src_tensor->dtype.param.scale;
    float dst_scale = dst_tensor->dtype.param.scale;

    size_t nr_elem = 1;
    for (int i = 0; i < src_layout.nr_dim; ++i) {
        nr_elem *= src_layout.dims[i];
    }
    ${gen_scale}
    scale = src_scale/dst_scale;
    vscale = GiBroadcastFloat32(scale);
    size_t idx = 0;
    
    for(; idx + SIMD_WIDTH <= nr_elem; idx += SIMD_WIDTH){
        ${gen_cvt}
        src += SIMD_WIDTH;
        dst += SIMD_WIDTH;
    }

    for(;idx < nr_elem;++idx){
        ${gen_cvt_remain}
        ++src;
        ++dst;
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("init_declare_str", init_declare(src_dtype_str, dst_dtype_str))
                    .add("src_specifier", src_specifier)
                    .add("dst_specifier", dst_specifier)
                    .add("gen_scale", gen_scale(src_dtype_str, dst_dtype_str))
                    .add("gen_cvt", gen_cvt(src_dtype_str, dst_dtype_str))
                    .add("gen_cvt_remain",
                         gen_cvt_remain(src_dtype_str, dst_dtype_str))
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen
