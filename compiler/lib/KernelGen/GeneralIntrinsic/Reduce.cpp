/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Reduce.h"
#include "GIMathHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
struct Rdeucer {
    std::string m_ctype;
    std::string simd_type;
    std::string init_str;
    std::string load_str;
    std::string store_str;
    Rdeucer(std::string ctype) : m_ctype(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            simd_type = "GI_FLOAT32_t";
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            simd_type = "GI_FLOAT16_t";
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }
        init_str = "GiBroadcast" + suffix_ins;
        load_str = "GiLoad" + suffix_ins;
        store_str = "GiStore" + suffix_ins;
    }
    virtual std::string gen_init(bool c_remain) = 0;
    virtual std::string gen_feed() = 0;
    virtual std::string gen_feed_remain() = 0;
    virtual std::string gen_post(bool c_remain) = 0;
};
struct MinReducerC1 final : public Rdeucer {
    std::string init_vaule;
    std::string core_str;
    std::string post_str;
    MinReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
            init_vaule = "__FLT_MAX__";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
            init_vaule = "65504";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMinimum" + suffix_ins;
        post_str = "GiReduceMinNan" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer = min(*temp_src0, answer);
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer =min(answer, " + post_str + "(answer_vec));\n" +
               "*dst = answer" + ";\n";
    };
};
struct MinReducer final : public Rdeucer {
    std::string init_vaule;
    std::string core_str;
    std::string post_str;
    MinReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
            init_vaule = "__FLT_MAX__";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
            init_vaule = "65504";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMinimum" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer = min(*temp_src0, answer);
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, answer_vec);\n";
        else
            return R"(
        *dst = answer;
            )";
    };
};
struct MaxReducerC1 final : public Rdeucer {
    std::string init_vaule;
    std::string core_str;
    std::string post_str;
    MaxReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
            init_vaule = "-__FLT_MAX__";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
            init_vaule = "-65504";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMaximum" + suffix_ins;
        post_str = "GiReduceMaxNan" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer = max(*temp_src0, answer);
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer =max(answer, " + post_str + "(answer_vec));\n" +
               "*dst = answer" + ";\n";
    };
};
struct MaxReducer final : public Rdeucer {
    std::string init_vaule;
    std::string core_str;
    MaxReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
            init_vaule = "-__FLT_MAX__";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
            init_vaule = "-65504";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMaximum" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer = max(*temp_src0, answer);
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, answer_vec);\n";
        else
            return R"(
        *dst = answer;
            )";
    };
};
struct SumReducerC1 final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    std::string post_str;
    SumReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiAdd" + suffix_ins;
        post_str = "GiReduceAdd" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer +=" + post_str + "(answer_vec);\n" + "*dst = answer" + ";\n";
    };
};
struct SumReducer final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    SumReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiAdd" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, answer_vec);\n";
        else
            return R"(
        *dst = answer;
            )";
    }
};
struct SumSqrReducerC1 final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    std::string post_str;
    SumSqrReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMultiplyAdd" + suffix_ins;
        post_str = "GiReduceAdd" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += (*temp_src0)*(*temp_src0);
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer +=" + post_str + "(answer_vec);\n" + "*dst = answer" + ";\n";
    };
};
struct SumSqrReducer final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    SumSqrReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMultiplyAdd" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += (*temp_src0)*(*temp_src0);
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, answer_vec);\n";
        else
            return R"(
        *dst = answer;
            )";
    }
};
struct MeanReducerC1 final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    std::string post_str;
    MeanReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiAdd" + suffix_ins;
        post_str = "GiReduceAdd" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer +=" + post_str + "(answer_vec);\n" + "answer /= B;\n" +
               "*dst = answer" + ";\n";
    };
};
struct MeanReducer final : public Rdeucer {
    std::string init_vaule = "0";
    std::string core_str;
    std::string post_str;
    MeanReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiAdd" + suffix_ins;
        post_str = "GiMultiplyScaler" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer += *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, " + post_str + "(answer_vec, 1.0/B));\n";
        else
            return R"(
        *dst = answer/B;
            )";
    }
};

struct ProductReducerC1 final : public Rdeucer {
    std::string init_vaule = "1";
    std::string core_str;
    std::string post_str;
    ProductReducerC1(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMultiply" + suffix_ins;
        post_str = "GiReduceMultiply" + suffix_ins;
    }
    std::string gen_init(bool c_remain = false) override {
        return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n" +
               m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer *= *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain = false) override {
        return " answer *=" + post_str + "(answer_vec);\n" + "*dst = answer" + ";\n";
    };
};
struct ProductReducer final : public Rdeucer {
    std::string init_vaule = "1";
    std::string core_str;
    ProductReducer(std::string ctype) : Rdeucer(ctype) {
        std::string suffix_ins;
        if (m_ctype == "float") {
            suffix_ins = "Float32";
        } else if (m_ctype == "gi_float16_t") {
            suffix_ins = "Float16";
        } else {
            CC_ABORT << "unsupported dtype for min reduce";
        }

        core_str = "GiMultiply" + suffix_ins;
    }
    std::string gen_init(bool c_remain) override {
        if (!c_remain)
            return simd_type + " answer_vec = " + init_str + "(" + init_vaule + ");\n";
        else
            return m_ctype + " answer = " + init_vaule + ";\n";
    };
    std::string gen_feed() override {
        return simd_type + " src_vec = " + load_str + "(temp_src0);\n" +
               " answer_vec = " + core_str + "(answer_vec, src_vec)" + ";\n";
    };
    std::string gen_feed_remain() override {
        return R"(
        answer *= *temp_src0;
        )";
    };
    std::string gen_post(bool c_remain) override {
        if (!c_remain)
            return store_str + "(dst, answer_vec);\n";
        else
            return R"(
        *dst = answer;
            )";
    }
};

std::shared_ptr<Rdeucer> get_reducer(
        std::string mode, bool c_is_one, std::string src_specifier) {
    if ("MIN" == mode) {
        if (c_is_one)
            return std::make_shared<MinReducerC1>(src_specifier);
        else
            return std::make_shared<MinReducer>(src_specifier);
    } else if ("MAX" == mode) {
        if (c_is_one)
            return std::make_shared<MaxReducerC1>(src_specifier);
        else
            return std::make_shared<MaxReducer>(src_specifier);
    } else if ("SUM" == mode) {
        if (c_is_one)
            return std::make_shared<SumReducerC1>(src_specifier);
        else
            return std::make_shared<SumReducer>(src_specifier);
    } else if ("SUM_SQR" == mode) {
        if (c_is_one)
            return std::make_shared<SumSqrReducerC1>(src_specifier);
        else
            return std::make_shared<SumSqrReducer>(src_specifier);
        return nullptr;
    } else if ("MEAN" == mode) {
        if (c_is_one)
            return std::make_shared<MeanReducerC1>(src_specifier);
        else
            return std::make_shared<MeanReducer>(src_specifier);
    } else if ("PRODUCT" == mode) {
        if (c_is_one)
            return std::make_shared<ProductReducerC1>(src_specifier);
        else
            return std::make_shared<ProductReducer>(src_specifier);
    } else {
        CC_ABORT << "unsupported reduce mode: " << mode;
    }
    return nullptr;
}

template <bool C_is_one>
std::string generate_reducer(std::string mode, std::string) {
    return "";
}

template <>
std::string generate_reducer<true>(std::string mode, std::string src_specifier) {
    auto reducer = get_reducer(mode, true, src_specifier);
    std::string func_body = R"(
for(size_t a = 0; a < A; ++a){
    ${src_specifier}* temp_src0 = src + a * B;
    size_t b = 0;
    ${init}
    for(; b +SIMD_WIDTH <= B; b += SIMD_WIDTH){
        ${feed};
        temp_src0 += SIMD_WIDTH;
    }
    for(; b < B; ++b){
        ${feed_remain};
        temp_src0++;
    }
    ${post};
    dst++;  
}
        )";
    return StringTemplate::StringTemplateArgs()
            .add("src_specifier", src_specifier)
            .add("init", reducer->gen_init(false))
            .add("feed", reducer->gen_feed())
            .add("feed_remain", reducer->gen_feed_remain())
            .add("post", reducer->gen_post(false))
            .render(func_body);
}

template <>
std::string generate_reducer<false>(std::string mode, std::string src_specifier) {
    auto reducer = get_reducer(mode, false, src_specifier);
    std::string func_body = R"(
for(size_t a = 0; a < A; ++a){
    size_t c = 0;
    for(; c + SIMD_WIDTH <= C; c += SIMD_WIDTH){
        ${init}
        ${src_specifier}* temp_src0 = src + a * B * C + c; 
        for (size_t b = 0; b < B; b++){
            ${feed};
            temp_src0 += C;
        }
        ${post};
        dst += SIMD_WIDTH;

       
    }
    for(; c < C; ++c){
        ${init_remain}
        ${src_specifier}* temp_src0 = src + a * B * C + c; 
        for (size_t b = 0; b < B; b++){
            ${feed_remain};
            temp_src0 +=C; 
        } 
        ${post_remain};
        dst++;
    }  
}
        )";
    return StringTemplate::StringTemplateArgs()
            .add("src_specifier", src_specifier)
            .add("init", reducer->gen_init(false))
            .add("init_remain", reducer->gen_init(true))
            .add("feed", reducer->gen_feed())
            .add("feed_remain", reducer->gen_feed_remain())
            .add("post", reducer->gen_post(false))
            .add("post_remain", reducer->gen_post(true))
            .render(func_body);
}

}  // namespace
bool ReduceKernel::IsAvailable(TContext* context) const {
    auto data_type = context->getAttrStr("data_type");
    auto operand_type = context->getAttrOprand("operand:0").dtype;
    if (data_type == "DEFAULT" && (operand_type == "f32" || operand_type == "f16")) {
        return true;
    }
    // TODO:implement int quantized operand_type
    return false;
}
//! kernel gen
std::string ReduceKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "GI_kernel_reduce";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrStr("data_type");
    ss << "_" << context->getAttrOprand("operand:0").dtype;
    ss << "_a" << context->getAttrInt("axis");
    return ss.str();
}

std::string ReduceKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    int axis = context->getAttrInt("axis");
    auto input = context->getAttrOprand("operand:0");
    auto src_dtype = input.dtype;
    std::stringstream writer;
    writer << R"(
#include "gi_float.h"
#include "gi_int.h"
    )";
    if (src_dtype == "f16") {
        writer << R"(
#include "gi_float16.h"
static inline gi_float16_t max(gi_float16_t a,gi_float16_t b){return a>b?a:b;}
static inline gi_float16_t min(gi_float16_t a,gi_float16_t b){return a<b?a:b;}    
    )";
        GIMathHelper gi_math;
        if ("MIN" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceMinNanFloat16() << "\n";
        } else if ("MAX" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceMaxNanFloat16() << "\n";
        } else if ("SUM" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceAddFloat16() << "\n";
        } else if ("SUM_SQR" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceAddFloat16() << "\n";
            writer << gi_math.GiMultiplyAddFloat16() << "\n";
        } else if ("MEAN" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceAddFloat16() << "\n";
        } else if ("PRODUCT" == mode) {
            writer << gi_math.FastFp32toFp16() << "\n";
            writer << gi_math.GiReduceMultiplyFloat16() << "\n";
        }
    } else {
        writer << R"(
static inline float max(float a,float b){return a>b?a:b;}
static inline float min(float a,float b){return a<b?a:b;}    
    )";
    }
    writer << GenCommonRet() << " " << GetKernelSignature(context);
    std::string tmp_body = R"({
    const size_t SIMD_WIDTH = ${simd_width};
    const size_t axis = ${axis};
    ${src_specifier}* src = (${src_specifier}*)inputs[0]->ptr;
    ${src_specifier}* dst = (${src_specifier}*)outputs[0]->ptr;
    Layout in_layout = inputs[0]->layout;
    size_t A = 1, B = 0, C = 1;
    for (size_t i = 0; i < axis; ++ i)
        A *= in_layout.dims[i];
    B = in_layout.dims[axis];
    for (size_t i = axis + 1; i < in_layout.nr_dim; ++ i){
        C *= in_layout.dims[i];
    }
    if(C == 1){
        ${gen_reducer_c1}
    }else{
        ${gen_reducer_cx}
    }
    return TinyNN_SUCCESS;
}
    )";
    writer << StringTemplate::StringTemplateArgs()
                      .add("axis", axis)
                      .add("src_specifier", Utils::cvt_dtype_specifier(src_dtype))
                      .add("gen_reducer_c1",
                           generate_reducer<true>(
                                   mode, Utils::cvt_dtype_specifier(src_dtype)))
                      .add("gen_reducer_cx",
                           generate_reducer<false>(
                                   mode, Utils::cvt_dtype_specifier(src_dtype)))
                      .add("simd_width",
                           (uint32_t)(16 / Utils::get_dtype_size(src_dtype)))
                      .render(tmp_body);
    // clang-format on
    return writer.str();
}

// vim: syntax=cpp.doxygen