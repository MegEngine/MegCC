/**
 * \file
 * compiler/lib/KernelGen/Arm/ArmCommon/Reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "NeonIntrinCompat.h"
#include "Reduce.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace ArmCommon {

namespace {

// the base class to output the Reducer code
struct ReducerBase {
    virtual std::string get_declaration_simd_width() { return ""; }
    virtual std::string get_declaration() { return ""; };
    virtual std::string get_init(std::string&) { return ""; };
    virtual std::string get_feed(std::string&) { return ""; };
    virtual std::string get_feed_remain(std::string&) { return ""; };
    virtual std::string get_post(std::string&) { return ""; };
    virtual std::string get_post_remain(std::string&) { return ""; };
};

/*****************************Mean Reducer***********************/
template <typename dtype, typename ctype, typename comp_type, bool C1>
struct MeanReducer;

template <>
struct MeanReducer<float, float, float, true> : public ReducerBase {
    std::string get_declaration_simd_width() override {
        return R"(const int SIMD_WIDTH = 4;)";
    }
    std::string get_declaration() override {
        return R"(
        float32x4_t res;
        float result,coef;
        )";
    };
    std::string get_init(std::string& cnt) override {
        return R"(
            result = 0.0f;
            coef = 1.0 / )" +
               cnt + R"(;
            res = vdupq_n_f32(0.0f);)";
    };
    std::string get_feed(std::string& val) override {
        return R"(
                res = vaddq_f32(vld1q_f32()" +
               val + R"(), res);)";
        ;
    };
    std::string get_feed_remain(std::string& val) override {
        return R"(
                result += )" +
               val + R"(;)";
    };
    std::string get_post(std::string& dst) override {
        return R"(
            result += vaddvq_f32(res);
            )"
               "*" +
               dst + R"( = result * coef;)";
    };
};

template <>
struct MeanReducer<float, float, float, false>
        : public MeanReducer<float, float, float, true> {
    std::string get_feed_remain(std::string& val) override {
        return R"(result += )" + val + R"(;)";
    };
    std::string get_post(std::string& dst) override {
        return R"(res = vmulq_n_f32(res, coef);
                vst1q_f32()" +
               dst + R"(, res);)";
    };
    std::string get_post_remain(std::string& dst) override {
        return "*" + dst + R"( = result * coef;)";
    };
};

/******************************max min Reducer****************************/
template <typename dtype, typename ctype, typename comp_type, bool C1>
struct maxReducer;
template <typename dtype, typename ctype, typename comp_type, bool C1>
struct minReducer;
#define REDUCER_MAX_MIN_C1(_mode, _dtype, _ctype, _comp_type, _stype, __stype, \
                           _num, _init)                                        \
    template <>                                                                \
    struct _mode##Reducer<_dtype, _ctype, _comp_type, true>                    \
            : public ReducerBase {                                             \
        std::string get_declaration_simd_width() override {                    \
            return R"(const int SIMD_WIDTH = 4;)";                             \
        }                                                                      \
        std::string get_declaration() override { return #__stype " res;"; };   \
        std::string get_init(std::string&) override {                          \
            return R"(  res = vdupq_n_)" #_stype "(" #_init ");";              \
        };                                                                     \
        std::string get_feed(std::string& val) override {                      \
            return "\n"                                                        \
                   R"(                )" #__stype " vval = vld1q_" #_stype     \
                   "(" +                                                       \
                   val +                                                       \
                   ");\n"                                                      \
                   R"(                )"                                       \
                   "res = v" #_mode "q_" #_stype "(vval,res);\n";              \
        };                                                                     \
        std::string get_feed_remain(std::string& val) override {               \
            return "\n"                                                        \
                   R"(                )" #__stype " vval = vdupq_n_" #_stype   \
                   "(" +                                                       \
                   val +                                                       \
                   ");\n"                                                      \
                   R"(                )"                                       \
                   "res = v" #_mode "q_" #_stype "(vval,res);\n";              \
        };                                                                     \
        std::string get_post(std::string& dst) override {                      \
            return "float32x2_t val = v" #_mode "_" #_stype                    \
                   "(vget_low_" #_stype "(res),vget_high_" #_stype             \
                   "(res));\n"                                                 \
                   R"(            )"                                           \
                   "*" +                                                       \
                   dst +                                                       \
                   " = " #_mode "(vget_lane_" #_stype                          \
                   "(val,0),vget_lane_" #_stype "(val,1));";                   \
        };                                                                     \
    }

REDUCER_MAX_MIN_C1(max, float, float, float, f32, float32x4_t, 4, -__FLT_MAX__);
REDUCER_MAX_MIN_C1(min, float, float, float, f32, float32x4_t, 4, __FLT_MAX__);
#undef REDUCER_MAX_MIN_C1

#define REDUCER_MAX_MIN_C(_mode, _dtype, _ctype, _comp_type, _stype, __stype, \
                          _num, _init)                                        \
    template <>                                                               \
    struct _mode##Reducer<_dtype, _ctype, _comp_type, false>                  \
            : public ReducerBase {                                            \
        std::string get_declaration_simd_width() override {                   \
            return R"(const int SIMD_WIDTH = 4;)";                            \
        }                                                                     \
        std::string get_declaration() override {                              \
            return "\n"                                                       \
                   R"(        )" #__stype                                     \
                   " res;\n"                                                  \
                   R"(        )" #_ctype " remain;";                          \
        };                                                                    \
        std::string get_init(std::string&) override {                         \
            return "\n"                                                       \
                   R"(                )"                                      \
                   "res = vdupq_n_" #_stype "(" #_init                        \
                   ");\n"                                                     \
                   R"(                )"                                      \
                   "remain = " #_init ";";                                    \
        };                                                                    \
        std::string get_feed(std::string& val) override {                     \
            return #__stype " vval = vld1q_" #_stype "(" + val +              \
                   ");\n"                                                     \
                   R"(                    )"                                  \
                   "res = v" #_mode "q_" #_stype "(vval,res);\n";             \
        };                                                                    \
        std::string get_feed_remain(std::string& val) override {              \
            return "remain = " #_mode "(" + val + ",remain);\n";              \
        };                                                                    \
        std::string get_post(std::string& dst) override {                     \
            return "vst1q_" #_stype "(" + dst + ",res);\n";                   \
        }                                                                     \
        std::string get_post_remain(std::string& dst) override {              \
            return "*" + dst + " = remain;\n";                                \
        }                                                                     \
    }

REDUCER_MAX_MIN_C(max, float, float, float, f32, float32x4_t, 4, -__FLT_MAX__);
REDUCER_MAX_MIN_C(min, float, float, float, f32, float32x4_t, 4, __FLT_MAX__);
#undef REDUCER_MAX_MIN_C

/***************************Sum Product Reducer***************************/

template <typename dtype, typename ctype, typename comp_type, bool C1>
struct SumReducer;
template <typename dtype, typename ctype, typename comp_type, bool C1>
struct ProductReducer;
typedef enum { PLUS, MUL } OPR_TYPE;
template <OPR_TYPE opr>
std::string get_express(std::string x, std::string y);

template <>
std::string get_express<OPR_TYPE::PLUS>(std::string x, std::string y) {
    return "(" + x + " + " + y + ")";
}

template <>
std::string get_express<OPR_TYPE::MUL>(std::string x, std::string y) {
    return "(" + x + " * " + y + ")";
}

#define REDUCER_SUM_PRODUCT_C1(_mode, _dtype, _ctype, _comp_type, _stype,      \
                               __stype, _num, _init, _act, _op)                \
    template <>                                                                \
    struct _mode##Reducer<_dtype, _ctype, _comp_type, true>                    \
            : public ReducerBase {                                             \
        std::string get_declaration_simd_width() override {                    \
            return R"(const int SIMD_WIDTH = 4;)";                             \
        }                                                                      \
        std::string get_declaration() override {                               \
            return #__stype                                                    \
                    " res;\n"                                                  \
                    R"(        )" #_ctype " remain;\n";                        \
        };                                                                     \
        std::string get_init(std::string&) override {                          \
            return "  res = vdupq_n_" #_stype "(" #_init                       \
                   ");\n"                                                      \
                   R"(            remain = )" #_init ";";                      \
        };                                                                     \
        std::string get_feed(std::string& val) override {                      \
            return "\n"                                                        \
                   R"(                )" #__stype " vval = vld1q_" #_stype     \
                   "(" +                                                       \
                   val +                                                       \
                   ");\n"                                                      \
                   R"(                )"                                       \
                   "res = v" #_act "q_" #_stype "(vval,res);\n";               \
        };                                                                     \
        std::string get_feed_remain(std::string& val) override {               \
            return "\n"                                                        \
                   R"(                )"                                       \
                   "remain = " +                                               \
                   get_express<_op>("remain", val) + ";";                      \
        };                                                                     \
        std::string get_post(std::string& dst) override {                      \
            return "float32x2_t val = v" #_act "_" #_stype                     \
                   "(vget_low_" #_stype "(res),vget_high_" #_stype             \
                   "(res));\n"                                                 \
                   R"(            )"                                           \
                   "*" +                                                       \
                   dst + " = " +                                               \
                   get_express<_op>(                                           \
                           "remain",                                           \
                           get_express<_op>("vget_lane_" #_stype "(val,0)",    \
                                            "vget_lane_" #_stype "(val,1)")) + \
                   ";";                                                        \
        }                                                                      \
    }

REDUCER_SUM_PRODUCT_C1(Sum, float, float, float, f32, float32x4_t, 4, 0, add,
                       OPR_TYPE::PLUS);
REDUCER_SUM_PRODUCT_C1(Product, float, float, float, f32, float32x4_t, 4, 1.0f,
                       mul, OPR_TYPE::MUL);
#undef REDUCER_SUM_PRODUCT_C1

#define REDUCER_SUM_PRODUCT_C(_mode, _dtype, _ctype, _comp_type, _stype, \
                              __stype, _num, _init, _act, _op)           \
    template <>                                                          \
    struct _mode##Reducer<_dtype, _ctype, _comp_type, false>             \
            : public _mode##Reducer<_dtype, _ctype, _comp_type, true> {  \
        std::string get_post(std::string& dst) override {                \
            return "vst1q_" #_stype "(" + dst + ",res);\n";              \
        }                                                                \
        std::string get_post_remain(std::string& dst) override {         \
            return "*" + dst + " = remain;\n";                           \
        }                                                                \
    }

REDUCER_SUM_PRODUCT_C(Sum, float, float, float, f32, float32x4_t, 4, 0, add,
                      plus);
REDUCER_SUM_PRODUCT_C(Product, float, float, float, f32, float32x4_t, 4, 1, mul,
                      multiplies);

#undef REDUCER_SUM_PRODUCT_C

/***************************SumSqr Reducer***************************/
template <typename dtype, typename ctype, typename comp_type, bool C1>
struct SumSqrReducer;

template <>
struct SumSqrReducer<float, float, float, true> : public ReducerBase {
    std::string get_declaration_simd_width() override {
        return R"(const int SIMD_WIDTH = 4;)";
    }
    std::string get_declaration() override {
        return R"(
        float32x4_t res;
        float result;
        )";
    };
    std::string get_init(std::string&) override {
        return R"(
            result = 0.0f;
            res = vdupq_n_f32(0.0f);)";
    };
    std::string get_feed(std::string& val) override {
        return R"(
                float32x4_t vval = vld1q_f32()" +
               val + R"();
                res = vaddq_f32(vmulq_f32(vval,vval),res);)";
        ;
    };
    std::string get_feed_remain(std::string& val) override {
        return R"(
                float vval = )" +
               val +
               R"(;)"
               R"(
                result += vval * vval;)";
    };
    std::string get_post(std::string& dst) override {
        return R"(
            result += vaddvq_f32(res);
            )"
               "*" +
               dst + R"( = result;)";
    };
};

template <>
struct SumSqrReducer<float, float, float, false>
        : public SumSqrReducer<float, float, float, true> {
    std::string get_declaration() override {
        return R"(
        float32x4_t res;
        float remain;
        )";
    };
    std::string get_init(std::string&) override {
        return R"(
            remain = 0.0f;
            res = vdupq_n_f32(0.0f);)";
    };
    std::string get_feed_remain(std::string& val) override {
        return R"(
                remain += ()" +
               val + R"() * ()" + val + R"();)";
    };
    std::string get_post(std::string& dst) override {
        return R"(vst1q_f32()" + dst + R"(,res);)";
    };
    std::string get_post_remain(std::string& dst) override {
        return R"(*)" + dst + R"( = remain;)";
    };
};

/***************************Reducer Operations***************************/

template <typename dtype, typename ctype, typename comp_type, bool C1>
std::unique_ptr<ReducerBase> getReducer(std::string& mode) {
    if (mode == "MEAN")
        return std::make_unique<MeanReducer<dtype, ctype, comp_type, C1>>();
    else if (mode == "MAX")
        return std::make_unique<maxReducer<dtype, ctype, comp_type, C1>>();
    else if (mode == "MIN")
        return std::make_unique<minReducer<dtype, ctype, comp_type, C1>>();
    else if (mode == "SUM")
        return std::make_unique<SumReducer<dtype, ctype, comp_type, C1>>();
    else if (mode == "PRODUCT")
        return std::make_unique<ProductReducer<dtype, ctype, comp_type, C1>>();
    else if (mode == "SUM_SQR")
        return std::make_unique<SumSqrReducer<dtype, ctype, comp_type, C1>>();
    return nullptr;
}

template <bool c1>
std::string generate_reducer(std::string& mode);

template <>
std::string generate_reducer<false>(std::string& mode) {
    std::stringstream writer;
    auto reducer = getReducer<float, float, float, false>(mode);
    std::string init_str = "B", feed_str = "temp_src",
                feed_remain_str = "*temp_src", post_str = "dst",
                post_remain_str = "dst";
    writer << reducer->get_declaration_simd_width() << R"(
        )";
    writer << reducer->get_declaration();
    writer << R"(
        for (size_t a = 0; a < A; a++) {
            size_t c = 0;
            for (; c + SIMD_WIDTH <= C; c += SIMD_WIDTH){
                )"
           << reducer->get_init(init_str) << R"(
                for (size_t b = 0; b < B; b++){
                    float* temp_src = src + c + C * b;
                    )"
           << reducer->get_feed(feed_str) << R"(
                }
                )"
           << reducer->get_post(post_str) << R"(
                dst += SIMD_WIDTH;
            })"
           << R"(
            for (; c < C; c++){)"
           << reducer->get_init(init_str) << R"(
                for (size_t b = 0; b < B; b++){
                    float* temp_src = src + c + C * b;
                    )"
           << reducer->get_feed_remain(feed_remain_str) << R"(
                }
                )"
           << reducer->get_post_remain(post_remain_str) << R"(
                dst++;
            })"
           << R"(
            src += B*C;)"
              R"(
        })";
    return writer.str();
}

template <>
std::string generate_reducer<true>(std::string& mode) {
    std::stringstream writer;
    auto reducer = getReducer<float, float, float, true>(mode);
    std::string init_str = "B", feed_str = "temp_src",
                feed_remain_str = "*temp_src", post_str = "dst";
    writer << reducer->get_declaration_simd_width() << R"(
        )";
    writer << reducer->get_declaration();
    writer << R"(
        for (uint32_t i = 0; i < A; ++ i) {
          )"
           << reducer->get_init(init_str) << R"(
            float* temp_src = src + i * B;
            size_t b = 0;
            for(;b + SIMD_WIDTH < B; b += SIMD_WIDTH){)"
           << reducer->get_feed(feed_str) << R"(
                temp_src += SIMD_WIDTH;
            }
            for(;b < B;b++){)"
           << reducer->get_feed_remain(feed_remain_str) << R"(
                temp_src++;
            }
            )"
           << reducer->get_post(post_str) << R"(
            dst++;
        })";
    return writer.str();
}
}  // namespace

bool ReduceKernel::IsAvailable(TContext* context) const {
    auto data_type = context->getAttrStr("data_type");
    auto operand_type = context->getAttrOprand("operand:0").dtype;
    if (data_type == "DEFAULT" && operand_type == "f32") {
        return true;
    }
    // TODO:implement int quantized operand_type
    return false;
}
//! kernel gen
std::string ReduceKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "ArmCommon_kernel_reduce";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrStr("data_type");
    ss << "_a" << context->getAttrInt("axis");
    return ss.str();
}

std::string ReduceKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    size_t axis = context->getAttrInt("axis");
    auto input = context->getAttrOprand("operand:0");
    std::stringstream writer;
    writer << "#include<arm_neon.h>\n";
    writer << "static inline float max(float a,float b){return a>b?a:b;}\n";
    writer << "static inline float min(float a,float b){return a<b?a:b;}\n";
    writer << gen_neon_intrin_compat();
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << "const size_t axis = " << axis << ";\n";
    // clang-format off
    writer << R"(
    float* src = (float*)inputs[0]->ptr;
    float* dst = (float*)outputs[0]->ptr;
    Layout in_layout = inputs[0]->layout;
    size_t A = 1, B, C = 1;
    for (size_t i = 0; i < axis; ++ i)
        A *= in_layout.dims[i];
    B = in_layout.dims[axis];
    for (size_t i = axis + 1; i < in_layout.nr_dim; ++ i)
        C *= in_layout.dims[i];
    )";
    writer << R"(
    if(C == 1){
        )"
         << generate_reducer<true>(mode) << R"(
    })"
         << R"(else{
        )"
         << generate_reducer<false>(mode) << R"(

    })";
    writer << R"(
        return TinyNN_SUCCESS;
    })";
    // clang-format on
    return writer.str();
}

}  // namespace ArmCommon
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen