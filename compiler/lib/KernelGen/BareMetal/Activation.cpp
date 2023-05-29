#include "Activation.h"
#include <sstream>
#include <string>

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

std::string GenActivation::gen_func_dep(std::string mode) {
    if (mode == "RELU") {
        return R"(
            static inline float relu_act(float res){
                return res > 0? res:0;
            }
        )";
    } else if (mode.size() == 0 || mode == "IDENTITY") {
        return "";
    } else if (mode == "H_SWISH") {
        return R"(
static inline float min(float a, float b){
    return a > b? b:a;
}
static inline float max(float a, float b){
    return a > b? a:b;
}
static inline float hswish_act(float res){
    return res * min(max(res + 3, 0.f), 6.f) * (1.f / 6.f);
}

)";
    } else if (mode == "SIGMOID") {
        return R"(
            #include <math.h>
            static inline float sigmoid_act(float res){
                return 1.f / (expf(-res) + 1.f);
            }
        )";
    } else {
        CC_ABORT << "not support " << mode.c_str() << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call_with_typecvt_dep(
        std::string mode, std::string src_specifier, std::string dst_specifier) {
    auto act_dep = gen_func_dep(mode);
    if (src_specifier == "int" && dst_specifier == "int8_t") {
        return act_dep + R"(
            #include <math.h>
            static inline int8_t fp32_to_int8(float src){
                int res = roundf(src);
                res = res > 127? 127:res;
                res = res < -128? -128:res;
                return (int8_t)(res);
            }
        )";
    } else if (src_specifier == dst_specifier) {
        std::string type_cvt = "";
        if (src_specifier == "gi_float16_t") {
            type_cvt = R"(
static float FastFp16toFp32(const gi_float16_t data) { 
    const unsigned short x =*(unsigned short*)&data; 
    const unsigned int e = (x&0x7C00)>>10; 
    const unsigned int m = (x&0x03FF)<<13; 
    float m_ = (float)m;
    const unsigned int m_0 = *(unsigned int*)&m_;
    const unsigned int v = m_0 >>23;
    const unsigned int answer = (x&0x8000)<<16 | (e!=0)*((e+112)<<23|m) | ((e==0)&(m!=0))*((v-37)<<23|((m<<(150-v))&0x007FE000)); 
    return *(float*)&answer; 
}

static gi_float16_t FastFp32toFp16(const float x) { 
    const unsigned int b = (*(unsigned int*)&x)+0x00001000; 
    const unsigned int e = (b&0x7F800000)>>23;
    const unsigned int m = b&0x007FFFFF;
    unsigned short answer = (b&0x80000000)>>16 | (e>112)*((((e-112)<<10)&0x7C00)|m>>13) | ((e<113)&(e>101))*((((0x007FF000+m)>>(125-e))+1)>>1) | (e>143)*0x7FFF; 
    return *(gi_float16_t*)&answer; 
}   

    )";
        }
        return act_dep + type_cvt;
    } else {
        CC_ABORT << "not support type" << src_specifier << " to " << dst_specifier
                 << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call(std::string mode, std::string val) {
    if (mode == "RELU") {
        return "relu_act(" + val + ")";
    } else if (mode.size() == 0 || mode == "IDENTITY") {
        return val;
    } else if (mode == "SIGMOID") {
        return "sigmoid_act(" + val + ")";
    } else if (mode == "H_SWISH") {
        return "hswish_act(" + val + ")";
    } else {
        CC_ABORT << "not support " << mode.c_str() << val.c_str() << "\n";
    }
    return "";
}

std::string GenActivation::gen_func_call_with_typecvt(
        std::string mode, std::string args, std::string src_specifier,
        std::string dst_specifier, std::string scale_name, std::string flt_scale_name,
        std::string div_scale_name) {
    if (src_specifier == "int" && dst_specifier == "int8_t") {
        auto act_str = gen_func_call(
                mode,
                args + "*" + scale_name + "*" + flt_scale_name + "/" + div_scale_name);
        return "fp32_to_int8(" + act_str + ")";
    } else if (src_specifier == dst_specifier) {
        if (src_specifier == "gi_float16_t") {
            auto act_str = gen_func_call(mode, "FastFp16toFp32(" + args + ")");
            return "FastFp32toFp16(" + act_str + ")";
        }
        return gen_func_call(mode, args);
    } else {
        CC_ABORT << "not support type" << src_specifier << " to " << dst_specifier
                 << "\n";
    }
    return "";
}

// vim: syntax=cpp.doxygen
