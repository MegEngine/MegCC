/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/Reduce.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include "Reduce.h"
#include "compiler/Common/Logger.h"
#include "Utils/StringTemplate.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace{
struct Rdeucer{
    virtual std::string gen_init(bool c_remain) = 0; 
    virtual std::string gen_feed() = 0;
    virtual std::string gen_feed_remain() = 0;
    virtual std::string gen_post(bool c_remain) = 0;
};
struct MinReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(__FLT_MAX__);
        float ans = __FLT_MAX__; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMinimumFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans = min(*temp_src0, ans);
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans =min(ans, GiReduceMinNanFloat32(ans_vec));
            *dst = ans;
        )";
    };
};
struct MinReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(__FLT_MAX__); 
            )";
        else
            return R"(
        float ans = __FLT_MAX__; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMinimumFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans = min(*temp_src0, ans);
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        *dst = ans;
            )";
          
    };
};
struct MaxReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(- __FLT_MAX__);
        float ans = - __FLT_MAX__; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMaximumFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans = max(*temp_src0, ans);
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans =max(ans, GiReduceMaxNanFloat32(ans_vec));
            *dst = ans;
        )";
    };
};
struct MaxReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(-__FLT_MAX__); 
            )";
        else
            return R"(
        float ans = - __FLT_MAX__; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMaximumFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans = max(*temp_src0, ans);
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        *dst = ans;
            )";
          
    };
};
struct SumReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0);
        float ans = 0; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiAddFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans += GiReduceAddFloat32(ans_vec);
            *dst = ans;
        )";
    };
};
struct SumReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0); 
            )";
        else
            return R"(
        float ans =0; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiAddFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        *dst = ans;
            )";
          
    };
};
struct SumSqrReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0);
        float ans = 0; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMultiplyAddFloat32(ans_vec, src_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += (*temp_src0)*(*temp_src0);
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans += GiReduceAddFloat32(ans_vec);
            *dst = ans;
        )";
    };
};
struct SumSqrReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0); 
            )";
        else
            return R"(
        float ans =0; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0); 
        ans_vec = GiMultiplyAddFloat32(ans_vec, src_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += (*temp_src0)*(*temp_src0);
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        *dst = ans;
            )";
          
    };
};
struct MeanReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0);
        float ans = 0; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiAddFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans += GiReduceAddFloat32(ans_vec);
            ans /= B;
            *dst = ans;
        )";
    };
};
struct MeanReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(0); 
            )";
        else
            return R"(
        float ans =0; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiAddFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans += *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        ans_vec = GiMultiplyScalerFloat32(ans_vec, 1.0/B);
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        ans /= B;
        *dst = ans;
            )";
          
    };
};

struct ProductReducerC1 final: public Rdeucer {
    std::string gen_init(bool c_remain = false) override {
        return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(1);
        float ans = 1; 
        )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMultiplyFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans *= *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain = false) override {
        return R"(
            ans *= GiReduceMultiplyFloat32(ans_vec);
            *dst = ans;
        )";
    };
};
struct ProductReducer final: public Rdeucer {
    std::string gen_init(bool c_remain) override {
        if(!c_remain)
            return R"(
        GI_FLOAT32_t ans_vec = GiBroadcastFloat32(1); 
            )";
        else
            return R"(
        float ans = 1; 
            )"; 
    }; 
    std::string gen_feed() override {
        return R"(
        GI_FLOAT32_t src_vec = GiLoadFloat32(temp_src0);
        ans_vec = GiMultiplyFloat32(ans_vec, src_vec);

        )";
    };
    std::string gen_feed_remain() override {
        return R"(
        ans *= *temp_src0;
        )"; 
    };
    std::string gen_post(bool c_remain) override {
        if(!c_remain)
            return R"(
        GiStoreFloat32(dst, ans_vec);
            )";
        else
            return R"(
        *dst = ans;
            )";
          
    };
};

std::shared_ptr<Rdeucer> get_reducer(std::string mode, bool c_is_one){
    if("MIN" == mode){
        if(c_is_one)
            return std::make_shared<MinReducerC1>();
        else
            return std::make_shared<MinReducer>(); 
    }else if ("MAX" == mode) {
        if(c_is_one)
            return std::make_shared<MaxReducerC1>();
        else
            return std::make_shared<MaxReducer>(); 
    }else if ("SUM" == mode) {
        if(c_is_one)
            return std::make_shared<SumReducerC1>();
        else
            return std::make_shared<SumReducer>(); 
    }else if ("SUM_SQR" == mode) {
         if(c_is_one)
            return std::make_shared<SumSqrReducerC1>();
        else
            return std::make_shared<SumSqrReducer>(); 
        return nullptr;
    }else if ("MEAN" == mode) {
        if(c_is_one)
            return std::make_shared<MeanReducerC1>();
        else
            return std::make_shared<MeanReducer>(); 
    }else if ("PRODUCT" == mode) {
        if(c_is_one)
            return std::make_shared<ProductReducerC1>();
        else
            return std::make_shared<ProductReducer>(); 
    }else{
        CC_ABORT << "unsupport reduce mode: " << mode;
    }
    return nullptr;
}

template <bool C_is_one>
std::string generate_reducer(std::string mode){
        return "";
   } 

template <>
std::string generate_reducer<true>(std::string mode){
        auto reducer = get_reducer(mode, true);
        std::string func_body = R"(
for(size_t a = 0; a < A; ++a){
    float* temp_src0 = src + a * B;
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
                    .add("init", reducer->gen_init(false))
                    .add("feed", reducer->gen_feed())
                    .add("feed_remain", reducer->gen_feed_remain())
                    .add("post", reducer->gen_post(false))
                    .render(func_body);
   }  

template <>
std::string generate_reducer<false>(std::string mode){
        auto reducer = get_reducer(mode, false);
        std::string func_body = R"(
for(size_t a = 0; a < A; ++a){
    size_t c = 0;
    for(; c + SIMD_WIDTH <= C; c += SIMD_WIDTH){
        ${init}
        float* temp_src0 = src + a * B * C + c; 
        for (size_t b = 0; b < B; b++){
            ${feed};
            temp_src0 += C;
        }
        ${post};
        dst += SIMD_WIDTH;

       
    }
    for(; c < C; ++c){
        ${init_remain}
        float* temp_src0 = src + a * B * C + c; 
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
                    .add("init", reducer->gen_init(false))
                    .add("init_remain", reducer->gen_init(true))
                    .add("feed", reducer->gen_feed())
                    .add("feed_remain", reducer->gen_feed_remain())
                    .add("post", reducer->gen_post(false))
                    .add("post_remain", reducer->gen_post(true))
                    .render(func_body);
   }      

}
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
    ss << "GI_kernel_reduce";
    ss << "_" << context->getAttrStr("mode");
    ss << "_" << context->getAttrStr("data_type");
    ss << "_a" << context->getAttrInt("axis");
    return ss.str();
}

std::string ReduceKernel::GetKernelBody(TContext* context) const {
    auto mode = context->getAttrStr("mode");
    int axis = context->getAttrInt("axis");
    auto input = context->getAttrOprand("operand:0");
    std::stringstream writer;
    writer << R"(
#include "gi_int.h"
#include "gi_float.h"
static inline float max(float a,float b){return a>b?a:b;}
static inline float min(float a,float b){return a<b?a:b;}    
    )";
    writer << GenCommonRet() << " " << GetKernelSignature(context);
    std::string tmp_body = R"({
    const size_t SIMD_WIDTH = 4;
    const size_t axis = ${axis};
    float* src = (float*)inputs[0]->ptr;
    float* dst = (float*)outputs[0]->ptr;
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
                    .add("gen_reducer_c1", generate_reducer<true>(mode))
                    .add("gen_reducer_cx", generate_reducer<false>(mode))
                    .render(tmp_body);
    // clang-format on
    return writer.str();
}

   // vim: syntax=cpp.doxygen