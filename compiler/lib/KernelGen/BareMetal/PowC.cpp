/**
 * \file
 * compiler/lib/KernelGen/BareMetal/PowC.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "PowC.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool PowCKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32";
    return ok_dtype;
}

//! kernel gen
std::string PowCKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    auto exp_str = std::to_string(context->getAttrFloat("exp"));
    if (exp_str.find('.') != std::string::npos) {
        exp_str[exp_str.find('.')] = '_';
    }
    if (exp_str.find('-') != std::string::npos) {
        exp_str[exp_str.find('-')] = 'N';
    }
    ss << "kernel_powc_" << exp_str;
    return ss.str();
}

std::string PowCKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    ss << "#include <math.h>\n";
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    float* a_data = (float*)inputs[0]->ptr;
    float* c_data = (float*)outputs[0]->ptr;
    TINYNN_ASSERT(a_data);
    TINYNN_ASSERT(c_data);
    const Tensor* a_tensor = inputs[0];
    const Tensor* c_tensor = outputs[0];
    const Layout a_layout = a_tensor->layout;
    size_t nr_elem = 1;
    for (int i = 0; i < a_layout.nr_dim; ++i) {
        nr_elem *= a_layout.dims[i];
    }
    
    for(size_t i = 0; i < nr_elem; ++i){
        c_data[i] = powf(a_data[i], ${exp});
    }
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("exp", std::to_string(context->getAttrFloat("exp")))
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen