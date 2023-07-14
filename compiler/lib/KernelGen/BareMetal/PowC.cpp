#include <float.h>
#include <sstream>

#include "FormatHelper.h"
#include "Fp16Common.h"
#include "PowC.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool PowCKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32" ||
                    context->getAttrOprand("operand:0").dtype == "f16";
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
    ss << "kernel_powc_" << exp_str << "_" << context->getAttrOprand("operand:0").dtype;
    return ss.str();
}

std::string PowCKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    ss << "#include <math.h>\n";
    std::string dtype =
            Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype);
    if (dtype == "gi_float16_t")
        ss << gen_fp16_define();
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    ${dtype}* a_data = (${dtype}*)inputs[0]->ptr;
    ${dtype}* c_data = (${dtype}*)outputs[0]->ptr;
    TINYNN_ASSERT(a_data);
    TINYNN_ASSERT(c_data);
    const Tensor* a_tensor = inputs[0];
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
                    .add("dtype", dtype)
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen