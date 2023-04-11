#include "Argmax.h"
#include "../Utils/Utils.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;
namespace {
std::string gen_init(std::string mode, std::string dtype) {
    Utils::DtypeHelper dtype_helper(dtype);
    std::stringstream writer;
    if (mode == "MAX") {
        writer << dtype_helper.min();
    } else if (mode == "MIN") {
        writer << dtype_helper.max();
    } else {
        CC_ABORT << "unknown argmxx mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
std::string gen_comp(std::string mode) {
    std::stringstream writer;
    if (mode == "MAX") {
        writer << "curr_val > best_val";
    } else if (mode == "MIN") {
        writer << "curr_val < best_val";
    } else {
        CC_ABORT << "unknown argmxx mode " << mode.c_str() << "\n";
    }
    return writer.str();
}
}  // namespace
bool ArgmaxKernel::IsAvailable(TContext* context) const {
    bool ok_dtype = context->getAttrOprand("operand:0").dtype == "f32";
    return ok_dtype;
}
//! kernel gen
std::string ArgmaxKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_Argmax_" << context->getAttrOprand("operand:0").dtype << "_a"
       << context->getAttrInt("axis");
    return ss.str();
}

std::string ArgmaxKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    size_t axis = context->getAttrInt("axis");
    auto input_dtype = context->getAttrOprand("operand:0").dtype;
    auto output_dtype = context->getAttrOprand("operand:1").dtype;
    auto input_specifier = Utils::cvt_dtype_specifier(input_dtype);
    auto output_specifier = Utils::cvt_dtype_specifier(output_dtype);
    writer << "#include <string.h>\n";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    writer << "const size_t axis = " << axis << ";\n";
    // clang-format off
    writer << StringTemplate::StringTemplateArgs()
                      .add("input_specifier", input_specifier)
                      .add("output_specifier", output_specifier)
                      .add("gen_init", gen_init("MAX", input_dtype))
                      .add("gen_comp", gen_comp("MAX"))
                      .render(R"(
        ${input_specifier}* input_data = (${input_specifier}*)inputs[0]->ptr;
        ${output_specifier}* output_data = (${output_specifier}*)outputs[0]->ptr;

        Layout in_layout = inputs[0]->layout;
        int A = 1, B, C = 1;
        for (int i = 0; i < axis; ++ i)
            A *= in_layout.dims[i];
        B = in_layout.dims[axis];
        for (int i = axis + 1; i < in_layout.nr_dim; ++ i)
            C *= in_layout.dims[i];

        for (int a = 0; a < A; ++ a) {
            for (int c = 0; c < C; ++ c) {
                ${input_specifier} best_val = ${gen_init};
                size_t best_arg = 0;
                for (int b = 0; b < B; ++ b) {
                    ${input_specifier} curr_val = input_data[(a * B + b) * C + c];
                    if(${gen_comp}){
                      best_val = curr_val;
                      best_arg = b;
                    }
                }
                
                output_data[a * C + c] = best_arg;
            }
        }
        return TinyNN_SUCCESS;
    })");
    // clang-format on
    return writer.str();
}
// vim: syntax=cpp.doxygen
