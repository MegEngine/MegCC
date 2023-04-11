#include "Concat.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

bool ConcatKernel::IsAvailable(TContext* context) const {
    return true;
}
//! kernel gen
std::string ConcatKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    auto dtype_size = Utils::get_dtype_size(context->getAttrOprand("operand:0").dtype);

    ss << "kernel_concat_byte_" << dtype_size << "_axis_"
       << context->getAttrInt("axis");
    return ss.str();
}

std::string ConcatKernel::GetKernelBody(TContext* context) const {
    std::stringstream writer;
    int axis = context->getAttrInt("axis");
    int dtype_size = Utils::get_dtype_size(context->getAttrOprand("operand:0").dtype);
    std::string ctype = Utils::get_common_dtype_specifier(dtype_size);
    writer << R"(
      #include "tensor_util.h"
    )";
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context);
    // clang-format off
    auto temp_body = R"({
    Tensor* output = outputs[0];
    int axis = ${axis};
    uint32_t dtype_len = ${dtype_size};
    int32_t axis_index = 0;
    for (int i = 0; i < nr_input; i++) {
        Tensor slice = *output;
        slice.layout.dims[axis]= inputs[i]->layout.dims[axis];
        uint32_t offset = axis_index * output->layout.stride[axis];
        slice.ptr += offset * dtype_len;
        axis_index += inputs[i]->layout.dims[axis];
        size_t nr_elem = 1;
        for (int j = 0; j < inputs[i]->layout.nr_dim; ++j) {
            nr_elem *= inputs[i]->layout.dims[j];
        }
        if (is_contiguous(slice.layout)) {
            memcpy(slice.ptr, inputs[i]->ptr, nr_elem * dtype_len);
        } else {
            NoconIter src_iter = init_iter(inputs[i]->layout);
            NoconIter dst_iter = init_iter(slice.layout);
    
            ${ctype}* dst_data = slice.ptr;                               
            ${ctype}* src_data = inputs[i]->ptr;                          
            for (size_t j = 0; j < nr_elem; ++j) {                     
                dst_data[dst_iter.offset] = src_data[src_iter.offset]; 
                inc_iter(inputs[i]->layout, &src_iter);                
                inc_iter(slice.layout, &dst_iter);                     
            }
        }
    }

    return TinyNN_SUCCESS;

    })";
    // clang-format on
    writer << StringTemplate::StringTemplateArgs()
                      .add("axis", axis)
                      .add("ctype", ctype)
                      .add("dtype_size", dtype_size)
                      .render(temp_body);
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen
