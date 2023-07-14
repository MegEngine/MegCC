#include "IndexingMultiAxisVec.h"
#include "Fp16Common.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"

namespace megcc {
namespace KernelGen {
namespace BareMetal {

bool IndexingMultiAxisKernel::IsAvailable(TContext* context) const {
    int nr_operand = context->getAttrInt("nr_operands");
    bool ok_operand = nr_operand > 2;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    std::string src_specifier =
            Utils::cvt_dtype_specifier(SymbolHelper::gen_valid_dtype(src_dtype));
    std::string dst_specifier = Utils::cvt_dtype_specifier(
            SymbolHelper::gen_valid_dtype(Utils::get_last_operand(context).dtype));
    bool ok_dtype = (src_specifier == dst_specifier);
    for (int i = 1; i < nr_operand - 1; ++i) {
        ok_operand =
                ok_operand &&
                context->getAttrOprand("operand:" + std::to_string(i)).shape.size() ==
                        1;
    }
    bool ok_axis = true;
    int last_axis = -1;
    for (int i = 0; i < nr_operand - 2; ++i) {
        int axis = context->getAttrInt("axis:" + std::to_string(i));
        if (last_axis >= 0) {
            ok_axis = ok_axis && (axis == last_axis + 1);
        }
        last_axis = axis;
    }
    return ok_dtype && ok_operand && ok_axis;
}
//! kernel gen
std::string IndexingMultiAxisKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_indexingmultiaxisvex";
    int nr_operand = context->getAttrInt("nr_operands");
    for (int i = 0; i < nr_operand - 2; ++i) {
        ss << "_" << context->getAttrInt("axis:" + std::to_string(i));
    }
    ss << "_" << context->getAttrOprand("operand:0").dtype;
    return ss.str();
}

std::string IndexingMultiAxisKernel::GetKernelBody(TContext* context) const {
    std::stringstream axis_init_ss;
    int nr_operand = context->getAttrInt("nr_operands");
    for (int i = 0; i < nr_operand - 2; ++i) {
        axis_init_ss << "axis_vec[" << i
                     << "] = " << context->getAttrInt("axis:" + std::to_string(i))
                     << ";\n";
    }
    std::string dtype_specifier = Utils::cvt_dtype_specifier(
            SymbolHelper::gen_valid_dtype(Utils::get_last_operand(context).dtype));
    std::stringstream writer;
    writer << "#include <string.h>\n";
    if (dtype_specifier == "gi_float16_t")
        writer << gen_fp16_define();
    writer << GenCommonRet() << " ";
    writer << GetKernelSignature(context) << "{\n";
    // clang-format off
    writer << StringTemplate::StringTemplateArgs(context).add("axis_init_str",axis_init_ss.str()).add("dtype_specifier", dtype_specifier).render(
    R"(
    ${dtype_specifier}* src = (${dtype_specifier}*)inputs[0]->ptr;
    ${dtype_specifier}* dst = (${dtype_specifier}*)outputs[0]->ptr;

    const Tensor* src_tensor = inputs[0];
    const Tensor* dst_tensor = outputs[0];
    const Layout src_layout = src_tensor->layout;
    const Layout dst_layout = dst_tensor->layout;

    int nr_index = nr_input - 1;
    const Tensor* idx_tensors[5];
    int axis_vec[5];
    ${axis_init_str}
    for(int i = 0; i < nr_index; ++i) {
        idx_tensors[i] = inputs[i + 1];
    }
    int batch = 1;
    for(int i = 0; i <= axis_vec[0] - 1; ++i) {
        batch = batch * src_layout.dims[i];
    }
    int first_last_index = axis_vec[0];
    int last_index_axis = axis_vec[nr_index - 1];
    int batch_stride_src = first_last_index - 1 >= 0? src_layout.stride[first_last_index - 1]:0;
    int batch_stride_dst = first_last_index - 1 >= 0? dst_layout.stride[first_last_index - 1]:0;
    int last_stride = src_layout.stride[last_index_axis];
    for(int batch_idx = 0; batch_idx < batch; ++batch_idx) {
        ${dtype_specifier}* src_ptr = src + batch_idx * batch_stride_src;
        ${dtype_specifier}* dst_ptr = dst + batch_idx * batch_stride_dst;
        for(int i = 0; i < idx_tensors[0]->layout.dims[0]; ++i) {
            int src_offet = 0;
            int shape_prod = 1;
            for(int axis_id = nr_index - 1; axis_id >= 0; --axis_id) {
                int* index_ptr = (int*)idx_tensors[axis_id]->ptr;
                src_offet += index_ptr[i] * shape_prod;
                shape_prod *= src_layout.dims[axis_vec[axis_id]];
            }
            memcpy(dst_ptr + i * last_stride, src_ptr + src_offet * last_stride, last_stride * sizeof(${dtype_specifier}));
        }
    }

    return TinyNN_SUCCESS;
    })"
    );
    // clang-format on
    return writer.str();
}

}  // namespace BareMetal
}  // namespace KernelGen
}  // namespace megcc
   // vim: syntax=cpp.doxygen