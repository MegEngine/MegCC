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
    writer << R"(
        #include "tensor_util.h"
    )";
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
    const Tensor* idx_tensors[7];
    int axis_vec[7];
    ${axis_init_str}

    for (int i = 0; i < nr_index; ++i) {
        idx_tensors[i] = inputs[i + 1];
    }
    // compute idx_axis start
    size_t idx_axis = 0;
    {
        int contig_idx = 1;
        for (size_t i = 1; i < nr_index; ++i) {
            if (axis_vec[i] != axis_vec[i - 1] + 1) {
                contig_idx = 0;
                break;
            }
        }
        if (contig_idx) {
            idx_axis = axis_vec[0];
        }
    }
    // compute idx_axis end

    // compute nonidx axes start
    size_t nonidx_axes[7], nr_nonidx_axes = 0;
    {
        size_t idx = 0;
        for (size_t i = 0; i < src_layout.nr_dim; ++i) {
            if (idx != nr_index && i == axis_vec[idx]) {
                ++idx;
            } else {
                nonidx_axes[nr_nonidx_axes++] = i;
            }
        }
    }
    // compute nonidx axes end

    // deduce shape index_shape start
    Layout index_shape, index_shapes[7];
    index_shape.nr_dim = 0;
    for (int i = 0; i < nr_index; ++i) {
        index_shapes[i] = idx_tensors[i]->layout;
        TINYNN_ASSERT_MSG(index_shapes[i].nr_dim, 
                "bad input shape for polyadic operator");
        if (!index_shape.nr_dim || is_layout_scalar(&index_shape))
            index_shape = index_shapes[i];
        else if (!is_layout_scalar(&index_shapes[i])) {
            int max_dim = index_shape.nr_dim > index_shapes[i].nr_dim ? 
                            index_shape.nr_dim : index_shapes[i].nr_dim;
            for (int j = 0; j < max_dim; ++j) {
                int cur_idx = index_shapes[i].nr_dim - j - 1;
                int dst_idx = index_shape.nr_dim - j - 1;
                if (cur_idx >= 0 && dst_idx >= 0) {
                    size_t v0 = index_shape.dims[dst_idx], v1 = index_shapes[i].dims[cur_idx];
                    if (v0 != v1) {
                        TINYNN_ASSERT_MSG(v0 <= 1 || v1 <= 1, 
                                "bad input shape for polyadic operator");
                    }
                    int final_idx = cur_idx > dst_idx ? cur_idx : dst_idx;
                    index_shape.dims[final_idx] = (v0 != 0 && v1 != 0) ? (v0>v1 ? v0:v1) : 0;
                } else {
                    if (dst_idx < 0) {
                        index_shape.dims[cur_idx] = index_shapes[i].dims[cur_idx];
                    }
                }
            }
            index_shape.nr_dim = max_dim;
        }
    }
    // deduce shape index_shape end

    // broadcast index layout start
    Layout idx_layout[7];
    for (int i = 0; i < nr_index; ++i) {
        idx_layout[i] = idx_tensors[i]->layout;
        broadcast_layout(&idx_layout[i], index_shape);
    }
    // broadcast index layout end

    int dst_size = 1;
    for(int i = 0; i < dst_layout.nr_dim; ++i) {
        dst_size *= dst_tensor->layout.dims[i];
    }
    // compute core start
    NoconIter dst_iter = init_iter(dst_layout);
    for (size_t _ = 0; _ < dst_size; ++_, inc_iter(dst_layout, &dst_iter)) {
        int offset = 0;
        int* index_idx = dst_iter.inc_dims + idx_axis;
        for (size_t i = 0; i < nr_index; ++i) {
            int axis = axis_vec[i],
                src_shape = src_layout.dims[axis],
                src_stride = src_layout.stride[axis];
            int index_offset = 0;
            for (size_t j = 0; j < idx_layout[i].nr_dim; ++j) {
                index_offset += index_idx[j] * idx_layout[i].stride[j];
            }
            int* index_ptr = (int*)idx_tensors[i]->ptr;
            int src_idx = index_ptr[index_offset];
            src_idx += src_idx < 0 ? src_shape : 0;
            TINYNN_ASSERT_MSG(src_idx >= 0 && src_idx < src_shape, 
                "invalid advanced indexing");
            offset += src_idx * src_stride;
        }
        for (size_t i = 0; i < nr_nonidx_axes; ++i) {
            int stride = src_layout.stride[nonidx_axes[i]];
            int idx = dst_iter.inc_dims[i + (i >= idx_axis) * index_shape.nr_dim];
            offset += stride * idx;
        }
        dst[dst_iter.offset] = src[offset];
    }
    // compute core end
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