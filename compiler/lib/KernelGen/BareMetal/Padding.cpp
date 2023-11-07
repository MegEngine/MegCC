#include <sstream>

#include "Fp16Common.h"
#include "Padding.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace BareMetal;

bool PaddingKernel::IsAvailable(TContext* context) const {
    std::string padding_mode = context->getAttrStr("padding_mode");
    bool mode_ok =
            (padding_mode == "REPLICATE" || padding_mode == "CONSTANT" ||
             padding_mode == "REFLECT");
    return mode_ok;
}

//! kernel gen
std::string PaddingKernel::GetKernelSymbol(TContext* context) const {
    std::stringstream ss;
    ss << "kernel_padding_front_offset_";
    for (int i = 0; i < 7; ++i) {
        ss << context->getAttrInt("front_offsets:" + std::to_string(i)) << "_";
    }
    ss << context->getAttrStr("padding_mode") << "_"
       << context->getAttrFloat("padding_val") << "_"
       << context->getAttrOprand("operand:0").dtype;
    return ss.str();
}

namespace {
std::string gen_replicate_padding(TContext* context, std::string* func_name) {
    *func_name = "replicate_padding";
    std::string func = R"(
        static void ${func_name}(
                const size_t ndim, const size_t total_out_nr, const ${dtype}* const src, ${dtype}* const dst,
                const int* front_offsets, const uint32_t* dst_shape, const int* dst_stride, const uint32_t* src_shape, const int* src_stride) {
            uint32_t **idx_tbl = (uint32_t**)tinynn_malloc(sizeof(uint32_t*) * ndim);
            for (size_t i = 0; i < ndim; ++i) {
                idx_tbl[i] = (uint32_t*)tinynn_malloc(sizeof(uint32_t) * dst_shape[i]);
                for (uint32_t idx = 0; idx < dst_shape[i]; ++idx) {
                    if (idx < front_offsets[i]) {
                        idx_tbl[i][idx] = 0;
                    } else if (idx >= front_offsets[i] + src_shape[i]) {
                        idx_tbl[i][idx] = src_shape[i] - 1;
                    } else {
                        idx_tbl[i][idx] = idx - front_offsets[i];
                    }
                }
            }
            
            for(size_t out_index = 0; out_index < total_out_nr; ++out_index) {
                size_t in_index = 0;
                size_t out_index_tmp = out_index;
                for (size_t dim = 0; dim <= ndim - 1; ++dim) {
                    size_t dim_index = out_index_tmp / dst_stride[dim];
                    out_index_tmp -= dim_index * dst_stride[dim];
                    in_index += idx_tbl[dim][dim_index] * src_stride[dim];
                }
                dst[out_index] = src[in_index];
            }

            for (size_t i = 0; i < ndim; ++i) {
                tinynn_free(idx_tbl[i]);
            }
            tinynn_free(idx_tbl);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("func_name", *func_name)
            .add("dtype",
                 Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype))
            .render(func);
}

std::string gen_constant_padding(TContext* context, std::string* func_name) {
    *func_name = "constant_padding";
    std::string func = R"(
        static void ${func_name}(
                const size_t ndim, const size_t total_out_nr, const ${dtype}* const src, ${dtype}* const dst,
                const int* front_offsets, const uint32_t* dst_shape, const int* dst_stride, const uint32_t* src_shape, const int* src_stride) {
            uint8_t **is_valid = (uint8_t**)tinynn_malloc(sizeof(uint8_t*) * ndim);
            for (size_t i = 0; i < ndim; ++i) {
                is_valid[i] = (uint8_t*)tinynn_malloc(sizeof(uint8_t) * dst_shape[i]);
                for (uint32_t idx = 0; idx < dst_shape[i]; ++idx) {
                    if (idx < front_offsets[i] || idx >= front_offsets[i] + src_shape[i]) {
                        is_valid[i][idx] = 0;
                    } else {
                        is_valid[i][idx] = 1;
                    }
                }
            }
            
            for(size_t out_index = 0; out_index < total_out_nr; ++out_index) {
                int in_src_valid_area = 1;
                size_t in_index = 0;
                size_t out_index_tmp = out_index;
                for (size_t dim = 0; dim <= ndim - 1; ++dim) {
                    size_t dim_index = out_index_tmp / dst_stride[dim];
                    out_index_tmp -= dim_index * dst_stride[dim];
                    if (!is_valid[dim][dim_index]) {
                        in_src_valid_area = 0;
                        break;
                    }
                    in_index += (dim_index - front_offsets[dim]) * src_stride[dim];
                }
                if (in_src_valid_area) {
                    dst[out_index] = src[in_index];
                } else {
                    dst[out_index] = (${dtype})${padding_val};
                }
            }

            for (size_t i = 0; i < ndim; ++i) {
                tinynn_free(is_valid[i]);
            }
            tinynn_free(is_valid);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("func_name", *func_name)
            .add("padding_val", std::to_string(context->getAttrFloat("padding_val")))
            .add("dtype",
                 Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype))
            .render(func);
}

std::string gen_reflect_padding(TContext* context, std::string* func_name) {
    *func_name = "reflect_padding";
    std::string func = R"(
        static void ${func_name}(
                const size_t ndim, const size_t total_out_nr, const ${dtype}* const src, ${dtype}* const dst,
                const int* front_offsets, const uint32_t* dst_shape, const int* dst_stride, const uint32_t* src_shape, const int* src_stride) {
            uint32_t **idx_tbl = (uint32_t**)tinynn_malloc(sizeof(uint32_t*) * ndim);
            for (size_t i = 0; i < ndim; ++i) {
                idx_tbl[i] = (uint32_t*)tinynn_malloc(sizeof(uint32_t) * dst_shape[i]);
                for (uint32_t idx = 0; idx < dst_shape[i]; ++idx) {
                    if (idx < front_offsets[i]) {
                        idx_tbl[i][idx] = front_offsets[i] - idx;
                    } else if (idx >= front_offsets[i] + src_shape[i]) {
                        idx_tbl[i][idx] = src_shape[i] * 2 - 2 - (idx - front_offsets[i]); //! (src_shape[i] - 1) - (idx - front_offsets[i] - (src_shape[i] - 1))
                    } else {
                        idx_tbl[i][idx] = idx - front_offsets[i];
                    }
                }
            }
            for(size_t out_index = 0; out_index < total_out_nr; ++out_index) {
                size_t in_index = 0;
                size_t out_index_tmp = out_index;
                for (size_t dim = 0; dim <= ndim - 1; ++dim) {
                    long long dim_index = out_index_tmp / dst_stride[dim];
                    out_index_tmp -= dim_index * dst_stride[dim];
                    in_index += idx_tbl[dim][dim_index] * (size_t)src_stride[dim];
                }
                dst[out_index] = src[in_index];
            }

            for (size_t i = 0; i < ndim; ++i) {
                tinynn_free(idx_tbl[i]);
            }
            tinynn_free(idx_tbl);
        }
    )";
    return StringTemplate::StringTemplateArgs()
            .add("func_name", *func_name)
            .add("dtype",
                 Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype))
            .render(func);
}
}  // namespace

std::string PaddingKernel::GetKernelBody(TContext* context) const {
    std::stringstream ss;
    ss << "#include \"utils.h\"\n";
    std::string dtype =
            Utils::cvt_dtype_specifier(context->getAttrOprand("operand:0").dtype);
    if (dtype == "gi_float16_t") {
        ss << gen_fp16_define();
    }
    std::string func_name;
    std::string padding_mode = context->getAttrStr("padding_mode");
    if (padding_mode == "REPLICATE") {
        ss << gen_replicate_padding(context, &func_name);
    } else if (padding_mode == "CONSTANT") {
        ss << gen_constant_padding(context, &func_name);
    } else {
        CC_ASSERT(padding_mode == "REFLECT");
        ss << gen_reflect_padding(context, &func_name);
    }
    ss << GenCommonRet() << " " << GetKernelSignature(context);
    std::string body_temp = R"({
    ${dtype}* a_data = (${dtype}*)inputs[0]->ptr;
    ${dtype}* c_data = (${dtype}*)outputs[0]->ptr;
    TINYNN_ASSERT(a_data);
    TINYNN_ASSERT(c_data);
    const Tensor* a_tensor = inputs[0];
    const Layout a_layout = a_tensor->layout;
    const Tensor* c_tensor = outputs[0];
    const Layout c_layout = c_tensor->layout;
    size_t nr_elem = 1;
    for (int i = 0; i < c_layout.nr_dim; ++i) {
        nr_elem *= c_layout.dims[i];
    }
#define MAX_NDIM 7
    int front_offsets[MAX_NDIM];
#undef MAX_NDIM
    front_offsets[0] = ${front_offset0};
    front_offsets[1] = ${front_offset1};
    front_offsets[2] = ${front_offset2};
    front_offsets[3] = ${front_offset3};
    front_offsets[4] = ${front_offset4};
    front_offsets[5] = ${front_offset5};
    front_offsets[6] = ${front_offset6};

    ${func_name}(a_layout.nr_dim, nr_elem, a_data, c_data, front_offsets, c_layout.dims, c_layout.stride, a_layout.dims, a_layout.stride);
    
    return TinyNN_SUCCESS;
})";

    ss << StringTemplate::StringTemplateArgs()
                    .add("dtype", dtype)
                    .add("func_name", func_name)
                    .add("front_offset0", context->getAttrInt("front_offsets:0"))
                    .add("front_offset1", context->getAttrInt("front_offsets:1"))
                    .add("front_offset2", context->getAttrInt("front_offsets:2"))
                    .add("front_offset3", context->getAttrInt("front_offsets:3"))
                    .add("front_offset4", context->getAttrInt("front_offsets:4"))
                    .add("front_offset5", context->getAttrInt("front_offsets:5"))
                    .add("front_offset6", context->getAttrInt("front_offsets:6"))
                    .render(body_temp);
    return ss.str();
}

// vim: syntax=cpp.doxygen