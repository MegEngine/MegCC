#pragma once
#include <string>
#include "Utils/StringTemplate.h"
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {

class RelayoutHelper {
public:
    static std::string GetLayoutHelper() {
        std::string body = R"(
        typedef struct{
            int inc_dims[MAX_DIM];
            int reset_stride[MAX_DIM];
            int offset;
        } NoconIter;        

        static inline NoconIter init_iter(const Layout layout){
            NoconIter iter;
            for (int i = layout.nr_dim - 1; i >= 0; --i){
                iter.inc_dims[i] = 0;
                iter.reset_stride[i] = (layout.dims[i] - 1) * layout.stride[i];
            }
            iter.offset = 0;
            return iter;
        }

        static inline void inc_iter(const Layout layout, NoconIter* iter, int base_axis){
            for (int i = base_axis; i >= 0; --i){
                iter->inc_dims[i] += 1;
                if (iter->inc_dims[i] >= layout.dims[i]){
                    iter->inc_dims[i] = 0;
                    iter->offset = iter->offset - iter->reset_stride[i];
                }else{
                    iter->offset = iter->offset + layout.stride[i];
                    break;
                }
            }
        }
        //! only for collapse layout
        static inline bool is_contiguous(Layout layout){
            return layout.nr_dim == 1 && layout.stride[0] == 1;
        }

        static inline Layout collapse_contiguous(Layout layout){
            if(layout.nr_dim == 0){
                return layout;
            }

            //! remove shape 1
            Layout temp;
            temp.nr_dim = 0;
            for (int i = 0; i < layout.nr_dim; ++i){
                if (layout.dims[i] == 1){
                    continue;
                }else{
                    temp.dims[temp.nr_dim] = layout.dims[i];
                    temp.stride[temp.nr_dim] = layout.stride[i];
                    ++temp.nr_dim;
                }
            }

            Layout res;
            res.nr_dim = 0;
            res.dims[0] = 1;
            res.stride[0] = temp.dims[0] * temp.stride[0];
            for (int i = 0; i < temp.nr_dim; ++i){
                if (res.stride[res.nr_dim] == temp.dims[i] * temp.stride[i]){
                    res.dims[res.nr_dim] *= temp.dims[i];
                    res.stride[res.nr_dim] = temp.stride[i];
                }else{
                    ++res.nr_dim;
                    res.dims[res.nr_dim] = temp.dims[i];
                    res.stride[res.nr_dim] = temp.stride[i];
                }
            }
            ++res.nr_dim;
            return res;
        }
    )";
        return body;
    }

    static std::string GetTransposeCall() {
        std::string temp = R"(
        {
            TransposeArg src_trans;
            TransposeArg dst_trans;
            src_trans.valid = false;
            dst_trans.valid = false;
            if (!src_contig){
                src_trans = transpose_check(src_layout, false);
            }
            if (!dst_contig){
                dst_trans = transpose_check(dst_layout, true);
            }
            if (src_trans.valid && dst_contig) {
                do_transpose(dst_data, src_data, src_trans);
                return TinyNN_SUCCESS;
            }else if (dst_trans.valid && src_contig){
                do_transpose(dst_data, src_data, dst_trans);
                return TinyNN_SUCCESS;
            }
        }
    )";
        return temp;
    }
    static std::string GetNonconMemcpyModule(const std::string& specifier) {
        std::string memcpy_nocontig_temp = R"(
        static inline bool copy_check(Layout layout){
            if (layout.nr_dim <= 3 && layout.stride[layout.nr_dim - 1] == 1){
                return true;
            }else{
                return false;
            }
        }
        static void reverse_memcpy(void* src, void* dst, size_t size){
            memcpy(dst, src, size);
        }
        static void postive_memcpy(void* dst, void* src, size_t size){
            memcpy(dst, src, size);
        }
        typedef void (*memcpy_policy_t)(void* cont, void* non_cont, size_t);
        static inline void memcpy_cont2nocont(${specifier}* dst, ${specifier}* src, Layout con_layout, Layout nocon_layout, memcpy_policy_t memcpy_call){     
            if(nocon_layout.nr_dim == 2){
                int batch = nocon_layout.dims[0];
                int batch_stride = nocon_layout.stride[0];
                int copy_elem = nocon_layout.dims[1];
                for(int i = 0; i < batch; ++i){
                    memcpy_call(dst, src, sizeof(${specifier}) * copy_elem);
                    dst += batch_stride;
                    src += copy_elem;
                }
            }else if(nocon_layout.nr_dim == 3){
                int batch = nocon_layout.dims[0];
                int batch_stride = nocon_layout.stride[0];
                int batch2 = nocon_layout.dims[1];
                int batch_stride2 = nocon_layout.stride[1];
                int copy_elem = nocon_layout.dims[2];
                for(int i = 0; i < batch; ++i){
                    ${specifier}* dst_s0 = dst;
                    for(int j = 0; j < batch2; ++j){
                        memcpy_call(dst_s0, src, sizeof(${specifier}) * copy_elem);
                        dst_s0 += batch_stride2;
                        src += copy_elem;
                    }
                    dst += batch_stride;
                }
            }else{
                TINYNN_ASSERT_MSG(0, "bug memcpy %d", nocon_layout.nr_dim);
            }
        }
    )";
        return StringTemplate::StringTemplateArgs()
                .add("specifier", specifier)
                .render(memcpy_nocontig_temp);
    }
    static std::string GetTransposeModule(
            const std::string& specifier, size_t type_size,
            const std::string fast_transpose_impl = "") {
        std::stringstream ss;
        ss << R"(
            typedef struct{
                bool valid;
                int batch;
                int batch_stride;
                int height;
                int width;
                int dst_batch_stride;
                int dst_step;
                int src_step;
            } TransposeArg;

            //! default layout is from src, dst is contig
            static inline  TransposeArg transpose_check(Layout layout, bool is_dst){
                TransposeArg res;
                res.valid = false;
                if(layout.nr_dim == 3 && layout.stride[0] == layout.stride[2] * layout.dims[2] && layout.stride[1] == 1 && layout.stride[2] >= layout.dims[1]){
                    if(is_dst){
                        res.batch = layout.dims[0];
                        res.height = layout.dims[1];
                        res.width = layout.dims[2];
                        res.batch_stride = layout.stride[0];
                        res.dst_batch_stride = res.height * res.width;
                        res.dst_step = layout.stride[2];
                        res.src_step = res.width;
                    }else{
                        res.batch = layout.dims[0];
                        res.height = layout.dims[2];
                        res.width = layout.dims[1];
                        res.batch_stride = layout.stride[0];
                        res.dst_batch_stride = res.height * res.width;
                        res.dst_step = res.height;
                        res.src_step = layout.stride[2];
                    }
                    res.valid = true;
                }else if(layout.nr_dim == 2 && layout.stride[0] == 1 && layout.stride[1] >= layout.dims[0]){
                    if(is_dst){
                        res.height = layout.dims[0];
                        res.width = layout.dims[1];
                        res.dst_step = layout.stride[1];
                        res.src_step = res.width;
                    }else{
                        res.height = layout.dims[1];
                        res.width = layout.dims[0];
                        res.dst_step = res.height;
                        res.src_step = layout.stride[1];
                    }
                    res.batch = 1;
                    res.batch_stride = 0;
                    res.dst_batch_stride = 0;
                    res.valid = true;
                }else if(layout.nr_dim == 3 && layout.stride[0] == 1 && layout.stride[1] == layout.dims[0] && layout.stride[2] == layout.dims[1] * layout.stride[1]){
                    if(is_dst){
                        res.batch = layout.dims[1];
                        res.height = layout.dims[0];
                        res.width = layout.dims[2];
                        res.batch_stride = res.width;
                        res.dst_batch_stride = layout.stride[1];
                        res.dst_step = layout.stride[2];
                        res.src_step = res.batch * res.width;
                    }else{
                        res.batch = layout.dims[1];
                        res.height = layout.dims[2];
                        res.width = layout.dims[0];
                        res.batch_stride = layout.stride[1];
                        res.dst_batch_stride = res.height;
                        res.dst_step = res.batch * res.height;
                        res.src_step = layout.stride[2];
                    }
                    res.valid = true;
                }
                return res;
            }
            
        )";
        ss << StringTemplate::StringTemplateArgs()
                        .add("specifier", specifier)
                        .add("gen_fast_impl", fast_transpose_impl)
                        .add("gen_fast_call",
                             [&]() -> std::string {
                                 if (fast_transpose_impl.size() == 0) {
                                     return R"(
                                         for (int h_idx = 0; h_idx < trans_arg.height; ++h_idx)
                                            for (int w_idx = 0; w_idx < trans_arg.width; ++w_idx) {
                                                int src_idx = h_idx * trans_arg.src_step + w_idx;
                                                int dst_idx = w_idx * trans_arg.dst_step + h_idx;                    
                                                dst_ptr[dst_idx] = src_ptr[src_idx];
                                            }
                                     )";
                                 } else {
                                     return "fast_transpose_impl_" +
                                            std::to_string(type_size * 8) +
                                            "(src_ptr, dst_ptr, "
                                            "trans_arg.width, "
                                            "trans_arg.height, 1, "
                                            "trans_arg.src_step, "
                                            "trans_arg.dst_step);";
                                 }
                             })
                        .render(
                                R"(
                ${gen_fast_impl}
                static inline void do_transpose(${specifier}* dst, ${specifier}* src, TransposeArg trans_arg){
                    int src_batch_stride = trans_arg.batch_stride;
                    int dst_batch_stride = trans_arg.dst_batch_stride;
                    for (int batch_idx = 0; batch_idx < trans_arg.batch; ++batch_idx) {
                        ${specifier}* src_ptr = src + batch_idx * src_batch_stride;
                        ${specifier}* dst_ptr = dst + batch_idx * dst_batch_stride;
                        ${gen_fast_call()}
                    }
                }
            )");
        return ss.str();
    }
};

}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
