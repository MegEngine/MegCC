#include "Im2colCommon.h"
#include <memory>
#include "Common/ConvKernel.h"
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/GISimdHelper.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
std::string nchw_im2col_kern = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    
    static inline void img2col(const float* __restrict src, float* __restrict dst,
                    const int OW, const int IC,
                    const int IH, const int IW, const int FH, const int FW,
                    const int SH, const int SW, const int cur_index,
                    const int block_size) {                                
        int start_h = cur_index / OW;
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }

        size_t i = 0;
        if (same_line) {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            dst[i++] =
                                    src[ic * IH * IW + (start_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }
                    }
                }
            }
        } else {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {

                        for (int w = cur_remain_w; w < OW; w++) {
                            dst[i++] =
                                    src[ic * IH * IW + (start_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                dst[i++] = src[ic * IH * IW + (h * SH + fh) * IW +
                                            (ow * SW + fw)];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (end_h * SH + fh) * IW +
                                        (w * SW + fw)];
                        }
                    }
                }
            }
        }
    }
)";

std::string nchw_im2col_s1_kern = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    static inline void img2col(const float* __restrict src, float* __restrict dst, 
             const int OW, const int IC, const int IH,
             const int IW, const int FH, const int FW, const int SH, const int SW, const int cur_index,
             const int block_size) {
        int start_h = cur_index / OW;
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool is_xcorr = true;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }
        int i = 0;
        if (same_line) {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {

                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (start_h + fh) * IW +
                                        (w + fw)];
                        }
                    }
                }
            }
        } else {
            rep(ic, IC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        
                        for (int w = cur_remain_w; w < OW; w++) {
                            dst[i++] = src[ic * IH * IW + (start_h + fh) * IW +
                                        (w + fw)];
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                dst[i++] = src[ic * IH * IW + (h + fh) * IW +
                                            (ow + fw)];
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            dst[i++] = src[ic * IH * IW + (end_h + fh) * IW +
                                        (w + fw)];
                        }
                    }
                }
            }
        }
    }
)";

std::string nchw_pad_src_kern = R"(
    static inline void pad_src(float* inptr, float* outptr, int ic, int ih, int iw, int pad_h, int pad_w){
        size_t out_idx = 0;
        int paded_iw = iw + 2 * pad_w;
        int nr_pad_top_bottom = pad_h * paded_iw;

        for(int ic_idx = 0; ic_idx < ic; ic_idx++){
            memset(outptr, 0, sizeof(float) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            
            for (int ih_idx = 0; ih_idx < ih; ++ih_idx){
                for(int i = 0; i < pad_w; ++i){
                    *outptr++ = 0.f;
                }
                memcpy(outptr, inptr + ih_idx * iw, sizeof(float) * iw);
                outptr += iw;
                for(int i = 0; i < pad_w; ++i){
                    *outptr++ = 0.f;
                }
            }
            memset(outptr, 0, sizeof(float) * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            inptr += ih * iw;
        }
    }
)";

int get_pack_c_size(TContext* ctx) {
    int pack_c_size = 1;
    auto fmt = ctx->getAttrStr("format");
    if (fmt == "NCHW44") {
        pack_c_size = 4;
    } else if (fmt == "NCHW88") {
        pack_c_size = 8;
    } else {
        CC_ASSERT(fmt == "NCHW");
    }
    return pack_c_size;
}
std::string gen_nchwxx_im2col_kern(TContext* ctx) {
    auto format = ctx->getAttrStr("format");
    auto pack_c_size = get_pack_c_size(ctx);
    if (format == "NCHW") {
        auto sh = ctx->getAttrInt("stride_h");
        auto sw = ctx->getAttrInt("stride_w");
        if (sh == sw && sw == 1) {
            return nchw_im2col_s1_kern;
        } else {
            return nchw_im2col_kern;
        }
    } else {
        auto dtype = ctx->getAttrOprand("operand:0").dtype;
        GISimdHelper simd_helper(dtype);

        static std::string nchwxx_im2col_temp = R"(
    #define rep(i, n) for (int i = 0; i < (n); ++i)
    static inline void img2col(const ${specifier}* __restrict src, ${specifier}* __restrict dst,
                        const int OW, const int IC,
                        const int IH, const int IW, const int FH, const int FW,
                        const int SH, const int SW, const int cur_index,
                        const int block_size) {
        int start_h = cur_index / OW;
         const int pack_c_size = ${PACK_C_SIZE};
        int cur_remain_w = cur_index % OW;
        int end_h = (cur_index + block_size) / OW;
        int end_remain_w = (cur_index + block_size) % OW;
        bool same_line = false;
        if (start_h == end_h) {
            same_line = true;
        }

        size_t newIC = IC /pack_c_size;
        size_t i = 0;
        
        if (same_line) {
            rep(ic, newIC) {
                rep(fh, FH) {
                    rep(fw, FW) {
                        for (int w = cur_remain_w; w < end_remain_w; w++) {
                            size_t index =pack_c_size * (ic * IH * IW +
                                                (start_h * SH + fh) * IW +
                                                (w * SW + fw));
                            ${simd_specifier} vec = ${simd_ld1q}(src+index);
                            ${simd_st1q}(dst+i, vec);
                            i+=pack_c_size;
                        }
                    }
                }
            }
        } else {
            rep(ic, newIC) {
                rep(fh, FH) {
                    rep(fw, FW) {                      
                        for (int w = cur_remain_w; w < OW; w++) {
                            size_t index =pack_c_size * (ic * IH * IW +
                                        (start_h * SH + fh) * IW +
                                        (w * SW + fw));
                            ${simd_specifier} vec = ${simd_ld1q}(src+index);
                            ${simd_st1q}(dst+i, vec);
                            i+=pack_c_size;
                        }

                        for (int h = start_h + 1; h < end_h; h++) {
                            rep(ow, OW) {
                                size_t index =pack_c_size * (ic * IH * IW +
                                                    (h * SH + fh) * IW +
                                                    (ow * SW + fw));
                                ${simd_specifier} vec = ${simd_ld1q}(src+index);
                                ${simd_st1q}(dst+i, vec);
                                i+=pack_c_size;
                            }
                        }

                        for (int w = 0; w < end_remain_w; w++) {
                            size_t index =pack_c_size * (ic * IH * IW +
                                                (end_h * SH + fh) * IW +
                                                (w * SW + fw));
                            ${simd_specifier} vec = ${simd_ld1q}(src+index);
                            ${simd_st1q}(dst+i, vec);
                            i+=pack_c_size;
                        }
                    }
                }
            }
        }
        
    }
)";
        return StringTemplate::StringTemplateArgs()
                .add("specifier", Utils::cvt_dtype_specifier(dtype))
                .add("simd_specifier", simd_helper.get_specifier_q_symbol())
                .add("simd_ld1q", simd_helper.get_ld1q_symbol())
                .add("simd_st1q", simd_helper.get_st1q_symbol())
                .add("PACK_C_SIZE", pack_c_size)
                .render(nchwxx_im2col_temp);
    }
}

std::string gen_nchwxx_pad_src_kern(TContext* ctx) {
    auto format = ctx->getAttrStr("format");
    auto pack_c_size = get_pack_c_size(ctx);
    if (format == "NCHW") {
        return nchw_pad_src_kern;

    } else {
        auto dtype = ctx->getAttrOprand("operand:0").dtype;
        GISimdHelper simd_helper(dtype);
        uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
        static std::string nchwxx_pad_src_temp = R"(
    static inline void pad_src(${specifier}* inptr, ${specifier}* outptr, int ic, int ih, int iw, int pad_h, int pad_w){
        const int pack_c_size = ${PACK_C_SIZE};
        const int paded_iw = iw + 2 * pad_w;
        const int nr_pad_top_bottom = pad_h * paded_iw * pack_c_size;
        const ${simd_specifier} vzero = ${simd_dup1q}(0);
        for(int ic_idx = 0; ic_idx < ic; ic_idx += pack_c_size){
            memset(outptr, 0, ${dtype_specifier_size} * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            
            for (int ih_idx = 0; ih_idx < ih; ++ih_idx){
                for(int i = 0; i < pad_w; ++i){
                    ${simd_st1q}(outptr, vzero);
                    outptr += pack_c_size;
                }
                memcpy(outptr, inptr + ih_idx * iw * pack_c_size, ${dtype_specifier_size} * iw * pack_c_size);
                outptr += iw * pack_c_size;
                for(int i = 0; i < pad_w; ++i){
                    ${simd_st1q}(outptr, vzero);
                    outptr += pack_c_size;
                }
            }
            memset(outptr, 0, ${dtype_specifier_size} * nr_pad_top_bottom);
            outptr += nr_pad_top_bottom;
            inptr += ih * iw * pack_c_size;
        }
    }
)";
        return StringTemplate::StringTemplateArgs()
                .add("specifier", Utils::cvt_dtype_specifier(dtype))
                .add("simd_specifier", simd_helper.get_specifier_q_symbol())
                .add("simd_dup1q", simd_helper.get_dupq_n_symbol())
                .add("simd_st1q", simd_helper.get_st1q_symbol())
                .add("PACK_C_SIZE", pack_c_size)
                .add("dtype_specifier_size", dtype_specifier_size)
                .render(nchwxx_pad_src_temp);
    }
}

int get_group_weight_ndim(TContext* ctx) {
    int group_weight_ndim = 5;
    auto fmt = ctx->getAttrStr("format");
    if (fmt == "NCHW44" || fmt == "NCHW88") {
        group_weight_ndim = 7;
    } else {
        CC_ASSERT(fmt == "NCHW");
    }
    return group_weight_ndim;
}

std::string workspace_template(TContext* ctx, Im2colStrategyBase* strategy) {
    std::stringstream ss;
    auto inner_ctx = strategy->cvt2matmul_ctx(ctx);
    uint32_t dtype_specifier_size =
            Utils::get_dtype_size(inner_ctx->getAttrStr("dtype"));
    const int group_weight_dim = get_group_weight_ndim(ctx);
    auto pack_c_size = get_pack_c_size(ctx);
    std::string workspace_temp =
            R"({
        TINYNN_ASSERT(workspace);
        const int pack_c_size = ${pack_c_size};
        const Layout in_layout = inputs[0]->layout;
        const uint32_t ic = in_layout.dims[1] * pack_c_size;
        const uint32_t ih = in_layout.dims[2];
        const uint32_t iw = in_layout.dims[3];
        const Layout weight_layout = inputs[1]->layout;
        uint32_t group = 1;
        uint32_t fh = weight_layout.dims[2];
        uint32_t fw = weight_layout.dims[3];
        if (weight_layout.nr_dim == ${group_weight_dim}) {
            group = weight_layout.dims[0];
            fh = weight_layout.dims[3];
            fw = weight_layout.dims[4];
        }
        const uint32_t icpg = ic / group;
        
        const uint32_t k = fh * fw * icpg; 
        const uint32_t oh = (ih - fh + 2 * ${pad_h}) / ${stride_h} + 1;
        const uint32_t ow = (iw - fw + 2 * ${pad_w}) / ${stride_w} + 1;        
        const uint32_t ohw = oh * ow;
        
        const int preset_block_ohw = 192;
        const int block_ohw = preset_block_ohw > ohw ? ohw : preset_block_ohw;
        size_t pad_out = (size_t) icpg * (ih + 2 * ${pad_h}) * (iw + 2 * ${pad_w}) * ${dtype_specifier_size} + 64;
        size_t im2col_out = (size_t)block_ohw * k * ${dtype_specifier_size} + 64;                
        size_t packed_out = ${packb_workspace_func}(0, block_ohw, 0, k);
        *workspace = pad_out + im2col_out + packed_out;
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .add("pack_c_size", pack_c_size)
                    .add("group_weight_dim", group_weight_dim)
                    .add("dtype_specifier_size", dtype_specifier_size)
                    .add("packb_workspace_func",
                         strategy->GetPackBWorkspaceSym(inner_ctx.get()))
                    .render(workspace_temp);
    return ss.str();
}

std::string init_template(TContext* ctx, Im2colStrategyBase* strategy) {
    std::stringstream writer;
    auto inner_ctx = strategy->cvt2matmul_ctx(ctx);
    auto pack_c_size = get_pack_c_size(ctx);
    auto dtype = inner_ctx->getAttrStr("dtype");
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const uint32_t nr_out_weight = 1;
    const std::string group_str = is_group ? "in_weights->layout.dims[0]" : "1";
    const int ocpg_offset = is_group ? 1 : 0;
    const std::string common_def = StringTemplate::StringTemplateArgs()
                                           .add("group_str", group_str)
                                           .add("ocpg_offset", ocpg_offset)
                                           .add("pack_c_size", pack_c_size)
                                           .render(R"(
        const int pack_c_size = ${pack_c_size};
        Tensor* in_weights = inputs[1];
        const int group = ${group_str};
        const int ymax = in_weights->layout.dims[${ocpg_offset}] * pack_c_size;
        const int kmax = pack_c_size * in_weights->layout.dims[${ocpg_offset} + 1] * in_weights->layout.dims[${ocpg_offset} + 2] * in_weights->layout.dims[${ocpg_offset} + 3];
        const int ldin = kmax * pack_c_size;
        )");
    const std::string fill_weight_attr =
            StringTemplate::StringTemplateArgs()
                    .add("type_enum", Utils::get_tinynn_dtype_string(dtype))
                    .render(R"(
        out_weights->layout.nr_dim = 2;
        out_weights->layout.dims[0] = group;
        out_weights->layout.dims[1] = )" +
                            strategy->GetPackAWorkspaceSym(inner_ctx.get()) +
                            R"((0, ymax, 0, kmax);
        out_weights->layout.stride[0] = out_weights->layout.dims[1];
        out_weights->layout.stride[1] = 1;
        out_weights->dtype.type_enum=${type_enum};
        out_weights->name = in_weights->name;
                      )");
    const std::string fill_weight_transform =
            StringTemplate::StringTemplateArgs()
                    .add("packa_sym", strategy->PackASym(inner_ctx.get()))
                    .add("specifier", Utils::cvt_dtype_specifier(dtype))
                    .render(
                            R"(    
        ${specifier}* outptr = out_weights->ptr;
        ${specifier}* inptr = in_weights->ptr;
        for(int group_idx = 0; group_idx < group; ++group_idx){
            ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
            outptr += out_weights->layout.dims[1];
            inptr += ymax * kmax;
        }
        )");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);

    return writer.str();
}

std::string kernbody_template(TContext* ctx, Im2colStrategyBase* strategy) {
    auto inner_ctx = strategy->cvt2matmul_ctx(ctx);
    std::stringstream writer;
    auto pack_c_size = get_pack_c_size(ctx);
    std::string bias_ptr_str = ConvImpl::is_bias(ctx) ? "inputs[2]->ptr;" : "NULL;";
    auto dtype = inner_ctx->getAttrStr("dtype");
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    std::string temp_body =
            R"({
        const int pack_c_size = ${pack_c_size};
        const uint32_t pad_h = ${pad_h};
        const uint32_t pad_w = ${pad_w};
        const uint32_t stride_h = ${stride_h};
        const uint32_t stride_w = ${stride_w};
        const uint32_t fh = ${kernel_h};
        const uint32_t fw = ${kernel_w};

        ${specifier}* input_data = inputs[0]->ptr;
        ${specifier}* output_data = outputs[0]->ptr;

        Layout in_layout = inputs[0]->layout;
        Layout out_layout = outputs[0]->layout;
        const int n = in_layout.dims[0];
        const int ic = in_layout.dims[1] * pack_c_size;
        const int ih = in_layout.dims[2];
        const int iw = in_layout.dims[3];

        const int oc = out_layout.dims[1] * pack_c_size;
        const int oh = out_layout.dims[2];
        const int ow = out_layout.dims[3];
        const int ohw = oh * ow;
        
        const size_t N = ohw;
        const size_t LDC = ohw * pack_c_size;
        const size_t align_size = 64;
        const int preset_block_ohw = 192;
        const int block_ohw = preset_block_ohw > ohw ? ohw : preset_block_ohw;

        Layout weight_layout = inputs[1]->layout;
        const int group = weight_layout.dims[0];
        const int icpg = ic / group;
        const int ocpg = oc / group;
        const int group_src_stride = icpg * ih * iw;
        const int group_weight_stride = weight_layout.dims[1];
        const size_t K = fh * fw * icpg;

        const size_t temp_pad_out = (size_t) icpg * (ih + 2 * pad_h) * (iw + 2 * pad_w) * ${dtype_specifier_size};
        const size_t pad_out_offset = (temp_pad_out + align_size - 1) / align_size * align_size;
        const size_t temp_im2col = block_ohw * K * ${dtype_specifier_size};
        const size_t im2col_offset = (temp_im2col + align_size - 1) / align_size * align_size;

        ${specifier}* pad_out_ptr = workspace->ptr;
        ${specifier}* im2col_ptr = workspace->ptr + pad_out_offset;
        ${specifier}* packb_ptr = workspace->ptr + pad_out_offset + im2col_offset;

        for (int n_idx = 0; n_idx < n; ++n_idx) {
            ${specifier}* weight_data = inputs[1]->ptr;
            ${specifier}* bias_data = ${bias_ptr_str}            
            for(int group_idx = 0; group_idx < group; ++group_idx){
                ${specifier}* group_weight_data = weight_data + group_idx * group_weight_stride;
                ${specifier}* group_bias_data = bias_data + group_idx * ocpg;
                ${specifier}* group_ouput_data = output_data + group_idx * ocpg * ohw;
                pad_src(input_data + group_idx * group_src_stride, pad_out_ptr, icpg, ih, iw, pad_h, pad_w);
                for(int ohw_idx = 0; ohw_idx < ohw; ohw_idx += block_ohw){
                    const int real_block_ohw = block_ohw < (ohw - ohw_idx)? block_ohw:(ohw - ohw_idx);
                    const int packed_iw = iw + 2 * pad_w; 
                    const int packed_ih = ih + 2 * pad_h;
                    
                    img2col(pad_out_ptr, im2col_ptr, ow, icpg, packed_ih, packed_iw, fh, fw,
                            stride_h, stride_w, ohw_idx, real_block_ohw);
                            
                    ${pack_b_sym}(packb_ptr, im2col_ptr, real_block_ohw * pack_c_size, 0, real_block_ohw, 0, K);

                    ${naked_kern_sym}(group_weight_data, packb_ptr, group_ouput_data + ohw_idx * pack_c_size, LDC, ocpg, real_block_ohw, K, group_bias_data);
                }
            }
            input_data += ic * ih * iw;
            output_data += oc * oh * ow;
        }
        return TinyNN_SUCCESS;
    })";
    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add("pack_c_size", pack_c_size)
                      .add("bias_ptr_str", bias_ptr_str)
                      .add("pack_b_sym", strategy->PackBSym(inner_ctx.get()))
                      .add("naked_kern_sym",
                           strategy->GetInnerCtxMatmulSym(inner_ctx.get()))
                      .add("specifier", Utils::cvt_dtype_specifier(dtype))
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .render(temp_body);
    return writer.str();
}

}  // namespace
std::string Im2colStrategyBase::PaddingSrc(TContext* ctx) {
    return gen_nchwxx_pad_src_kern(ctx);
}

std::string Im2colStrategyBase::Im2col(TContext* ctx) {
    return gen_nchwxx_im2col_kern(ctx);
}

std::shared_ptr<TContext> Im2colStrategyBase::cvt2matmul_ctx(TContext* ctx) {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("dtype", ctx->getAttrOprand("operand:0").dtype);
    auto fmt = ctx->getAttrStr("format");
    if (fmt == "NCHW44") {
        inner_ctx->setAttr("format", "MK4");
    } else if (fmt == "NCHW88") {
        inner_ctx->setAttr("format", "MK8");
    } else {
        CC_ASSERT(fmt == "NCHW");
        inner_ctx->setAttr("format", "NCHW");
    }
    return inner_ctx;
}

std::string Im2colStrategyBase::GetPackASignature(TContext* inner_ctx) {
    std::stringstream writer;
    auto dtype = inner_ctx->getAttrStr("dtype");
    auto pack_c_size = inner_ctx->getAttrStr("format") == "MK8"
                             ? 8
                             : (inner_ctx->getAttrStr("format") == "MK4" ? 4 : 1);
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    writer << StringTemplate::StringTemplateArgs(inner_ctx)
                      .add("specifier", Utils::cvt_dtype_specifier(dtype))
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .add("pack_c_size", pack_c_size)
                      .render(
                              R"(
static void pack_A(${specifier}* outptr, ${specifier}* inptr, int ldin, int y0, int ymax, int k0, int kmax){
        size_t PACK_C_SIZE=${pack_c_size};
        size_t cp_length = (kmax - k0) * PACK_C_SIZE;
        for (int m = y0; m < ymax; m += PACK_C_SIZE) {
            const ${specifier}* src = inptr + (m / PACK_C_SIZE) * ldin + k0 * PACK_C_SIZE;
            memcpy(outptr, src, cp_length *${dtype_specifier_size});
            outptr += cp_length;
        }
} 
        )");
    return writer.str();
}

std::string Im2colStrategyBase::GetPackBSignature(TContext* inner_ctx) {
    std::stringstream writer;
    auto dtype = inner_ctx->getAttrStr("dtype");
    auto pack_c_size = inner_ctx->getAttrStr("format") == "MK8"
                             ? 8
                             : (inner_ctx->getAttrStr("format") == "MK4" ? 4 : 1);
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    writer << StringTemplate::StringTemplateArgs(inner_ctx)
                      .add("specifier", Utils::cvt_dtype_specifier(dtype))
                      .add("pack_c_size", pack_c_size)
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .render(
                              R"(
static void pack_B(${specifier}* outptr, ${specifier}* inptr, int ldin, int x0, int xmax, int k0, int kmax){
        size_t PACK_C_SIZE=${pack_c_size};
        size_t cp_length = (xmax - x0) * PACK_C_SIZE;
        for (int m = k0; m < kmax; m += PACK_C_SIZE) {
            const ${specifier}* src = inptr + (m / PACK_C_SIZE) * ldin + x0 * PACK_C_SIZE;
            memcpy(outptr, src, cp_length * ${dtype_specifier_size});
            outptr += cp_length;
        } 
} 
        )");
    return writer.str();
}

std::string Im2colStrategyBase::GetPackAWorkspaceSignature(TContext* inner_ctx) {
    std::stringstream writer;
    auto dtype = inner_ctx->getAttrStr("dtype");
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    writer << StringTemplate::StringTemplateArgs(inner_ctx)
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .render(R"(
static size_t pack_A_workspace(int y0, int ymax, int k0, int kmax){
    return (kmax-k0)*(ymax-y0)*${dtype_specifier_size}; 
} 
        )");
    return writer.str();
}

std::string Im2colStrategyBase::GetPackBWorkspaceSignature(TContext* inner_ctx) {
    std::stringstream writer;
    auto dtype = inner_ctx->getAttrStr("dtype");
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    writer << StringTemplate::StringTemplateArgs(inner_ctx)
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .render(R"(
static size_t pack_B_workspace(int x0, int xmax, int k0, int kmax){
    return (kmax-k0)*(xmax-x0)*${dtype_specifier_size}; 
} 
        )");
    return writer.str();
}

std::string Im2colStrategyBase::GetPackBWorkspaceBody(TContext* inner_ctx) {
    std::stringstream writer;
    auto dtype = inner_ctx->getAttrStr("dtype");
    uint32_t dtype_specifier_size = Utils::get_dtype_size(dtype);
    writer << StringTemplate::StringTemplateArgs(inner_ctx)
                      .add("dtype_specifier_size", dtype_specifier_size)
                      .render(R"(

    return (kmax-k0)*(xmax-x0)*${dtype_specifier_size}; 

        )");
    return writer.str();
}
std::string Im2colFrameNchwxx::GenGetWorkSpaceCode(
        TContext* context, Im2colStrategyBase* strategy) {
    CC_ASSERT(
            context->getAttrStr("format") == "NCHW" ||
            context->getAttrStr("format") == "NCHW44" ||
            context->getAttrStr("format") == "NCHW88")
            << "format mismatch  now: " << context->getAttrStr("format")
            << ", expect: NCHW, NCHW44 or MCHW88\n";
    return workspace_template(context, strategy);
}

std::string Im2colFrameNchwxx::GenInitCode(
        TContext* ctx, Im2colStrategyBase* strategy) {
    return init_template(ctx, strategy);
}

std::string Im2colFrameNchwxx::GenKernelBodyCode(
        TContext* ctx, Im2colStrategyBase* strategy) {
    return kernbody_template(ctx, strategy);
}

// vim: syntax=cpp.doxygen
