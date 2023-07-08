#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "Arm/Armv7/Activation.h"
#include "Arm/Armv7/armv7_asm_utils.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Armv7;
using namespace ArmCommon;
namespace {
std::string prefetch() {
    return R"(
#define ASM_PREFETCH(address) "PLD " address "\n"
    )" + KernelGen::ArmCommon::gen_common_prefetch_2x_f32();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const size_t CACHELINE_SIZE = 64;
        const size_t m_align_size = 16;
        const size_t packed_m = 4;
        const size_t packed_k = 16;
        size_t k = kmax - k0;
        size_t m = ymax - y0;
        size_t round_m = (m + packed_m -1) / packed_m * packed_m;
        size_t round_k = (k + packed_k -1) / packed_k * packed_k;
        size_t ws_size = sizeof(int8_t) * round_m * round_k;
        return (ws_size + CACHELINE_SIZE-1)/CACHELINE_SIZE*CACHELINE_SIZE + m_align_size;
    }
)";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const size_t CACHELINE_SIZE = 64;
        const size_t m_align_size = 16;
        const size_t packed_n = 2;
        const size_t packed_k = 16;
        size_t k = kmax - k0;
        size_t hw = xmax - x0;
        size_t round_n = (hw+packed_n-1)/packed_n*packed_n;
        size_t round_k = (k+packed_k-1)/packed_k*packed_k;
        size_t ws_size = sizeof(int8_t) * round_n * round_k;
        return (ws_size+CACHELINE_SIZE-1)/CACHELINE_SIZE*CACHELINE_SIZE + m_align_size;
    }
)";
    return ss.str();
}

std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    //! pack form {oc/4, ic/4, 4(ic), 4(oc)} to {oc/4, ic/16, 4(oc), 16(ic)}
    int8_t zerobuff[4][64];
    memset(zerobuff, 0, sizeof(int8_t) * 64 * 4);
    TINYNN_ASSERT(
            ymax % 4 == 0 && y0 % 4 == 0 && (ymax - y0) % 4 == 0);
    TINYNN_ASSERT(
            kmax % 4 == 0 && k0 % 4 == 0 && (kmax - k0) % 4 == 0);
    size_t roundk = round_up(kmax - k0, 16);
    size_t out_offset = roundk * 4;
    int y = y0;
    int start_y = y0 / 4;
    for (; y + 15 < ymax; y += 16, start_y += 4) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_4x4_4_b(
                    &inptr0, &inptr1, &inptr2, &inptr3, output, out_offset);
            output += 64;
        }
        if (K > 0) {
            memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            memcpy(zerobuff[1], inptr1, sizeof(int8_t) * K * 4);
            memcpy(zerobuff[2], inptr2, sizeof(int8_t) * K * 4);
            memcpy(zerobuff[3], inptr3, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            inptr1 = zerobuff[1];
            inptr2 = zerobuff[2];
            inptr3 = zerobuff[3];
            transpose_interleave_4x4_4_b(
                    &inptr0, &inptr1, &inptr2, &inptr3, output, out_offset);
            output += 64;
        }
    }
    for (; y + 3 < ymax; y += 4, start_y++) {
        const int8_t* inptr0 = inptr + start_y * ldin + k0 * 4;
        int8_t* output = outptr + (y - y0) / 4 * out_offset;
        prefetch_2x(inptr0);
        int K = kmax - k0;
        for (; K > 15; K -= 16) {
            transpose_interleave_1x4_4_b(&inptr0, output, 64);
            output += 64;
        }
        if (K > 0) {
            memcpy(zerobuff[0], inptr0, sizeof(int8_t) * K * 4);
            inptr0 = zerobuff[0];
            transpose_interleave_1x4_4_b(&inptr0, output, 64);
            output += 64;
        }
    }
}
)";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    int32_t zerobuff[4];
    memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ICB = (ksize) / 4;
    const int ksize2 = round_up(ICB, 4) * 2;
    int32_t* out = (int32_t*)(outptr);
    TINYNN_ASSERT(
            kmax % 4 == 0 && k0 % 4 == 0 && ksize % 4 == 0);

    int k = k0 / 4;
    for (; k + 3 < ICB; k += 4) {
        const int32_t* inptr0 = (const int32_t*)(inptr + k * ldin + x0);
        const int32_t* inptr1 =
                (const int32_t*)(inptr + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                (const int32_t*)(inptr + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                (const int32_t*)(inptr + (k + 3) * ldin + x0);
        int32_t* outptr_inner = out;

        int x = x0;
        for (; x + 1 < xmax; x += 2) {
            transpose_4x2_1_s(&inptr0, &inptr1, &inptr2, &inptr3, outptr_inner, 8);
            outptr_inner += ksize2;
        }
        if (x < xmax) {
            *outptr_inner++ = *inptr0++;
            *outptr_inner++ = *inptr1++;
            *outptr_inner++ = *inptr2++;
            *outptr_inner++ = *inptr3++;
        }
        out += 4 * 2;
    }
    if (k < ICB) {
        const int32_t* inptr0 = (const int32_t*)(inptr + k * ldin + x0);
        const int32_t* inptr1 =
                (const int32_t*)(inptr + (k + 1) * ldin + x0);
        const int32_t* inptr2 =
                (const int32_t*)(inptr + (k + 2) * ldin + x0);
        const int32_t* inptr3 =
                (const int32_t*)(inptr + (k + 3) * ldin + x0);
        int32_t* outptr_inner = out;

        int x = x0;
        for (; x + 1 < xmax; x += 2) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            transpose_4x2_1_s(&inptr0, &inptr1, &inptr2, &inptr3, outptr_inner, 8);
            outptr_inner += ksize2;
        }
        if (x < xmax) {
            if (k + 3 >= ICB) {
                switch (k + 3 - ICB) {
                    case 2:
                        inptr1 = zerobuff;
                    case 1:
                        inptr2 = zerobuff;
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            *outptr_inner++ = *inptr0;
            *outptr_inner++ = *inptr1;
            *outptr_inner++ = *inptr2;
            *outptr_inner++ = *inptr3;
        }
        out += 4 * 2;
    }
}
)";
    return ss.str();
}

std::string GetKern4x2(TContext* ctx, const std::string& dst_specifier) {
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
static void kern_4x2_bias_relu(const int8_t* packA, const int8_t* packB, int K, ${dst_specifier}* output, 
        int n_remain,  const int32_t* bias_ptr, float src_scale, float dst_scale){
    K /= 16;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    float* src_scale_ptr = &src_scale;
    float* dst_scale_ptr = &dst_scale;
    asm volatile(
            "vldr d0, [%[b_ptr], #0]\n"
            "vmov.i32 q8, #0\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmov.i32 q9, #0\n"
            "vldr d1, [%[b_ptr], #8]\n"
            "vmov.i32 q10, q8\n"
            "vldr d5, [%[a_ptr], #8]\n"
            "vmov.i32 q11, q8\n"
            "vldr d2, [%[b_ptr], #16]\n"
            "vmov.i32 q12, q8\n"
            "vldr d6, [%[a_ptr], #16]\n"
            "vmov.i32 q13, q8\n"
            "vldr d3, [%[b_ptr], #24]\n"
            "vmov.i32 q14, q8\n"
            "vldr d7, [%[a_ptr], #24]\n"
            "vmov.i32 q15, q8\n"

            // General loop.
            "1:\n"
            "vmull.s8    q4,  d0,  d4\n"
            "add %[b_ptr], %[b_ptr], #32\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #32]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d6, [%[a_ptr], #48]\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vldr d5, [%[a_ptr], #40]\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"
            "vldr d7, [%[a_ptr], #56]\n"

            "vpadal.s16   q8,  q4\n"
            "add %[a_ptr], %[a_ptr], #64\n"
            "vpadal.s16   q9,  q5\n"
            "subs %[K], %[K], #1\n"
            "vpadal.s16   q10, q6\n"
            "vpadal.s16   q11, q7\n"

            "beq 2f\n"

            "vmull.s8    q4,  d0,  d4\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vldr d4, [%[a_ptr], #0]\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vldr d0, [%[b_ptr], #0]\n"
            "vmull.s8    q7,  d2,  d6\n"
            "vldr d2, [%[b_ptr], #16]\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vldr d6, [%[a_ptr], #16]\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vldr d5, [%[a_ptr], #8]\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vldr d1, [%[b_ptr], #8]\n"
            "vmlal.s8    q7,  d3,  d7\n"
            "vldr d3, [%[b_ptr], #24]\n"

            // Add pairwise, accumulate into 32-bit accumulators.
            "vpadal.s16   q12, q4\n"
            "vldr d7, [%[a_ptr], #24]\n"
            "vpadal.s16   q13, q5\n"
            "vpadal.s16   q14, q6\n"
            "vpadal.s16   q15, q7\n"

            "b 1b\n"

            "2:\n"
            // Multiply first half.
            "vmull.s8    q4,  d0,  d4\n"
            "vmull.s8    q5,  d2,  d4\n"
            "vmull.s8    q6,  d0,  d6\n"
            "vmull.s8    q7,  d2,  d6\n"

            "vmlal.s8    q4,  d1,  d5\n"
            "vmlal.s8    q5,  d3,  d5\n"
            "vmlal.s8    q6,  d1,  d7\n"
            "vmlal.s8    q7,  d3,  d7\n"

            "vpadal.s16   q12, q4\n"
            "vpadal.s16   q13, q5\n"
            "vpadal.s16   q14, q6\n"
            "vpadal.s16   q15, q7\n"

            // Reduce 32bit accumulators horizontally.
            "vpadd.s32 d0, d16, d17\n"
            "vpadd.s32 d1, d18, d19\n"
            "vpadd.s32 d2, d20, d21\n"
            "vpadd.s32 d3, d22, d23\n"
            "vpadd.s32 d4, d24, d25\n"
            "vpadd.s32 d5, d26, d27\n"
            "vpadd.s32 d6, d28, d29\n"
            "vpadd.s32 d7, d30, d31\n"
            
                      )");
    if (with_bias) {
        writer << R"(
            "cmp %[n_remain], #1\n"
            "beq 3f\n"
            "vld1.32 {q9}, [%[bias_ptr]]\n"
            "3:\n"
            "vld1.32 {q8}, [%[bias_ptr]]\n"
        )";
    } else {
        writer << R"(
            "cmp %[n_remain], #1\n"
            "beq 3f\n"
            "veor q9, q9, q9\n"
            "3:\n"
            "veor q8, q8, q8\n"
        )";
    }
    writer << R"(
            "vpadd.s32 d8, d0, d2\n"
            "vpadd.s32 d9, d4, d6\n"
            "vpadd.s32 d10, d1, d3\n"
            "vpadd.s32 d11, d5, d7\n"

            "vadd.s32 q4, q8, q4\n"
            "vadd.s32 q5, q9, q5\n"

            "cmp %[n_remain], #1\n"
            "beq 4f\n"
            "vstr d10, [%[outptr], #16]\n"
            "vstr d11, [%[outptr], #24]\n"
            "4:\n"
            "vstr d8, [%[outptr]]\n"
            "vstr d9, [%[outptr], #8]\n"

            : [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [bias_ptr] "+r"(bias_ptr),  
              [K] "+r"(K), [outptr] "+r"(output), [n_remain] "+r"(n_remain)
            :
            : "cc", "memory", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8",
              "q9", "q10", "q11", "q12", "q13", "q14", "q15");
}
)";
    return writer.str();
}

std::string gen_kernel(
        const std::string& dst_specifier, const std::string& sig, TContext* ctx,
        const std::string& postprocess_call, const std::string& preset_str = "",
        bool with_temp_dst = false) {
    std::string gemm_output = "C";
    if (with_temp_dst) {
        gemm_output = "workspace";
    }
    std::stringstream writer;
    std::string kernel_body_temp = R"(
${kernel_sig} {
    ${preset_str}
    const size_t A_INTERLEAVE = 4;
    const size_t B_INTERLEAVE = 2;
    //! K is packed to times of 4
    TINYNN_ASSERT(M % 4 == 0);
    TINYNN_ASSERT(K % 4 == 0);
    K = round_up(K, 16);
    const int K4 = K * 4;
    const int K2 = K * 2;
    ${dst_specifier}* gemm_output = (${dst_specifier}*)${gen_gemm_output};
    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        ${dst_specifier}* output = gemm_output + (m / 4 * LDC);

        size_t n = 0;
        const int8_t* cur_packB = pack_b;
        for (; n < N; n += B_INTERLEAVE) {
            kern_4x2_bias_relu(pack_a, cur_packB, K, output, MIN(N - n, 2), bias_ptr + m, temp_scale, dst_scale_inv);
            output += B_INTERLEAVE * 4;
            cur_packB += K2;
        }
        pack_a += K4;
    }
    ${postprocess_call}
}
)";
    return StringTemplate::StringTemplateArgs()
            .add("gen_gemm_output", gemm_output)
            .add("dst_specifier", dst_specifier)
            .add("postprocess_call", postprocess_call)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .render(kernel_body_temp);
}
}  // namespace

std::string Int8x8x32MK4MatMulKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Armv7_gemm_int8x8x32";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    auto dtype = ctx->getAttrStr("dtype");
    if (Utils::is_quant_dtype(dtype)) {
        ss << "_qsi8";
    } else {
        CC_ASSERT(dtype == "8832");
        ss << "_" << dtype;
    }
    if (ctx->haveAttr("last_dtype")) {
        auto last_dtype = ctx->getAttrStr("last_dtype");
        ss << "_"
           << "output_dtype_" << last_dtype;
    }
    return ss.str();
}

std::vector<KernelObj> Int8x8x32MK4MatMulKernel::GetDependInternalSymbol(
        TContext* ctx) const {
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    std::vector<KernelObj> depends;
    if (nonline_mode == "SIGMOID") {
        ExpNeonKernel kern;
        depends.emplace_back(
                kern.GetKernelSymbol(ctx), kern.GetKernelBody(ctx),
                kern.GetBodyGuardBegin(ctx), kern.GetBodyGuardEnd(ctx));
    }
    return depends;
}

std::string Int8x8x32MK4MatMulKernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string Int8x8x32MK4MatMulKernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

std::string Int8x8x32MK4MatMulKernel::GetKernelBody(TContext* context) const {
    auto d_type = context->getAttrStr("dtype");
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <math.h>\n";
    writer << "#include <marm_neon.h>\n";
    writer << "#include \"utils.h\"\n";
    std::string dst_specifier = "int32_t";

    writer << GetKern4x2(context, dst_specifier);
    writer << "\n\n";
    writer << R"(
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define round_up(dividend, divisor) ((dividend + divisor - 1) / (divisor)) * (divisor)
    )";
    writer << prefetch();
    writer << KernelGen::Armv7::gen_armv7_transpose_interleave_4x4_4_b_int8();
    writer << KernelGen::Armv7::gen_armv7_transpose_interleave_1x4_4_b_int8();
    writer << KernelGen::Armv7::gen_armv7_transpose_4x2_1_s_int32();
    writer << gen_pack_a(GetPackASignature(context));
    writer << gen_pack_b(GetPackBSignature(context));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(context));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(context));
    if (d_type != "8832") {
        auto postprocess_pair = gen_postprocess_inline(context, true);
        writer << postprocess_pair.first;
        writer << gen_kernel(
                dst_specifier, GetNakedKernelSignature(context), context,
                postprocess_pair.second, "", true);
    } else {
        std::string preset_temp = R"(
    size_t pack_a_size = ${packa_workspace_sym}(0, M, 0, K);
    int8_t* pack_a = (int8_t*)workspace;
    int8_t* pack_b = (int8_t*)workspace + pack_a_size;

    ${packa_sym}(pack_a, A, LDA, 0, M, 0, K);
    ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
)";
        std::string preset_str =
                StringTemplate::StringTemplateArgs()
                        .add("packa_workspace_sym", GetPackAWorkspaceSymbol(context))
                        .add("packa_sym", GetPackASymbol(context))
                        .add("packb_sym", GetPackBSymbol(context))
                        .render(preset_temp);
        writer << gen_kernel(
                dst_specifier, GetKernelSignature(context), context, "", preset_str,
                false);
    }
    return writer.str();
}