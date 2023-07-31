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
        const size_t packed_m = 6;
        const size_t packed_k = 4;
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
        const size_t packed_n = 8;
        const size_t packed_k = 4;
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
    int8_t zerobuff[16];
    memset(zerobuff, 0, sizeof(int8_t) * 16);

    int y = y0;
    for (; y < ymax; y += 6) {
        const int8_t* inptr0 = inptr + y * ldin + k0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        const int8_t* inptr4 = inptr3 + ldin;
        const int8_t* inptr5 = inptr4 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);
        prefetch_2x(inptr4);
        prefetch_2x(inptr5);
        int K = kmax - k0;
        for (; K > 31; K -= 32) {
            if (y + 5 >= ymax) {
                switch (y + 5 - ymax) {
                    case 4:
                        inptr1 = zerobuff;
                    case 3:
                        inptr2 = zerobuff;
                    case 2:
                        inptr3 = zerobuff;
                    case 1:
                        inptr4 = zerobuff;
                    case 0:
                        inptr5 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            interleave_6x4_8_b_int8(&inptr0, &inptr1, &inptr2, &inptr3, &inptr4, &inptr5, outptr);
            outptr += 6 * 32;
        }
        for (; K > 15; K -= 16) {
            if (y + 5 >= ymax) {
                switch (y + 5 - ymax) {
                    case 4:
                        inptr1 = zerobuff;
                    case 3:
                        inptr2 = zerobuff;
                    case 2:
                        inptr3 = zerobuff;
                    case 1:
                        inptr4 = zerobuff;
                    case 0:
                        inptr5 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            interleave_6x4_4_b_int8(&inptr0, &inptr1, &inptr2, &inptr3, &inptr4, &inptr5, outptr);
            outptr += 6 * 16;
        }
        if (K > 0) {
            if (y + 5 >= ymax) {
                switch (y + 5 - ymax) {
                    case 4:
                        inptr1 = zerobuff;
                    case 3:
                        inptr2 = zerobuff;
                    case 2:
                        inptr3 = zerobuff;
                    case 1:
                        inptr4 = zerobuff;
                    case 0:
                        inptr5 = zerobuff;
                        break;
                    default:
                        TINYNN_ASSERT(0);
                }
            }
            interleave_6_int8(&inptr0, &inptr1, &inptr2, &inptr3, &inptr4, &inptr5, outptr, 4, K, 0);
            outptr += 6 * ((K + 3) / 4 * 4);
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
    int8_t zerobuff[16];
    memset(zerobuff, 0, sizeof(int8_t) * 16);
    const int ksize = kmax - k0;
    const int ksize8 = (ksize + 3) / 4 * 4 * 8;
    const int ksize4 = (ksize + 3) / 4 * 4 * 4;
    int8_t* outptr_base = outptr;
    //! 4x4 block output start pos
    int8_t* outptr_base4 = outptr + ((xmax - x0) / 8) * ksize8;

    int k = k0;
    for (; k < kmax; k += 4) {
        const int8_t* inptr0 = inptr + k * ldin + x0;
        const int8_t* inptr1 = inptr0 + ldin;
        const int8_t* inptr2 = inptr1 + ldin;
        const int8_t* inptr3 = inptr2 + ldin;
        prefetch_2x(inptr0);
        prefetch_2x(inptr1);
        prefetch_2x(inptr2);
        prefetch_2x(inptr3);

        int x = x0;
        outptr = outptr_base;
        for (; x + 7 < xmax; x += 8) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_8x4_1_b_int8(&inptr0, &inptr1, &inptr2, &inptr3, outptr);
            outptr += ksize8;
        }

        outptr = outptr_base4;
        for (; x + 3 < xmax; x += 4) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_4_int8(&inptr0, &inptr1, &inptr2, &inptr3, outptr, 4, 4, 0);
            outptr += ksize4;
        }

        if (x < xmax) {
            if (k + 3 >= kmax) {
                switch (k + 3 - kmax) {
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

            transpose_4_int8(&inptr0, &inptr1, &inptr2, &inptr3, outptr, 4, xmax - x, 0);
        }

        outptr_base += 8 * 4;
        outptr_base4 += 4 * 4;
    }
}
)";
    return ss.str();
}

std::string GenKern(TContext* ctx, const std::string& dst_specifier) {
    bool with_bias = ctx->getAttrBool("with_bias");
    std::stringstream writer;
    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
static void kern_6x8_bias(const int8_t* packA, const int8_t* packB, int K, ${dst_specifier}* output, int LDC, int is_first_k, 
        int m_remain,  const int32_t* bias_ptr){
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.
    int oddk = (K & 1);
    int k = ((K + 1) / 2) - 1;

    register int32_t* outptr0 asm("r0") = output;
    register int32_t* outptr1 asm("r1") = outptr0 + LDC;
    register int32_t* outptr2 asm("r2") = outptr1 + LDC;
    register int32_t* outptr3 asm("r3") = outptr2 + LDC;
    register int32_t* outptr4 asm("r4") = outptr3 + LDC;
    register int32_t* outptr5 asm("r5") = outptr4 + LDC;

// clang-format off
#define STORE_LINE(reg_index1, reg_index2, reg_index3, reg_index4, n)   \
    "cmp r12, #0 \n"                                                    \
    "beq 101f\n"                                                        \
    "vst1.32 {d" reg_index1 ", d" reg_index2 ", d" reg_index3 ", d"     \
    reg_index4 "}, [r" n "]!\n"                                         \
    "subs r12, r12, #1\n"

#define STORE_C                                    \
    "mov r12, %[m_remain]\n"                       \
    STORE_LINE("8", "9", "10", "11", "0")          \
    STORE_LINE("12", "13", "14", "15", "1")        \
    STORE_LINE("16", "17", "18", "19", "2")        \
    STORE_LINE("20", "21", "22", "23", "3")        \
    STORE_LINE("24", "25", "26", "27", "4")        \
    STORE_LINE("28", "29", "30", "31", "5")        \
    "101:\n"

    // clang-format on

    asm volatile(
                      )");
    if (with_bias)
        writer << R"(
            "vld1.32 {q1}, [%[bias_ptr]]!\n"
            "vld1.32 {d6}, [%[bias_ptr]]\n"
            "vdup.32 q4, d2[0]\n"
            "pld [%[outptr0]]  \n"
            "vdup.32 q5, d2[0]\n"
            "vdup.32 q6, d2[1]\n"
            "pld [%[outptr1]]  \n"
            "vdup.32 q7, d2[1]\n"
            "vdup.32 q8, d3[0]\n"
            "pld [%[outptr2]]  \n"
            "vdup.32 q9, d3[0]\n"
            "vdup.32 q10, d3[1]\n"
            "pld [%[outptr3]]  \n"
            "vdup.32 q11, d3[1]\n"
            "vdup.32 q12, d6[0]\n"
            "pld [%[outptr4]]  \n"
            "vdup.32 q13, d6[0]\n"
            "vdup.32 q14, d6[1]\n"
            "pld [%[outptr5]]  \n"
            "vdup.32 q15, d6[1]\n"
            )";
    else
        writer << R"(
            "veor.s32 q4, q4, q4\n"
            "pld [%[outptr0]]  \n"
            "veor.s32 q5, q5, q5\n"
            "veor.s32 q6, q6, q6\n"
            "pld [%[outptr1]]  \n"
            "veor.s32 q7, q7, q7\n"
            "veor.s32 q8, q8, q8\n"
            "pld [%[outptr2]]  \n"
            "veor.s32 q9, q9, q9\n"
            "veor.s32 q10, q10, q10\n"
            "pld [%[outptr3]]  \n"
            "veor.s32 q11, q11, q11\n"
            "veor.s32 q12, q12, q12\n"
            "pld [%[outptr4]]  \n"
            "veor.s32 q13, q13, q13\n"
            "veor.s32 q14, q14, q14\n"
            "pld [%[outptr5]]  \n"
            "veor.s32 q15, q15, q15\n"
            )";
    writer << R"(
            "6: \n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"

            // Skip loop if we are doing zero iterations of it.
            "cmp %[k], #0      \n"
            "beq 4f            \n"

            // Loop proper
            "1:\n"
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            "vsdot.s8 q5, q3, d1[0]\n"
            "vsdot.s8 q7, q3, d1[1]\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"
            "vsdot.s8 q9, q3, d2[0]\n"
            "vsdot.s8 q11, q3, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vsdot.s8 q13, q3, d3[0]\n"
            "vsdot.s8 q15, q3, d3[1]\n"
            ///////////////////////////////////////
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            "vsdot.s8 q5, q3, d1[0]\n"
            "vsdot.s8 q7, q3, d1[1]\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"
            "vsdot.s8 q9, q3, d2[0]\n"
            "pld [%[b_ptr]]  \n"
            "vsdot.s8 q11, q3, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "subs  %[k], %[k], #1\n"
            "vsdot.s8 q13, q3, d3[0]\n"
            "pld [%[a_ptr]]  \n"
            "vsdot.s8 q15, q3, d3[1]\n"

            "bne  1b\n"

            // Target to use when K is 1 or 2 (i.e. zero iterations of main
            // loop)
            "4:\n"

            // Branch to alternative tail for odd K
            "cmp %[oddk], #0      \n"
            "bne 2f            \n"

            // Detached final iteration (even K)
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            "vsdot.s8 q5, q3, d1[0]\n"
            "vsdot.s8 q7, q3, d1[1]\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"
            "vsdot.s8 q9, q3, d2[0]\n"
            "vsdot.s8 q11, q3, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vsdot.s8 q13, q3, d3[0]\n"
            "vsdot.s8 q15, q3, d3[1]\n"
            ///////////////////////////////////////

            "2:\n"
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            "vsdot.s8 q5, q3, d1[0]\n"
            "vsdot.s8 q7, q3, d1[1]\n"
            "vsdot.s8 q9, q3, d2[0]\n"
            "vsdot.s8 q11, q3, d2[1]\n"
            "vsdot.s8 q13, q3, d3[0]\n"
            "vsdot.s8 q15, q3, d3[1]\n" STORE_C

            : [k] "+r"(k), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [oddk] "+r"(oddk),
            [m_remain] "+r"(m_remain),
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
              [outptr3] "+r"(outptr3), [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [bias_ptr] "+r"(bias_ptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
              "q12", "q13", "q14", "q15", "r12", "cc", "memory");
#undef STORE_LINE
#undef STORE_C
}
                      )";

    writer << StringTemplate::StringTemplateArgs()
                      .add("dst_specifier", dst_specifier)
                      .render(R"(
static void kern_6x4_bias(const int8_t* packA, const int8_t* packB, int K, ${dst_specifier}* output, int LDC, int is_first_k, 
        int n_remain, int m_remain,  const int32_t* bias_ptr){
    K /= 4;
    const int8_t* a_ptr = packA;
    const int8_t* b_ptr = packB;
    // Fix up for odd lengths - set a flag if K is odd, but make
    // sure we round up the iteration count.

// clang-format off
#define STORE_LINE(reg_index1, reg_index2, n)                   \
    "cmp r12, #0 \n"                                            \
    "beq 105f\n"                                                \
    "cmp %[n_remain], #4\n"                                     \
    "blt 103" n "f\n"                                           \
    "vst1.32 {d" reg_index1 ", d" reg_index2 "}, [r" n " ]!\n"  \
    "b 104" n "f\n"                                             \
    "103" n ":\n"                                               \
    "cmp %[n_remain], #0\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index1 "[0]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #1\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index1 "[1]}, [r" n " ]!\n"                \
    "cmp %[n_remain], #2\n"                                     \
    "beq 104" n "f\n"                                           \
    "vst1.32 {d" reg_index2 "[0]}, [r" n " ]!\n"                \
    "104" n ":\n"                                               \
    "subs r12, r12, #1\n"

#define STORE_C                     \
    "mov r12, %[m_remain]\n"        \
    STORE_LINE("8", "9", "0")       \
    STORE_LINE("12", "13", "1")     \
    STORE_LINE("16", "17", "2")     \
    STORE_LINE("20", "21", "3")     \
    STORE_LINE("24", "25", "4")     \
    STORE_LINE("28", "29", "5")     \
    "105:\n"

    // clang-format on

    register int32_t* outptr0 asm("r0") = output;
    register int32_t* outptr1 asm("r1") = outptr0 + LDC;
    register int32_t* outptr2 asm("r2") = outptr1 + LDC;
    register int32_t* outptr3 asm("r3") = outptr2 + LDC;
    register int32_t* outptr4 asm("r4") = outptr3 + LDC;
    register int32_t* outptr5 asm("r5") = outptr4 + LDC;

    asm volatile(
                      )");
    if (with_bias)
        writer << R"(
            "vld1.32 {q5}, [%[bias_ptr]]!\n"
            "vld1.32 {d14}, [%[bias_ptr]]\n"
            "vdup.32 q4, d10[0]\n"
            "pld [%[outptr0]]  \n"
            "vdup.32 q6, d10[1]\n"
            "pld [%[outptr1]]  \n"
            "vdup.32 q8, d11[0]\n"
            "pld [%[outptr2]]  \n"
            "vdup.32 q10, d11[1]\n"
            "pld [%[outptr3]]  \n"
            "vdup.32 q12, d14[0]\n"
            "pld [%[outptr4]]  \n"
            "vdup.32 q14, d14[1]\n"
            "pld [%[outptr5]]  \n"
            )";
    else
        writer << R"(
            "veor.s32 q4, q4, q4\n"
            "pld [%[outptr0]]  \n"
            "veor.s32 q6, q6, q6\n"
            "pld [%[outptr1]]  \n"
            "veor.s32 q8, q8, q8\n"
            "pld [%[outptr2]]  \n"
            "veor.s32 q10, q10, q10\n"
            "pld [%[outptr3]]  \n"
            "veor.s32 q12, q12, q12\n"
            "pld [%[outptr4]]  \n"
            "veor.s32 q14, q14, q14\n"
            "pld [%[outptr5]]  \n"
            )";
    writer << R"(
            "6:\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"

            // Skip loop if we are doing zero iterations of it.
            "cmp  %[k], #2\n"
            "bgt  1f\n"
            "beq  4f\n"
            "blt  2f\n"

            // Loop proper
            "1:\n"
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            ///////////////////////////////////////
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q3, d1[0]\n"
            "vsdot.s8 q6 , q3, d1[1]\n"
            "vsdot.s8 q8 , q3, d2[0]\n"
            "vld1.s8  {q2}, [%[b_ptr]]!\n"
            "vsdot.s8 q10 , q3, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vsdot.s8 q12 , q3, d3[0]\n"
            "vsdot.s8 q14 , q3, d3[1]\n"

            "sub  %[k], %[k], #2\n"
            "cmp  %[k], #2\n"
            "bgt  1b\n"
            "blt  2f\n"

            // Target to use when left K is 2
            "4:\n"

            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vld1.s8  {q3}, [%[b_ptr]]!\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vld1.s8  {d1}, [%[a_ptr]]!\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            ///////////////////////////////////////
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q3, d1[0]\n"
            "vsdot.s8 q6 , q3, d1[1]\n"
            "vsdot.s8 q8 , q3, d2[0]\n"
            "vsdot.s8 q10 , q3, d2[1]\n"
            "vsdot.s8 q12 , q3, d3[0]\n"
            "vsdot.s8 q14 , q3, d3[1]\n"
            "b 3f\n"

            // tail for left K is 1

            "2:\n"
            "vld1.s8  {q1}, [%[a_ptr]]!\n"
            "vsdot.s8 q4 , q2, d1[0]\n"
            "vsdot.s8 q6 , q2, d1[1]\n"
            "vsdot.s8 q8 , q2, d2[0]\n"
            "vsdot.s8 q10 , q2, d2[1]\n"
            "vsdot.s8 q12 , q2, d3[0]\n"
            "vsdot.s8 q14 , q2, d3[1]\n"

            "3:\n"

            STORE_C

            : [k] "+r"(K), [a_ptr] "+r"(a_ptr), [b_ptr] "+r"(b_ptr), [n_remain] "+r"(n_remain),
              [m_remain] "+r"(m_remain),
              [outptr0] "+r"(outptr0), [outptr1] "+r"(outptr1), [outptr2] "+r"(outptr2),
              [outptr3] "+r"(outptr3), [outptr4] "+r"(outptr4), [outptr5] "+r"(outptr5), [bias_ptr] "+r"(bias_ptr)
            :
            : "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11",
              "q12", "q13", "q14", "q15", "r12", "cc", "memory");
#undef STORE_LINE
#undef STORE_C
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
    const size_t A_INTERLEAVE = 6;
    const size_t B_INTERLEAVE = 8;
    K = round_up(K, 4);
    const int K4 = K * 4;
    const int K6 = K * 6;
    const int K8 = K * 8;
    ${dst_specifier}* gemm_output = (${dst_specifier}*)${gen_gemm_output};
    size_t m = 0;
    for (; m + A_INTERLEAVE - 1 < M; m += A_INTERLEAVE) {
        ${dst_specifier}* output = gemm_output + (m * LDC);

        size_t n = 0;
        const int8_t* cur_packB = pack_b;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_6x8_bias(pack_a, cur_packB, K, output, LDC, 1, 6, bias_ptr + m);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }
        for (; n < N; n += 4){
            kern_6x4_bias(pack_a, cur_packB, K, output, LDC, 1, MIN(N - n, 4), 6, bias_ptr + m);
            output += MIN(N - n, 4);
            cur_packB += K4;
        }
        pack_a += K6;
    }
    if (m < M) {
        ${dst_specifier}* output = gemm_output + (m * LDC);
        int m_remain = MIN(M - m, 6);
        int32_t bias_remain[6] = {0};
        if(bias_ptr)
            memcpy(bias_remain, bias_ptr + m, m_remain * sizeof(int32_t));

        size_t n = 0;
        const int8_t* cur_packB = pack_b;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_6x8_bias(pack_a, cur_packB, K, output, LDC, 1, m_remain, bias_remain);
            output += B_INTERLEAVE;
            cur_packB += K8;
        }
        for (; n < N; n += 4){
            kern_6x4_bias(pack_a, cur_packB, K, output, LDC, 1, MIN(N - n, 4), m_remain, bias_remain);
            output += MIN(N - n, 4);
            cur_packB += K4;
        }
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

std::string DotInt8x8x32M6N8K4MatMulKernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Armv7_gemm_MK_dot_int8x8x32_M8N6K4";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        ss << "_" << ctx->getAttrStr("nonlineMode");
    }
    auto dtype = ctx->getAttrStr("dtype");
    CC_ASSERT(Utils::is_quant_dtype(dtype, 8)) << "Only support qsi8 dtype.\n";
    ss << "_qsi8";
    if (ctx->haveAttr("last_dtype")) {
        auto last_dtype = ctx->getAttrStr("last_dtype");
        ss << "_"
           << "output_dtype_" << last_dtype;
    }
    return ss.str();
}

std::vector<KernelObj> DotInt8x8x32M6N8K4MatMulKernel::GetDependInternalSymbol(
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

std::string DotInt8x8x32M6N8K4MatMulKernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string DotInt8x8x32M6N8K4MatMulKernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

std::string DotInt8x8x32M6N8K4MatMulKernel::GetKernelBody(TContext* context) const {
    auto d_type = context->getAttrStr("dtype");
    CC_ASSERT(d_type != "8832");
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <math.h>\n";
    writer << "#include <marm_neon.h>\n";
    writer << "#include \"utils.h\"\n";
    std::string dst_specifier = "int32_t";

    writer << GenKern(context, dst_specifier);
    writer << "\n\n";
    writer << R"(
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define round_up(dividend, divisor) ((dividend + divisor - 1) / (divisor)) * (divisor)
    )";
    writer << prefetch();
    writer << KernelGen::Armv7::gen_armv7_interleave_helper_int8();
    writer << KernelGen::Armv7::gen_armv7_interleave_8x4_8_b_int8();
    writer << KernelGen::Armv7::gen_armv7_interleave_8x4_4_b_int8();
    writer << KernelGen::Armv7::gen_armv7_interleave_6_int8();
    writer << KernelGen::Armv7::gen_armv7_transpose_8x4_1_b_int8();
    writer << KernelGen::Armv7::gen_armv7_transpose_4_int8();
    writer << gen_pack_a(GetPackASignature(context));
    writer << gen_pack_b(GetPackBSignature(context));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(context));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(context));
    auto postprocess_pair = gen_postprocess_inline(context, true);
    writer << postprocess_pair.first;
    writer << gen_kernel(
            dst_specifier, GetNakedKernelSignature(context), context,
            postprocess_pair.second, "", true);

    return writer.str();
}