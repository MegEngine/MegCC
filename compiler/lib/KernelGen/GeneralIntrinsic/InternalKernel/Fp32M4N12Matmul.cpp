#include <string>
#include "GeneralIntrinsic/Activation.h"
#include "GeneralIntrinsic/ElemwiseHelper/ElemwiseHelper.h"
#include "GeneralIntrinsic/GIMathHelper.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;
namespace {
std::string utilsFunc(void) {
    return R"(
static inline size_t min(size_t a,size_t b){
    return a < b ? a : b;
}
    )";
}

std::string interleave_4x4_1_s() {
    return R"(
static GI_FORCEINLINE void interleave_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr1);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr2);
    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr3);
    inptr0 += 4;
    inptr1 += 4;
    inptr2 += 4;
    inptr3 += 4;

    GiStoreFloat32(outptr, d0d1);
    GiStoreFloat32(outptr + 1 * 4, d2d3);
    GiStoreFloat32(outptr + 2 * 4, d4d5);
    GiStoreFloat32(outptr + 3 * 4, d6d7);
    outptr += 16;
}
)";
}

std::string interleave_4x12_1_s() {
    return R"(
static GI_FORCEINLINE void interleave_4x12_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr0 + 1 * 4);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr0 + 2 * 4);
    inptr0 += 12;

    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr1);
    GI_FLOAT32_t d8d9 = GiLoadFloat32(inptr1 + 1 * 4);
    GI_FLOAT32_t d10d11 = GiLoadFloat32(inptr1 + 2 * 4);
    inptr1 += 12;

    GI_FLOAT32_t d12d13 = GiLoadFloat32(inptr2);
    GI_FLOAT32_t d14d15 = GiLoadFloat32(inptr2 + 1 * 4);
    GI_FLOAT32_t d16d17 = GiLoadFloat32(inptr2 + 2 * 4);
    inptr2 += 12;

    GI_FLOAT32_t d18d19 = GiLoadFloat32(inptr3);
    GI_FLOAT32_t d20d21 = GiLoadFloat32(inptr3 + 1 * 4);
    GI_FLOAT32_t d22d23 = GiLoadFloat32(inptr3 + 2 * 4);
    inptr3 += 12;

    GiStoreFloat32(outptr, d0d1);
    GiStoreFloat32(outptr + 1 * 4, d2d3);
    GiStoreFloat32(outptr + 2 * 4, d4d5);
    GiStoreFloat32(outptr + 3 * 4, d6d7);
    GiStoreFloat32(outptr + 4 * 4, d8d9);
    GiStoreFloat32(outptr + 5 * 4, d10d11);
    GiStoreFloat32(outptr + 6 * 4, d12d13);
    GiStoreFloat32(outptr + 7 * 4, d14d15);
    GiStoreFloat32(outptr + 8 * 4, d16d17);
    GiStoreFloat32(outptr + 9 * 4, d18d19);
    GiStoreFloat32(outptr + 10 * 4, d20d21);
    GiStoreFloat32(outptr + 11 * 4, d22d23);
    outptr += 48;
})";
}

std::string interleave_1x12_1_s() {
    return R"(

static GI_FORCEINLINE void interleave_1x12_1_s(const float* inptr0, float* outptr) {
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr0 + 1 * 4);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr0 + 2 * 4);
    inptr0 += 12;

    GiStoreFloat32(outptr, d0d1);
    GiStoreFloat32(outptr + 1 * 4, d2d3);
    GiStoreFloat32(outptr + 2 * 4, d4d5);
    outptr += 12;
}
    )";
}
std::string interleave_1x4_1_s() {
    return R"(

static GI_FORCEINLINE void interleave_1x4_1_s(const float* inptr0, float* outptr) {
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    inptr0 += 4;

    GiStoreFloat32(outptr, d0d1);
    outptr += 4;
}
    )";
}
std::string interleave_helper() {
    return R"(
static GI_FORCEINLINE void interleave_helper(
        const float* inptr, float* outptr, int unroll_k, int ksize, float val) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}
    )";
}
std::string interleave_1() {
    return R"(

static GI_FORCEINLINE void interleave_1(
        const float* inptr0, float* outptr, int unroll_k, int ksize, float val) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size;outptr+=unroll_k;
    }
}
    )";
}

std::string interleave_4() {
    return R"(
static GI_FORCEINLINE void interleave_4(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int unroll_k, int ksize, float val) {
     for (int k = 0; k < ksize; k += unroll_k) {
        int size = min(unroll_k, ksize - k);
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size; outptr += unroll_k;
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        inptr1 += size; outptr += unroll_k;
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        inptr2 += size; outptr += unroll_k;
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        inptr3 += size; outptr += unroll_k;
    }
}
    )";
}

std::string transpose_4x4_1_s() {
    return R"(

static GI_FORCEINLINE void transpose_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int stride) {

    stride = stride / sizeof(float);
    stride -= 2;
    GI_FLOAT32_t d0d1 = GiLoadFloat32(inptr0);
    GI_FLOAT32_t d2d3 = GiLoadFloat32(inptr1);
    GI_FLOAT32_t d4d5 = GiLoadFloat32(inptr2);
    GI_FLOAT32_t d6d7 = GiLoadFloat32(inptr3);
    inptr0 += 4;
    inptr1 += 4;
    inptr2 += 4;
    inptr3 += 4;

    GI_FLOAT32_V2_t q0q1 = GiZipqFloat32(d0d1, d2d3);
    GI_FLOAT32_V2_t q2q3 = GiZipqFloat32(d4d5, d6d7);

    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q0q1, 0)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q2q3, 0)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q0q1, 0)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q2q3, 0)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q0q1, 1)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetLowFloat32(GiGetSubVectorFloat32V2(q2q3, 1)));
    outptr += stride;

    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q0q1, 1)));
    outptr += 2;
    GiSt1Float32(outptr, GiGetHighFloat32(GiGetSubVectorFloat32V2(q2q3, 1)));
    outptr += stride;
}
)";
}

std::string pack_A_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_n" +
           GeneralIntrinsic::MatmulInternal::GenPackACall(ctx) +
           R"({
    float zerobuff[4];
    memset(zerobuff, 0, sizeof(float) * 4);
    int y = y0;
    for (; y < ymax; y += 4) {
        const float* inptr0 = inptr + y * ldin + k0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;
        int K = (kmax - k0);
        for (; K > 3; K -= 4) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        
                    case 1:
                        inptr2 = zerobuff;
                        
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:;
                }
            }
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr, 16);
            inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
            outptr+=16;
        }
        if (K > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        
                    case 1:
                        inptr2 = zerobuff;
                        
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:;
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, outptr, 1, K, 0);
            outptr+=4*K;
        }
    }
}
    )";
}

std::string pack_A_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_t" +
           GeneralIntrinsic::MatmulInternal::GenPackACall(ctx) +
           R"( {
    int x0 = y0;
    int xmax = ymax;
    int ksize = kmax - k0;
    int ksize4 = (ksize << 2);
    float* outptr_base = outptr;
    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr0 = inptr + k * ldin + x0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;
        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            float* outptr_interleave = access_outptr;
            interleave_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
            inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 4, xmax - x, 0);
        }
        outptr_base += 4 * 4;
    }
    for (; k < kmax; k++) {
        const float* inptr0 = inptr + k * ldin + x0;
        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 4 <= xmax; x += 4) {
            float* outptr_interleave = access_outptr;
            interleave_1x4_1_s(inptr0, outptr_interleave);
            inptr0 += 4;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            interleave_1(inptr0, access_outptr, 4, xmax - x, 0);
            inptr0 += 4;
        }
        outptr_base += 4;
    }
}
    )";
}

std::string pack_B_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_n" +
           GeneralIntrinsic::MatmulInternal::GenPackBCall(ctx) +
           R"( {
    int ksize = kmax - k0;
    int ksize12 = ksize * 12;
    int ksize4 = (ksize << 2);
    float* outptr_base = outptr;
    float* outptr_base4 = outptr_base + (xmax - x0) / 12 * ksize12;
    int k = k0;
    for (; k + 3 < kmax; k += 4) {
        const float* inptr0 = inptr + k * ldin + x0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;
        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            float* outptr_interleave = access_outptr;
            interleave_4x12_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
            inptr0 += 12;inptr1 += 12;inptr2 += 12;inptr3 += 12;
            access_outptr += ksize12;
        }
        access_outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            float* outptr_interleave = access_outptr;
            interleave_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_interleave);
            inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 4, xmax - x, 0);
        }
        outptr_base += 12 * 4;
        outptr_base4 += 4 * 4;
    }
    for (; k < kmax; k++) {
        const float* inptr0 = inptr + k * ldin + x0;
        int x = x0;
        float* access_outptr = outptr_base;
        for (; x + 12 <= xmax; x += 12) {
            float* outptr_interleave = access_outptr;
            interleave_1x12_1_s(inptr0, outptr_interleave);
            inptr0 += 12;
            access_outptr += ksize12;
        }
        access_outptr = outptr_base4;
        for (; x + 4 <= xmax; x += 4) {
            float* outptr_interleave = access_outptr;
            interleave_1x4_1_s(inptr0, outptr_interleave);
            inptr0 += 4;
            access_outptr += ksize4;
        }
        if (x < xmax) {
            interleave_1(inptr0, access_outptr, 4, xmax - x, 0);
        }
        outptr_base += 12;
        outptr_base4 += 4;
    }
}
    )";
}

std::string pack_B_t(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_t" +
           GeneralIntrinsic::MatmulInternal::GenPackBCall(ctx) +
           R"({
    int y0 = x0;
    int ymax = xmax;
    float* access_outptr = outptr;
    float zerobuff[4];
    memset(zerobuff, 0, sizeof(float) * 4);
    int K12 = 12 * (kmax - k0);
    int y = y0;
    for (; y + 12 <= ymax; y += 12) {
        int yi = y;
        for (; yi < y + 12; yi += 4) {
            const float* inptr0 = inptr + yi * ldin + k0;
            const float* inptr1 = inptr0 + ldin;
            const float* inptr2 = inptr1 + ldin;
            const float* inptr3 = inptr2 + ldin;
            float* outptr_inner = access_outptr + yi - y;
            int x = (kmax - k0);
            for (; x > 3; x -= 4) {
                transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, outptr_inner, 48);
                inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
                outptr_inner+=48;
            }
            for (; x > 0; x--) {
                *outptr_inner++ = *inptr0++;
                *outptr_inner++ = *inptr1++;
                *outptr_inner++ = *inptr2++;
                *outptr_inner++ = *inptr3++;
                outptr_inner += 8;
            }
        }
        access_outptr += K12;
    }
    for (; y < ymax; y += 4) {
        const float* inptr0 = inptr + y * ldin + k0;
        const float* inptr1 = inptr0 + ldin;
        const float* inptr2 = inptr1 + ldin;
        const float* inptr3 = inptr2 + ldin;
        /* Cope with ragged cases by copying from a buffer of zeroes instead
        */
        int x = (kmax - k0);
        for (; x > 3; x -= 4) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        
                    case 1:
                        inptr2 = zerobuff;
                        
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:;
                }
            }
            transpose_4x4_1_s(inptr0, inptr1, inptr2, inptr3, access_outptr, 16);
            inptr0 += 4;inptr1 += 4;inptr2 += 4;inptr3 += 4;
            access_outptr+=16;
        }
        if (x > 0) {
            if ((y + 3) >= ymax) {
                switch ((y + 3) - ymax) {
                    /* Everything falls through in here */
                    case 2:
                        inptr1 = zerobuff;
                        
                    case 1:
                        inptr2 = zerobuff;
                        
                    case 0:
                        inptr3 = zerobuff;
                        break;
                    default:;
                }
            }
            interleave_4(inptr0, inptr1, inptr2, inptr3, access_outptr, 1, x, 0);
            inptr0 += x;inptr1 += x;inptr2 += x;inptr3 += x;
            access_outptr+=4*x;
        }
    }
}
    )";
}

std::string kern4x12(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << R"(
static void kern_4x12(
        const float* packA, const float* packB, int K, float* output, int LDC,
        int m_remain, const float* bias_ptr) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;

    float* r0 = output;
    float* r1 = r0 + LDC;
    float* r2 = r1 + LDC;
    float* r3 = r2 + LDC;

    GI_FLOAT32_t d0d1, d2d3, d4d5, d6d7, d8d9, d10d11, d12d13, d14d15, d16d17, d18d19,
            d20d21, d22d23, d24d25, d26d27, d28d29, d30d31;
)";
    if (with_bias) {
        ss << R"(
        float tmp_bias[4];
        memset(tmp_bias, 0, sizeof(float) * 4);
        memcpy(tmp_bias, bias_ptr, sizeof(float) * m_remain);
        d6d7 = GiLoadFloat32(tmp_bias);
        d8d9 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 0));
        d10d11 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 0));
        d12d13 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 0));
        d14d15 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 1));
        d16d17 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 1));
        d18d19 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 1));
        d20d21 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 0));
        d22d23 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 0));
        d24d25 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 0));
        d26d27 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 1));
        d28d29 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 1));
        d30d31 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 1));
    )";
    } else {
        ss << R"(
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);
        d12d13 = GiBroadcastFloat32(0.0f);
        d14d15 = GiBroadcastFloat32(0.0f);
        d16d17 = GiBroadcastFloat32(0.0f);
        d18d19 = GiBroadcastFloat32(0.0f);
        d20d21 = GiBroadcastFloat32(0.0f);
        d22d23 = GiBroadcastFloat32(0.0f);
        d24d25 = GiBroadcastFloat32(0.0f);
        d26d27 = GiBroadcastFloat32(0.0f);
        d28d29 = GiBroadcastFloat32(0.0f);
        d30d31 = GiBroadcastFloat32(0.0f);
        )";
    }
    std::string body_temp = R"(
    d2d3 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;
    d6d7 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;

    for (; K > 0; K--) {
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 0);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d0d1, 0);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d0d1, 1);
        d16d17 = GiSimdFmaLane(d16d17, d4d5, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d6d7, d0d1, 1);
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d4d5, d0d1, 2);
        d24d25 = GiSimdFmaLane(d24d25, d6d7, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d0d1, 3);
        d28d29 = GiSimdFmaLane(d28d29, d4d5, d0d1, 3);
        d30d31 = GiSimdFmaLane(d30d31, d6d7, d0d1, 3);

        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 0);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d0d1, 0);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d0d1, 1);
        d16d17 = GiSimdFmaLane(d16d17, d4d5, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d6d7, d0d1, 1);
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d4d5, d0d1, 2);
        d24d25 = GiSimdFmaLane(d24d25, d6d7, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d0d1, 3);
        d28d29 = GiSimdFmaLane(d28d29, d4d5, d0d1, 3);
        d30d31 = GiSimdFmaLane(d30d31, d6d7, d0d1, 3);

        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
    }

    if (1 == oddk) {
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 0);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d0d1, 0);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d0d1, 1);
        d16d17 = GiSimdFmaLane(d16d17, d4d5, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d6d7, d0d1, 1);
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d4d5, d0d1, 2);
        d24d25 = GiSimdFmaLane(d24d25, d6d7, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d0d1, 3);
        d28d29 = GiSimdFmaLane(d28d29, d4d5, d0d1, 3);
        d30d31 = GiSimdFmaLane(d30d31, d6d7, d0d1, 3);

    } else {
        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 0);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d0d1, 0);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d0d1, 1);
        d16d17 = GiSimdFmaLane(d16d17, d4d5, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d6d7, d0d1, 1);
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d4d5, d0d1, 2);
        d24d25 = GiSimdFmaLane(d24d25, d6d7, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d0d1, 3);
        d28d29 = GiSimdFmaLane(d28d29, d4d5, d0d1, 3);
        d30d31 = GiSimdFmaLane(d30d31, d6d7, d0d1, 3);

        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d2d3 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d2d3, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 0);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d0d1, 0);
        d14d15 = GiSimdFmaLane(d14d15, d2d3, d0d1, 1);
        d16d17 = GiSimdFmaLane(d16d17, d4d5, d0d1, 1);
        d18d19 = GiSimdFmaLane(d18d19, d6d7, d0d1, 1);
        d20d21 = GiSimdFmaLane(d20d21, d2d3, d0d1, 2);
        d22d23 = GiSimdFmaLane(d22d23, d4d5, d0d1, 2);
        d24d25 = GiSimdFmaLane(d24d25, d6d7, d0d1, 2);
        d26d27 = GiSimdFmaLane(d26d27, d2d3, d0d1, 3);
        d28d29 = GiSimdFmaLane(d28d29, d4d5, d0d1, 3);
        d30d31 = GiSimdFmaLane(d30d31, d6d7, d0d1, 3);
    }
    ${GenActivate(d8d9, d8d9)}
    ${GenActivate(d10d11, d10d11)}
    ${GenActivate(d12d13, d12d13)}
    ${GenActivate(d14d15, d14d15)}
    ${GenActivate(d16d17, d16d17)}
    ${GenActivate(d18d19, d18d19)}
    ${GenActivate(d20d21, d20d21)}
    ${GenActivate(d22d23, d22d23)}
    ${GenActivate(d24d25, d24d25)}
    ${GenActivate(d26d27, d26d27)}
    ${GenActivate(d28d29, d28d29)}
    ${GenActivate(d30d31, d30d31)}
    if (m_remain == 4) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);

        GiStoreFloat32(r2, d20d21);
        GiStoreFloat32(r2 + 4, d22d23);
        GiStoreFloat32(r2 + 8, d24d25);

        GiStoreFloat32(r3, d26d27);
        GiStoreFloat32(r3 + 4, d28d29);
        GiStoreFloat32(r3 + 8, d30d31);
    } else if (m_remain == 3) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);

        GiStoreFloat32(r2, d20d21);
        GiStoreFloat32(r2 + 4, d22d23);
        GiStoreFloat32(r2 + 8, d24d25);
    } else if (m_remain == 2) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);

        GiStoreFloat32(r1, d14d15);
        GiStoreFloat32(r1 + 4, d16d17);
        GiStoreFloat32(r1 + 8, d18d19);
    } else if (m_remain == 1) {
        GiStoreFloat32(r0, d8d9);
        GiStoreFloat32(r0 + 4, d10d11);
        GiStoreFloat32(r0 + 8, d12d13);
    }
}
    )";
    ss << activation_gen->GenIntrinsicInitFloat();
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenActivate",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenIntrinsicFloat(args[0], args[1]);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string kern4x4(TContext* ctx) {
    std::stringstream ss;
    bool with_bias = ctx->getAttrBool("with_bias");
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto activation_gen = create_activation_gener_instrinsic(nonline_mode);
    ss << R"(

static void kern_4x4(
        const float* packA, const float* packB, int K, float* output, int LDC,
        int m_remain, int n_remain, const float* bias_ptr) {
    const float* a_ptr = packA;
    const float* b_ptr = packB;
    int oddk = (K & 1);
    K = ((K + 1) / 2) - 1;
    float* r0 = output;
    float* r1 = r0 + LDC;
    float* r2 = r1 + LDC;
    float* r3 = r2 + LDC;
    size_t d_size = sizeof(float);

    GI_FLOAT32_t d0d1, d2d3, d4d5, d6d7, d8d9, d10d11, d12d13, d14d15;
    float tmp[4];
    )";
    if (with_bias) {
        ss << R"(
        float tmp_bias[4];
        memset(tmp_bias, 0, sizeof(float) * 4);
        memcpy(tmp_bias, bias_ptr, sizeof(float) * m_remain);
        d6d7 = GiLoadFloat32(tmp_bias);
        d8d9 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 0));
        d10d11 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetLowFloat32(d6d7), 1));
        d12d13 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 0));
        d14d15 = GiBroadcastFloat32(GiGetLaneFloat32(GiGetHighFloat32(d6d7), 1));
       
        )";
    } else {
        ss << R"(
        d8d9 = GiBroadcastFloat32(0.0f);
        d10d11 = GiBroadcastFloat32(0.0f);
        d12d13 = GiBroadcastFloat32(0.0f);
        d14d15 = GiBroadcastFloat32(0.0f);
        
    )";
    }
    std::string body_temp = R"(
    d0d1 = GiLoadFloat32(a_ptr);
    a_ptr = a_ptr + 4;
    d4d5 = GiLoadFloat32(b_ptr);
    b_ptr = b_ptr + 4;

    for (; K > 0; K--) {
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d4d5, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 1);
        d12d13 = GiSimdFmaLane(d12d13, d4d5, d0d1, 2);
        d14d15 = GiSimdFmaLane(d14d15, d4d5, d0d1, 3);

        d0d1 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d4d5 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d6d7, d2d3, 0);
        d10d11 = GiSimdFmaLane(d10d11, d6d7, d2d3, 1);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d2d3, 2);
        d14d15 = GiSimdFmaLane(d14d15, d6d7, d2d3, 3);
    }

    if (1 == oddk) {
        d8d9 = GiSimdFmaLane(d8d9, d4d5, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 1);
        d12d13 = GiSimdFmaLane(d12d13, d4d5, d0d1, 2);
        d14d15 = GiSimdFmaLane(d14d15, d4d5, d0d1, 3);

    } else {
        d2d3 = GiLoadFloat32(a_ptr);
        a_ptr = a_ptr + 4;
        d6d7 = GiLoadFloat32(b_ptr);
        b_ptr = b_ptr + 4;

        d8d9 = GiSimdFmaLane(d8d9, d4d5, d0d1, 0);
        d10d11 = GiSimdFmaLane(d10d11, d4d5, d0d1, 1);
        d12d13 = GiSimdFmaLane(d12d13, d4d5, d0d1, 2);
        d14d15 = GiSimdFmaLane(d14d15, d4d5, d0d1, 3);

        d8d9 = GiSimdFmaLane(d8d9, d6d7, d2d3, 0);
        d10d11 = GiSimdFmaLane(d10d11, d6d7, d2d3, 1);
        d12d13 = GiSimdFmaLane(d12d13, d6d7, d2d3, 2);
        d14d15 = GiSimdFmaLane(d14d15, d6d7, d2d3, 3);
    }
    ${GenActivate(d8d9, d8d9)}
    ${GenActivate(d10d11, d10d11)}
    ${GenActivate(d12d13, d12d13)}
    ${GenActivate(d14d15, d14d15)}
    if (m_remain >= 1) {
        GiStoreFloat32(tmp, d8d9);
        memcpy(r0, tmp, d_size * n_remain);
    }
    if (m_remain >=2) {
        GiStoreFloat32(tmp, d10d11);
        memcpy(r1, tmp, d_size * n_remain);
    }
    if (m_remain >= 3) {
        GiStoreFloat32(tmp, d12d13);
        memcpy(r2, tmp, d_size * n_remain);
    }
    if (m_remain == 4) {
        GiStoreFloat32(tmp, d14d15);
        memcpy(r3, tmp, d_size * n_remain);
    }
}
    )";
    ss << activation_gen->GenIntrinsicInitFloat();
    ss << StringTemplate::StringTemplateArgs()
                    .add("GenActivate",
                         [=](std::vector<std::string> args) {
                             return activation_gen->GenIntrinsicFloat(args[0], args[1]);
                         })
                    .render(body_temp);
    return ss.str();
}

std::string naked_kern(const std::string& sig, TContext* ctx) {
    std::stringstream ss;
    std::string post_process_str;
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    if (nonline_mode == "SIGMOID") {
        std::vector<CCOperand> operands;
        operands.resize(2);
        GIMathHelper gi_math;
        auto ElemwiseImpl =
                std::make_shared<ElemwiseGenUnarySigmoid>("f32", "f32", true);
        std::string post_process_temp = R"(
            if (LDC == N){
                ${ElemwiseImplName}(C, C, M * N);
            }else{
                for(int m_idx = 0; m_idx < M; ++m_idx){
                    ${ElemwiseImplName}(C + m_idx * LDC, C + m_idx * LDC, N);
                }
            }
        )";
        post_process_str =
                StringTemplate::StringTemplateArgs()
                        .add("ElemwiseImplName", ElemwiseImpl->GenInlineName())
                        .render(post_process_temp);
        ss << R"(
#include "gi_int.h"
            )";
        ss << gi_math.GiExpPsFloat32() << "\n";
        ss << gi_math.GiSigmoidPsFloat32() << "\n";
        ss << ElemwiseImpl->GenCodeBody({});
    }
    ss << sig;
    ss << R"({
    size_t m = 0;
    const int K12 = K * 12;
    const int K4 = K * 4;
    const size_t A_INTERLEAVE = 4;
    const size_t B_INTERLEAVE = 12;
    for (; m < M; m += A_INTERLEAVE) {
        float* output = C + (m * LDC);
        size_t n = 0;
        const float* cur_pack_b = pack_b;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(pack_a, cur_pack_b, K, output, LDC, 
                                  min(M - m, 4), bias_ptr);
            output += B_INTERLEAVE;
            cur_pack_b += K12;
        }

        for (; n < N; n += 4) {
            kern_4x4(pack_a, cur_pack_b, K, output, LDC, 
                                 min(M - m, 4),
                                 min(N - n, 4), bias_ptr);
            output += 4;
            cur_pack_b += K4;
        }
        pack_a += K4;
        bias_ptr += A_INTERLEAVE;
    }
 )";
    ss << post_process_str;
    ss << "\n}";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_m = 4;
        int k = kmax - k0;
        int m = ymax - y0;
        int round_m = (m + packed_m - 1) / packed_m * packed_m;
        size_t res = (size_t)k * round_m * sizeof(float);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        const int packed_n = 12;
        const size_t packed_hw = (xmax - x0 + packed_n - 1) / packed_n * packed_n;
        size_t res = (size_t)(kmax - k0) * packed_hw * sizeof(float);
        return res;
    })";
    return ss.str();
}

}  // namespace

std::string MatmulM4N12Kernel::GetKernelSymbol(TContext* ctx) const {
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string bias_suffix = with_bias ? "_bias" : "";
    std::string act_suffix = "";
    if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
        act_suffix = "_" + ctx->getAttrStr("nonlineMode");
    }
    return "GI_fp32_m4_n12_matmul" + bias_suffix + act_suffix;
}

std::string MatmulM4N12Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto kern_sym = GetKernelSymbol(ctx);
    writer << "#include \"gi_float.h\"\n";
    writer << "#include <string.h>\n";
    writer << utilsFunc();
    writer << interleave_helper();
    writer << interleave_1();
    writer << interleave_4();
    writer << interleave_1x4_1_s();
    writer << interleave_1x12_1_s();
    writer << interleave_4x12_1_s();
    writer << interleave_4x4_1_s();
    writer << transpose_4x4_1_s();
    writer << pack_A_n(kern_sym, ctx);
    writer << pack_A_t(kern_sym, ctx);
    writer << pack_B_n(kern_sym, ctx);
    writer << pack_B_t(kern_sym, ctx);
    writer << kern4x12(ctx);
    writer << kern4x4(ctx);
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << naked_kern(GetNakedKernelSignature(ctx), ctx);
    return writer.str();
}

std::string MatmulM4N12Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulM4N12Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
