#include "InternalKernel.h"
#include "compiler/Common/Logger.h"
#include "../../Utils/StringTemplate.h"
using namespace megcc;
using namespace KernelGen;
using namespace WebAssembly;
namespace {
std::string transpose_4x4_1_s(void) {
    return R"(
static inline void transpose_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int stride) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0);     //A0A1A2A3
    __m128 xmm1 = _mm_loadu_ps(inptr1); //B0B1B2B3
    __m128 xmm2 = _mm_loadu_ps(inptr2); //C0C1C2C3
    __m128 xmm3 = _mm_loadu_ps(inptr3); //D0D1D2D3

    __m128 xmm4 = _mm_unpacklo_ps(xmm0, xmm2); //A0C0A1C1
    __m128 xmm5 = _mm_unpackhi_ps(xmm0, xmm2); //A2C2A3C3
    __m128 xmm6 = _mm_unpacklo_ps(xmm1, xmm3); //B0D0B1D1
    __m128 xmm7 = _mm_unpackhi_ps(xmm1, xmm3); //B2D2B3D3

    xmm0 = _mm_unpacklo_ps(xmm4, xmm6); //A0B0C0D0
    xmm1 = _mm_unpackhi_ps(xmm4, xmm6); //A1B1C1D1
    xmm2 = _mm_unpacklo_ps(xmm5, xmm7); //A2B2C2D2
    xmm3 = _mm_unpackhi_ps(xmm5, xmm7); //A3B3C3D3

    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm1);
    _mm_storeu_ps(outptr + 8, xmm2);
    _mm_storeu_ps(outptr + 12, xmm3);
#else
    #error "SSE not supported."
#endif
}
    )";
}

std::string interleave(void) {
    return R"(

static inline void interleave_helper(
        const float* inptr, float* outptr, int unroll_k, int ksize, float val) {
    int k = 0;
    for (; k < ksize; k++) {
        *outptr++ = *inptr++;
    }
    for (; k < unroll_k; k++) {
        *outptr++ = val;
    }
}
static inline void interleave_1(
        const float* inptr0, float* outptr, int unroll_k, int ksize, float val) {
    for (int k = 0; k < ksize; k += unroll_k) {
        int size = unroll_k > (ksize - k)? (ksize - k) : unroll_k;
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size;outptr+=unroll_k;
    }
}

static inline void interleave_4(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr, int unroll_k, int ksize, float val) {
     for (int k = 0; k < ksize; k += unroll_k) {
        int size = unroll_k > (ksize - k)? (ksize - k) : unroll_k;
        interleave_helper(inptr0, outptr, unroll_k, size, val);
        inptr0 += size;outptr+=unroll_k;
        interleave_helper(inptr1, outptr, unroll_k, size, val);
        inptr1 += size;outptr+=unroll_k;
        interleave_helper(inptr2, outptr, unroll_k, size, val);
        inptr2 += size;outptr+=unroll_k;
        interleave_helper(inptr3, outptr, unroll_k, size, val);
        inptr3 += size;outptr+=unroll_k;
    }
}

    )";
}

static std::string interleave_4x12_1_s(void) {
    return R"(
static inline void interleave_4x12_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0); 
    __m128 xmm1 = _mm_loadu_ps(inptr0 + 4); 
    __m128 xmm2 = _mm_loadu_ps(inptr0 + 8);
    
    __m128 xmm3 = _mm_loadu_ps(inptr1); 
    __m128 xmm4 = _mm_loadu_ps(inptr1 + 4); 
    __m128 xmm5 = _mm_loadu_ps(inptr1 + 8);
    
    __m128 xmm6 = _mm_loadu_ps(inptr2); 
    __m128 xmm7 = _mm_loadu_ps(inptr2 + 4); 
    __m128 xmm8 = _mm_loadu_ps(inptr2 + 8);

    __m128 xmm9 = _mm_loadu_ps(inptr3); 
    __m128 xmm10 = _mm_loadu_ps(inptr3 + 4); 
    __m128 xmm11 = _mm_loadu_ps(inptr3 + 8);

    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm1);
    _mm_storeu_ps(outptr + 8, xmm2);
    _mm_storeu_ps(outptr + 12, xmm3);
    _mm_storeu_ps(outptr + 16, xmm4);
    _mm_storeu_ps(outptr + 20, xmm5);
    _mm_storeu_ps(outptr + 24, xmm6);
    _mm_storeu_ps(outptr + 28, xmm7);
    _mm_storeu_ps(outptr + 32, xmm8);
    _mm_storeu_ps(outptr + 36, xmm9);
    _mm_storeu_ps(outptr + 40, xmm10);
    _mm_storeu_ps(outptr + 44, xmm11);
#else
    #error "SSE not supported."
#endif
}

static inline void interleave_1x12_1_s(const float* inptr0, float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0); 
    __m128 xmm1 = _mm_loadu_ps(inptr0 + 4); 
    __m128 xmm2 = _mm_loadu_ps(inptr0 + 8);

    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm1);
    _mm_storeu_ps(outptr + 8, xmm2);
#else
    #error "SSE not supported."
#endif
}
    )";
}

static std::string interleave_4x4_1_s(void) {
    return R"(
static inline void interleave_4x4_1_s(
        const float* inptr0, const float* inptr1, const float* inptr2, const float* inptr3,
        float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0); 
    __m128 xmm1 = _mm_loadu_ps(inptr1); 
    __m128 xmm2 = _mm_loadu_ps(inptr2);
    __m128 xmm3 = _mm_loadu_ps(inptr3);
  
    _mm_storeu_ps(outptr, xmm0);
    _mm_storeu_ps(outptr + 4, xmm1);
    _mm_storeu_ps(outptr + 8, xmm2);
    _mm_storeu_ps(outptr + 12, xmm3);
#else
    #error "SSE not supported."
#endif
}

static inline void interleave_1x4_1_s(const float* inptr0, float* outptr) {
#ifdef __SSE__
    __m128 xmm0 = _mm_loadu_ps(inptr0); 
    _mm_storeu_ps(outptr, xmm0);
#else
    #error "SSE not supported."
#endif
}
    )";
}

static std::string kern_4x12(TContext* ctx) {
    std::stringstream writer;
    // TODO:with bias not implemented
    writer << R"(static void kern_4x12(const float* packA, const float* packB, int K,
                          float* output, int LDC, 
                          int m_remain, const float* bias_ptr) {
#ifdef __WASM_SIMD128_H
    v128_t xmm0, xmm1, xmm2, xmm3;
    /*Res*/
    v128_t xmm4, xmm5, xmm6, xmm7;
    v128_t xmm8, xmm9, xmm10, xmm11;
    v128_t xmm12, xmm13, xmm14, xmm15;

    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output;

    for (size_t k = 0; k < K; k+=4) {
        if (k == 0) {
            xmm4 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm5 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm6 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm7 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm8 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm9 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm10 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm11 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm12 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm13 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm14 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm15 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
        }
        int remain = K - k;
        if (remain > 4) {
            remain = 4;
        }
        b_ptr += (remain - 1) * 12;
        switch (remain) {
            case 4:
                xmm1 = wasm_v128_load(b_ptr);
                xmm2 = wasm_v128_load(b_ptr + 4);
                xmm0 = wasm_f32x4_splat(*(a_ptr+12));
                xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
                xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
                xmm3 = wasm_v128_load(b_ptr + 8); 
                xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+13));
                xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+14));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
                xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+15));
                xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
                xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
                xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
                b_ptr -= 12;
            case 3:
                xmm1 = wasm_v128_load(b_ptr);
                xmm0 = wasm_f32x4_splat(*(a_ptr+8));
                xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
                xmm2 = wasm_v128_load(b_ptr + 4);
                xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
                xmm3 = wasm_v128_load(b_ptr + 8); 
                xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+9));
                xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+10));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
                xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+11));
                xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
                xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
                xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
                b_ptr -= 12;
            case 2:
                xmm1 = wasm_v128_load(b_ptr);
                xmm0 = wasm_f32x4_splat(*(a_ptr+4));
                xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
                xmm2 = wasm_v128_load(b_ptr + 4);
                xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
                xmm3 = wasm_v128_load(b_ptr + 8); 
                xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+5));

                xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+6));
                
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
                xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+7));

                xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
                xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
                xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
                b_ptr -= 12;
            case 1:
                xmm1 = wasm_v128_load(b_ptr);
                xmm2 = wasm_v128_load(b_ptr + 4);
                xmm3 = wasm_v128_load(b_ptr + 8); 
                xmm0 = wasm_f32x4_splat(*(a_ptr+0));
                xmm4 = wasm_f32x4_add(xmm4, wasm_f32x4_mul(xmm0, xmm1));
                xmm5 = wasm_f32x4_add(xmm5, wasm_f32x4_mul(xmm0, xmm2));
                xmm6 = wasm_f32x4_add(xmm6, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+1));
                xmm7 = wasm_f32x4_add(xmm7, wasm_f32x4_mul(xmm0, xmm1));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm2));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+2));
                
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm1));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm2));
                xmm12 = wasm_f32x4_add(xmm12, wasm_f32x4_mul(xmm0, xmm3));
                xmm0 = wasm_f32x4_splat(*(a_ptr+3));

                xmm13 = wasm_f32x4_add(xmm13, wasm_f32x4_mul(xmm0, xmm1));
                xmm14 = wasm_f32x4_add(xmm14, wasm_f32x4_mul(xmm0, xmm2));
                xmm15 = wasm_f32x4_add(xmm15, wasm_f32x4_mul(xmm0, xmm3));
                break;
            default:;
        }
    
        b_ptr += 48;
        a_ptr += 16; 
    }
    
    // store

    switch (m_remain) {
        case 4:
            wasm_v128_store(output + LDC * 3 + 0, xmm13);
            wasm_v128_store(output + LDC * 3 + 4, xmm14);
            wasm_v128_store(output + LDC * 3 + 8, xmm15);
        case 3:
            wasm_v128_store(output + LDC * 2 + 0, xmm10);
            wasm_v128_store(output + LDC * 2 + 4, xmm11);
            wasm_v128_store(output + LDC * 2 + 8, xmm12);
        case 2:
            wasm_v128_store(output + LDC + 0, xmm7);
            wasm_v128_store(output + LDC + 4, xmm8);
            wasm_v128_store(output + LDC + 8, xmm9);
        case 1:
            wasm_v128_store(output + 0, xmm4);
            wasm_v128_store(output + 4, xmm5);
            wasm_v128_store(output + 8, xmm6);
            break;
        default:;
    }
#else
    #error "WebAssembly not supported."
#endif     

}
    )";
    return writer.str();
}

static std::string kern_4x4(TContext* crx) {
    std::stringstream writer;
    writer << R"(static void kern_4x4(const float* packA, const float* packB, int K,
                          float* output, int LDC, 
                          int m_remain, int n_remain, const float* bias_ptr) {
#ifdef __WASM_SIMD128_H
    v128_t xmm0, xmm1, xmm2, xmm3;
    v128_t xmm4, xmm5, xmm6, xmm7;
    /*Res*/
    v128_t xmm8, xmm9, xmm10, xmm11;

    const float* a_ptr = packA;
    const float* b_ptr = packB;
    float* output0 = output; 
    for (size_t k = 0; k < K; k+=4) {
        if (k == 0) {
            xmm8 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm9 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm10 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
            xmm11 = wasm_f32x4_make(0.0f, 0.0f, 0.0f, 0.0f);
        }
        // line 1
    
        int remain = K - (k + 1);
        if (remain > 4){
            remain = 4;
        }
        switch(remain) {
            case 4:
                xmm3 = wasm_v128_load(b_ptr + 12);
                xmm4 = wasm_f32x4_splat(*(a_ptr+12));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm3, xmm4));
                xmm5 = wasm_f32x4_splat(*(a_ptr+13));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm3, xmm5));
                xmm6 = wasm_f32x4_splat(*(a_ptr+14));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm3, xmm6));
                xmm7 = wasm_f32x4_splat(*(a_ptr+15));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm3, xmm7));
            case 3:
                xmm2 = wasm_v128_load(b_ptr + 8);
                xmm4 = wasm_f32x4_splat(*(a_ptr+8));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm2, xmm4));
                xmm5 = wasm_f32x4_splat(*(a_ptr+9));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm2, xmm5));
                xmm6 = wasm_f32x4_splat(*(a_ptr+10));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm2, xmm6));
                xmm7 = wasm_f32x4_splat(*(a_ptr+11));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm2, xmm7));
            case 2:
                xmm1 = wasm_v128_load(b_ptr + 4);
                xmm4 = wasm_f32x4_splat(*(a_ptr+4));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm1, xmm4));            
                xmm5 = wasm_f32x4_splat(*(a_ptr+5));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm1, xmm5));
                xmm6 = wasm_f32x4_splat(*(a_ptr+6));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm1, xmm6));
                xmm7 = wasm_f32x4_splat(*(a_ptr+7));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm1, xmm7));
            case 1:
                xmm0 = wasm_v128_load(b_ptr);
                xmm4 = wasm_f32x4_splat(*(a_ptr+0));
                xmm8 = wasm_f32x4_add(xmm8, wasm_f32x4_mul(xmm0, xmm4));
                xmm5 = wasm_f32x4_splat(*(a_ptr+1));
                xmm9 = wasm_f32x4_add(xmm9, wasm_f32x4_mul(xmm0, xmm5));
                xmm6 = wasm_f32x4_splat(*(a_ptr+2));
                xmm10 = wasm_f32x4_add(xmm10, wasm_f32x4_mul(xmm0, xmm6));
                xmm7 = wasm_f32x4_splat(*(a_ptr+3));
                xmm11 = wasm_f32x4_add(xmm11, wasm_f32x4_mul(xmm0, xmm7));
                break;
            default:;

        }
        
        a_ptr += 16;
        b_ptr += 16;
    }

    float dst[4 * 4];
    wasm_v128_store(dst + 0, xmm8);
    wasm_v128_store(dst + 4, xmm9);
    wasm_v128_store(dst + 8, xmm10);
    wasm_v128_store(dst + 12, xmm11);
    
    for (int i = 0; i < n_remain; i++) {
        for (int j = 0; j < m_remain; j++) {
            output[LDC * j + i] = dst[4 * j + i];
        }
    }

#else
    #error "WebAssembly not supported."
#endif


})";
    return writer.str();
}

std::string pack_A_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packa_n" +
        WebAssemblyMatmulInternal::GenPackACall(ctx) +
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

std::string pack_B_n(const std::string kern_sym, TContext* ctx) {
    return "void " + kern_sym + "_packb_n" +
           WebAssemblyMatmulInternal::GenPackBCall(ctx) +
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

std::string gen_kernel(
        const std::string& sig, TContext* ctx) {
    std::stringstream ss;
    std::string post_process_str;
    ss << sig;
    ss << R"({
    size_t m = 0;
    const int K12 = K * 12;
    const int K4 = K * 4;
    const size_t A_INTERLEAVE = 4;
    const size_t B_INTERLEAVE = 12;

    for(;m <= M;m += A_INTERLEAVE){
        float* output = C + (m * LDC);
        size_t n = 0;
        const float* cur_pack_b = pack_b;
        for (; n + B_INTERLEAVE - 1 < N; n += B_INTERLEAVE) {
            kern_4x12(pack_a, cur_pack_b, K, output, LDC, 
                                  (M - m) > 4 ? 4 : (M - m), bias_ptr);
            output += B_INTERLEAVE;
            cur_pack_b += K12;
        }

        for (; n < N; n += 4) {
            kern_4x4(pack_a, cur_pack_b, K, output, LDC, 
                                 (M - m) > 4 ? 4 : (M - m),
                                 (N - n) > 4 ? 4 : (N - n), bias_ptr);
            output += 4;
            cur_pack_b += K4;
        }
        pack_a += K4;
        bias_ptr += A_INTERLEAVE;
    }
    }
    )";
    return ss.str();
}

} //namespace

std::string MatmulM4N12Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "WebAssembly_fp32_m4_n12_matmul";
    if (ctx->getAttrBool("with_bias")) {
        ss << "_bias";
    }
    // if (ctx->haveAttr("nonlineMode") && ctx->getAttrStr("nonlineMode") != "IDENTITY") {
    //     ss << "_" << ctx->getAttrStr("nonlineMode");
    // }
    return ss.str();
}

std::string MatmulM4N12Kernel::GetKernelBody(TContext* ctx) const {
   
    std::stringstream writer;
    auto kern_sym = GetKernelSymbol(ctx);
    writer << "#include <string.h>\n";
    writer << "#include <xmmintrin.h>\n";
    writer << "#include <wasm_simd128.h>\n";
    writer << interleave();
    writer << interleave_4x12_1_s();
    writer << interleave_4x4_1_s();
    writer << transpose_4x4_1_s();
    writer << kern_4x12(ctx);
    writer << kern_4x4(ctx);
    writer << pack_A_n(kern_sym, ctx);
    writer << pack_B_n(kern_sym, ctx);
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
    writer << gen_kernel(GetNakedKernelSignature(ctx), ctx);

    return writer.str();
}


std::string MatmulM4N12Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}

std::string MatmulM4N12Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}