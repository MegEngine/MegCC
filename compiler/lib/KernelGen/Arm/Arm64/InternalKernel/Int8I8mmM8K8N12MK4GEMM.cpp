#include "Arm/Arm64/Activation.h"
#include "Arm/ArmCommon/MatmulCommon.h"
#include "Arm/ArmCommon/common_asm_utils.h"
#include "InternalKernel.h"
#include "Utils/StringTemplate.h"
#include "Utils/Utils.h"
#include "compiler/Common/Logger.h"
using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;

namespace {
//! (M, K, 4(K), 4(M)) --> (M/2, K/2, 8(M), 2(K), 4(K))
std::string gen_pack_a(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
    const uint32_t LDA = ldin;
    int8_t *dst = outptr;
    const int8_t *src = inptr;
    TINYNN_ASSERT((ymax - y0) % 4 == 0);
    TINYNN_ASSERT((kmax - k0) % 4 == 0);
    const int M = (ymax - y0) / 4;
    const int K = (kmax - k0) / 4;
    int8_t idx_tbl[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    int8x16_t vidx_tbl = vld1q_s8(idx_tbl);
    int m = 0;
    for (; m + 1 < M; m += 2) {
        int k = 0;
        for (; k + 3 < K; k += 4) {
            int8x16x4_t row0 = vld4q_s8(src);
            int8x16x4_t row1 = vld4q_s8(src + LDA);
            src += 64;

            int8x16_t a0 = vtrn1q_s64(row0.val[0], row0.val[1]);
            vst1q_s8(dst, a0);
            dst += 16;
            int8x16_t a1 = vtrn1q_s64(row0.val[2], row0.val[3]);
            vst1q_s8(dst, a1);
            dst += 16;
            int8x16_t a2 = vtrn1q_s64(row1.val[0], row1.val[1]);
            vst1q_s8(dst, a2);
            dst += 16;
            int8x16_t a3 = vtrn1q_s64(row1.val[2], row1.val[3]);
            vst1q_s8(dst, a3);
            dst += 16;

            a0 = vtrn2q_s64(row0.val[0], row0.val[1]);
            vst1q_s8(dst, a0);
            dst += 16;
            a1 = vtrn2q_s64(row0.val[2], row0.val[3]);
            vst1q_s8(dst, a1);
            dst += 16;
            a2 = vtrn2q_s64(row1.val[0], row1.val[1]);
            vst1q_s8(dst, a2);
            dst += 16;
            a3 = vtrn2q_s64(row1.val[2], row1.val[3]);
            vst1q_s8(dst, a3);
            dst += 16;
        }
        if (k + 1 < K) {
            int8x8x4_t row0 = vld4_s8(src);
            vst1_s8(dst, row0.val[0]);
            dst += 8;
            vst1_s8(dst, row0.val[1]);
            dst += 8;
            int8x8x4_t row1 = vld4_s8(src + LDA);
            vst1_s8(dst, row0.val[2]);
            dst += 8;
            vst1_s8(dst, row0.val[3]);
            dst += 8;
            vst1_s8(dst, row1.val[0]);
            dst += 8;
            vst1_s8(dst, row1.val[1]);
            dst += 8;
            vst1_s8(dst, row1.val[2]);
            dst += 8;
            vst1_s8(dst, row1.val[3]);
            dst += 8;

            src += 32;
            k += 2;
        }
        if (k < K) {
            memset(dst, 0, 64);
            int8x16_t row0 = vld1q_s8(src);
            int8x16_t row1 = vld1q_s8(src + LDA);
            int8x8x2_t src_tbl;
            src_tbl.val[0] = vget_low_s8(row0);
            src_tbl.val[1] = vget_high_s8(row0);
            int8x8_t low = vtbl2_s8(src_tbl, vget_low_s8(vidx_tbl));
            int8x8_t high = vtbl2_s8(src_tbl, vget_high_s8(vidx_tbl));
            vst1_lane_s32(dst, low, 0);
            dst += 8;
            vst1_lane_s32(dst, low, 1);
            dst += 8;
            vst1_lane_s32(dst, high, 0);
            dst += 8;
            vst1_lane_s32(dst, high, 1);
            dst += 8;
            src_tbl.val[0] = vget_low_s8(row1);
            src_tbl.val[1] = vget_high_s8(row1);
            int8x8_t low1 = vtbl2_s8(src_tbl, vget_low_s8(vidx_tbl));
            int8x8_t high1 = vtbl2_s8(src_tbl, vget_high_s8(vidx_tbl));
            vst1_lane_s32(dst, low1, 0);
            dst += 8;
            vst1_lane_s32(dst, low1, 1);
            dst += 8;
            vst1_lane_s32(dst, high1, 0);
            dst += 8;
            vst1_lane_s32(dst, high1, 1);
            dst += 8;

            src += 16;
        }
        src += LDA;
    }
    if (m < M) {
        memset(dst, 0, ROUND_UP(K, 2) * 4 * 8);
        int k = 0;
        for (; k + 3 < K; k += 4) {
            int8x16x4_t row0 = vld4q_s8(src);
            src += 64;

            int8x16_t a0 = vtrn1q_s64(row0.val[0], row0.val[1]);
            vst1q_s8(dst, a0);
            dst += 16;
            int8x16_t a1 = vtrn1q_s64(row0.val[2], row0.val[3]);
            vst1q_s8(dst, a1);
            dst += 48;

            a0 = vtrn2q_s64(row0.val[0], row0.val[1]);
            vst1q_s8(dst, a0);
            dst += 16;
            a1 = vtrn2q_s64(row0.val[2], row0.val[3]);
            vst1q_s8(dst, a1);
            dst += 48;
        }
        if (k + 1 < K) {
            int8x8x4_t row0 = vld4_s8(src);
            vst1_s8(dst, row0.val[0]);
            dst += 8;
            vst1_s8(dst, row0.val[1]);
            dst += 8;
            int8x8x4_t row1 = vld4_s8(src + LDA);
            vst1_s8(dst, row0.val[2]);
            dst += 8;
            vst1_s8(dst, row0.val[3]);
            dst += 40;

            src += 32;
            k += 2;
        }
        if (k < K) {
            int8x16_t row0 = vld1q_s8(src);
            int8x8x2_t src_tbl;
            src_tbl.val[0] = vget_low_s8(row0);
            src_tbl.val[1] = vget_high_s8(row0);
            int8x8_t low = vtbl2_s8(src_tbl, vget_low_s8(vidx_tbl));
            int8x8_t high = vtbl2_s8(src_tbl, vget_high_s8(vidx_tbl));
            vst1_lane_s32(dst, low, 0);
            dst += 8;
            vst1_lane_s32(dst, low, 1);
            dst += 8;
            vst1_lane_s32(dst, high, 0);
            dst += 8;
            vst1_lane_s32(dst, high, 1);
            dst += 40;

            src += 16;
        }
    }
})";
    return ss.str();
}

std::string gen_pack_b(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
#define N_block_12 12
#define N_block_8 8
#define N_block_4 4
  const uint32_t LDB = ldin;
  TINYNN_ASSERT((kmax - k0) % 4 == 0);
  const int K = (kmax - k0) / 4;
  const int N = xmax - x0;
  int8_t *dst = outptr;
  const int8_t *src = inptr;

  int8_t *dst_ptr = dst;
  int8_t *dst_ptr_n8 =
      dst + N / N_block_12 * N_block_12 * (ROUND_UP(K, 2) * 4);
  int8_t *dst_ptr_n4 = dst_ptr_n8 + (N % N_block_12) / N_block_8 * N_block_8 *
                                        (ROUND_UP(K, 2) * 4);
  int k = 0;
  for (; k + 1 < K; k += 2) {
    int8_t *dst_base = dst_ptr;
    int8_t *dst_base_n8 = dst_ptr_n8;
    int8_t *dst_base_n4 = dst_ptr_n4;
    int n = 0;
    for (; n + N_block_12 - 1 < N; n += N_block_12) {
      int8x16_t t0 = vld1q_s8(src);
      int8x16_t t1 = vld1q_s8(src + LDB);
      int32x4x2_t d01 = vzipq_s32(t0, t1);
      vst1q_s8(dst_base, d01.val[0]);
      vst1q_s8(dst_base + 16, d01.val[1]);

      int8x16_t t2 = vld1q_s8(src + 16);
      int8x16_t t3 = vld1q_s8(src + LDB + 16);
      int32x4x2_t d23 = vzipq_s32(t2, t3);
      vst1q_s8(dst_base + 32, d23.val[0]);
      vst1q_s8(dst_base + 48, d23.val[1]);

      int8x16_t t4 = vld1q_s8(src + 32);
      int8x16_t t5 = vld1q_s8(src + LDB + 32);
      int32x4x2_t d45 = vzipq_s32(t4, t5);
      vst1q_s8(dst_base + 64, d45.val[0]);
      vst1q_s8(dst_base + 80, d45.val[1]);

      src += 48;
      dst_base += N_block_12 * 4 * ROUND_UP(K, 2);
    }
    for (; n + N_block_8 - 1 < N; n += N_block_8) {
      int8x16_t t0 = vld1q_s8(src);
      int8x16_t t1 = vld1q_s8(src + LDB);
      src += 16;
      int32x4x2_t d01 = vzipq_s32(t0, t1);
      vst1q_s8(dst_base_n8, d01.val[0]);
      vst1q_s8(dst_base_n8 + 16, d01.val[1]);

      int8x16_t t2 = vld1q_s8(src);
      int8x16_t t3 = vld1q_s8(src + LDB);
      src += 16;
      int32x4x2_t d23 = vzipq_s32(t2, t3);
      vst1q_s8(dst_base_n8 + 32, d23.val[0]);
      vst1q_s8(dst_base_n8 + 48, d23.val[1]);

      dst_base_n8 += N_block_8 * 4 * ROUND_UP(K, 2);
    }
    for (; n + N_block_4 - 1 < N; n += N_block_4) {
      int8x16_t t0 = vld1q_s8(src);
      int8x16_t t1 = vld1q_s8(src + LDB);
      src += 16;
      int32x4x2_t d01 = vzipq_s32(t0, t1);
      vst1q_s8(dst_base_n4, d01.val[0]);
      vst1q_s8(dst_base_n4 + 16, d01.val[1]);

      dst_base_n4 += N_block_4 * 4 * ROUND_UP(K, 2);
    }
    if (n < N) {
      memset(dst_base_n4, 0, 32);
      for (int i = 0; i < N - n; ++i) {
        int8x8_t t0 = vld1_dup_s32(src);
        int8x8_t t1 = vld1_dup_s32(src + LDB);
        src += 4;
        int8x8_t d0 = vzip1_s32(t0, t1);
        vst1_s8(dst_base_n4 + i * 8, d0);
      }

      dst_base_n4 += N_block_4 * 4 * ROUND_UP(K, 2);
    }
    dst_ptr += 8 * N_block_12;
    dst_ptr_n8 += 8 * N_block_8;
    dst_ptr_n4 += 8 * N_block_4;
    src += LDB;
  }
  if (k < K) {
    int8x16_t zero = vdupq_n_s8(0);
    int8x8_t zero_half = vdup_n_s8(0);
    int8_t *dst_base = dst_ptr;
    int8_t *dst_base_n8 = dst_ptr_n8;
    int8_t *dst_base_n4 = dst_ptr_n4;
    int n = 0;
    for (; n + N_block_12 - 1 < N; n += N_block_12) {
      int8x16_t t0 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d01 = vzipq_s32(t0, zero);
      vst1q_s8(dst_base, d01.val[0]);
      vst1q_s8(dst_base + 16, d01.val[1]);

      int8x16_t t2 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d23 = vzipq_s32(t2, zero);
      vst1q_s8(dst_base + 32, d23.val[0]);
      vst1q_s8(dst_base + 48, d23.val[1]);

      int8x16_t t4 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d45 = vzipq_s32(t4, zero);
      vst1q_s8(dst_base + 64, d45.val[0]);
      vst1q_s8(dst_base + 80, d45.val[1]);

      dst_base += N_block_12 * 4 * ROUND_UP(K, 2);
    }
    for (; n + N_block_8 - 1 < N; n += N_block_8) {
      int8x16_t t0 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d01 = vzipq_s32(t0, zero);
      vst1q_s8(dst_base_n8, d01.val[0]);
      vst1q_s8(dst_base_n8 + 16, d01.val[1]);

      int8x16_t t2 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d23 = vzipq_s32(t2, zero);
      vst1q_s8(dst_base_n8 + 32, d23.val[0]);
      vst1q_s8(dst_base_n8 + 48, d23.val[1]);

      dst_base_n8 += N_block_8 * 4 * ROUND_UP(K, 2);
    }
    for (; n + N_block_4 < N; n += N_block_4) {
      int8x16_t t0 = vld1q_s8(src);
      src += 16;
      int32x4x2_t d01 = vzipq_s32(t0, zero);
      vst1q_s8(dst_base_n4, d01.val[0]);
      vst1q_s8(dst_base_n4 + 16, d01.val[1]);

      dst_base_n4 += N_block_4 * 4 * ROUND_UP(K, 2);
    }
    if (n < N) {
      memset(dst_base_n4, 0, 32);
      for (int i = 0; i < N - n; ++i) {
        memcpy(dst_base_n4 + i * 8, src + i * 4, 4);
      }
      src += 4 * (N - n);

      dst_base_n4 += N_block_4 * 4 * ROUND_UP(K, 2);
    }
    dst_ptr += 8 * N_block_12;
    dst_ptr_n8 += 8 * N_block_8;
    dst_ptr_n4 += 8 * N_block_4;
  }
#undef N_block_12
#undef N_block_8
#undef N_block_4
    })";
    return ss.str();
}

std::string gen_pack_a_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)((kmax - k0 + 7) / 8 * 8) * ((ymax - y0 + 7) / 8 * 8) * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_pack_b_workspace(const std::string& sig) {
    std::stringstream ss;
    ss << sig;
    ss << R"({
        size_t res = (size_t)((kmax - k0 + 7) / 8 * 8) * ((xmax - x0 + 3) / 4 * 4) * sizeof(int8_t);
        return res;
    })";
    return ss.str();
}

std::string gen_kernel(
        const std::string& dst_specifier, const std::string& sig, TContext* ctx,
        const std::string& preset_str = "", bool need_postprocess = false) {
    std::string nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    auto nonline_gen = create_activation_gener_instrinsic(nonline_mode);
    auto nonline_gen_func = [&](std::vector<std::string> str) -> std::string {
        return nonline_gen->GenIntrinsicQuantStore(str[0], str[1], str[2], str[3]);
    };
    auto nonline_gen_init = [&]() -> std::string {
        return nonline_gen->GenIntrinsicInitFloat();
    };
    bool with_bias = ctx->getAttrBool("with_bias");
    std::string kern_body =
            R"(
    ${kernel_sig}{
        ${preset_str}
        ${nonline_gen_init()}
#define N_block_12 12
#define N_block_8 8
#define N_block_4 4
  TINYNN_ASSERT(M % 4 == 0);
  TINYNN_ASSERT(K % 4 == 0);
  K /= 4;
  K = ROUND_UP(K, 2);
  M /= 4;
  ${dst_specifier} *C0 = C;
  const int8_t *A0 = pack_a, *A_ptr = NULL;
  const int8_t *B0 = pack_b, *B_ptr = NULL;
  uint32_t m = 0;
  for (; m + 1 < M; m += 2) {
    B_ptr = B0;
    C = C0 + m * LDC;
    int n = 0;
    for (; n + N_block_12 - 1 < N; n += N_block_12) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[2][N_block_12];
      )" +
            (with_bias ? std::string(R"(
      c[0][2] = vld1q_s32(bias_ptr + m * 4);
      c[0][0] = vzip1q_s64(c[0][2], c[0][2]);
      c[0][1] = vzip2q_s64(c[0][2], c[0][2]);
      c[0][2] = c[0][0];
      c[0][3] = c[0][1];
      c[0][4] = c[0][0];
      c[0][5] = c[0][1];
      c[0][6] = c[0][0];
      c[0][7] = c[0][1];
      c[0][8] = c[0][0];
      c[0][9] = c[0][1];
      c[0][10] = c[0][0];
      c[0][11] = c[0][1];

      c[1][2] = vld1q_s32(bias_ptr + (m + 1) * 4);
      c[1][0] = vzip1q_s64(c[1][2], c[1][2]);
      c[1][1] = vzip2q_s64(c[1][2], c[1][2]);
      c[1][2] = c[1][0];
      c[1][3] = c[1][1];
      c[1][4] = c[1][0];
      c[1][5] = c[1][1];
      c[1][6] = c[1][0];
      c[1][7] = c[1][1];
      c[1][8] = c[1][0];
      c[1][9] = c[1][1];
      c[1][10] = c[1][0];
      c[1][11] = c[1][1];
      )")
                       : std::string(R"(
      c[0][0] = vdupq_n_s32(0);
      c[0][1] = c[0][0];
      c[0][2] = c[0][0];
      c[0][3] = c[0][0];
      c[0][4] = c[0][0];
      c[0][5] = c[0][0];
      c[0][6] = c[0][0];
      c[0][7] = c[0][0];
      c[0][8] = c[0][0];
      c[0][9] = c[0][0];
      c[0][10] = c[0][0];
      c[0][11] = c[0][0];

      c[1][0] = c[0][0];
      c[1][1] = c[0][0];
      c[1][2] = c[0][0];
      c[1][3] = c[0][0];
      c[1][4] = c[0][0];
      c[1][5] = c[0][0];
      c[1][6] = c[0][0];
      c[1][7] = c[0][0];
      c[1][8] = c[0][0];
      c[1][9] = c[0][0];
      c[1][10] = c[0][0];
      c[1][11] = c[0][0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        int8x16_t b2_3 = vld1q_s8(B_ptr + 16);

        c[0][0] = vmmlaq_s32(c[0][0], b0_1, a0_1); //! c00 c10 c01 c11
        int8x16_t b4_5 = vld1q_s8(B_ptr + 32);
        c[0][2] = vmmlaq_s32(c[0][2], b2_3, a0_1); //! c02 c12 c03 c13
        int8x16_t b6_7 = vld1q_s8(B_ptr + 48);
        c[0][4] = vmmlaq_s32(c[0][4], b4_5, a0_1);
        int8x16_t b8_9 = vld1q_s8(B_ptr + 64);
        c[0][6] = vmmlaq_s32(c[0][6], b6_7, a0_1);
        int8x16_t b10_11 = vld1q_s8(B_ptr + 80);
        c[0][8] = vmmlaq_s32(c[0][8], b8_9, a0_1);
        int8x16_t a2_3 = vld1q_s8(A_ptr + 16);
        c[0][10] = vmmlaq_s32(c[0][10], b10_11, a0_1);

        a0_1 = vld1q_s8(A_ptr + 32);

        c[0][1] = vmmlaq_s32(c[0][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[0][3] = vmmlaq_s32(c[0][3], b2_3, a2_3); //! c22 c32 c23 c33
        c[0][5] = vmmlaq_s32(c[0][5], b4_5, a2_3);
        c[0][7] = vmmlaq_s32(c[0][7], b6_7, a2_3);
        c[0][9] = vmmlaq_s32(c[0][9], b8_9, a2_3);
        c[0][11] = vmmlaq_s32(c[0][11], b10_11, a2_3);

        a2_3 = vld1q_s8(A_ptr + 48);
        A_ptr += 64;
        B_ptr += 96;

        c[1][0] = vmmlaq_s32(c[1][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[1][2] = vmmlaq_s32(c[1][2], b2_3, a0_1); //! c02 c12 c03 c13
        c[1][4] = vmmlaq_s32(c[1][4], b4_5, a0_1);
        c[1][6] = vmmlaq_s32(c[1][6], b6_7, a0_1);
        c[1][8] = vmmlaq_s32(c[1][8], b8_9, a0_1);
        c[1][10] = vmmlaq_s32(c[1][10], b10_11, a0_1);

        c[1][1] = vmmlaq_s32(c[1][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[1][3] = vmmlaq_s32(c[1][3], b2_3, a2_3); //! c22 c32 c23 c33
        c[1][5] = vmmlaq_s32(c[1][5], b4_5, a2_3);
        c[1][7] = vmmlaq_s32(c[1][7], b6_7, a2_3);
        c[1][9] = vmmlaq_s32(c[1][9], b8_9, a2_3);
        c[1][11] = vmmlaq_s32(c[1][11], b10_11, a2_3);
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}
      int32x4_t c4 = vtrn1q_s64(c[0][4], c[0][5]);
      int32x4_t c5 = vtrn2q_s64(c[0][4], c[0][5]);
      ${nonline_gen_func(c4, C + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + 20, temp_scale, dst_scale_inv)}
      int32x4_t c6 = vtrn1q_s64(c[0][6], c[0][7]);
      int32x4_t c7 = vtrn2q_s64(c[0][6], c[0][7]);
      ${nonline_gen_func(c6, C + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + 28, temp_scale, dst_scale_inv)}
      c0 = vtrn1q_s64(c[0][8], c[0][9]);
      c1 = vtrn2q_s64(c[0][8], c[0][9]);
      ${nonline_gen_func(c0, C + 32, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 36, temp_scale, dst_scale_inv)}
      c2 = vtrn1q_s64(c[0][10], c[0][11]);
      c3 = vtrn2q_s64(c[0][10], c[0][11]);
      ${nonline_gen_func(c2, C + 40, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 44, temp_scale, dst_scale_inv)}

      c4 = vtrn1q_s64(c[1][0], c[1][1]);
      c5 = vtrn2q_s64(c[1][0], c[1][1]);
      ${nonline_gen_func(c4, C + LDC, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + LDC + 4, temp_scale, dst_scale_inv)}
      c6 = vtrn1q_s64(c[1][2], c[1][3]);
      c7 = vtrn2q_s64(c[1][2], c[1][3]);
      ${nonline_gen_func(c6, C + LDC + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + LDC + 12, temp_scale, dst_scale_inv)}
      c0 = vtrn1q_s64(c[1][4], c[1][5]);
      c1 = vtrn2q_s64(c[1][4], c[1][5]);
      ${nonline_gen_func(c0, C + LDC + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + LDC + 20, temp_scale, dst_scale_inv)}
      c2 = vtrn1q_s64(c[1][6], c[1][7]);
      c3 = vtrn2q_s64(c[1][6], c[1][7]);
      ${nonline_gen_func(c2, C + LDC + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + LDC + 28, temp_scale, dst_scale_inv)}
      c4 = vtrn1q_s64(c[1][8], c[1][9]);
      c5 = vtrn2q_s64(c[1][8], c[1][9]);
      ${nonline_gen_func(c4, C + LDC + 32, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + LDC + 36, temp_scale, dst_scale_inv)}
      c6 = vtrn1q_s64(c[1][10], c[1][11]);
      c7 = vtrn2q_s64(c[1][10], c[1][11]);
      ${nonline_gen_func(c6, C + LDC + 40, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + LDC + 44, temp_scale, dst_scale_inv)}
      )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);
      int32x4_t c4 = vtrn1q_s64(c[0][4], c[0][5]);
      int32x4_t c5 = vtrn2q_s64(c[0][4], c[0][5]);
      vst1q_s32(C + 16, c4);
      vst1q_s32(C + 20, c5);
      int32x4_t c6 = vtrn1q_s64(c[0][6], c[0][7]);
      int32x4_t c7 = vtrn2q_s64(c[0][6], c[0][7]);
      vst1q_s32(C + 24, c6);
      vst1q_s32(C + 28, c7);
      c0 = vtrn1q_s64(c[0][8], c[0][9]);
      c1 = vtrn2q_s64(c[0][8], c[0][9]);
      vst1q_s32(C + 32, c0);
      vst1q_s32(C + 36, c1);
      c2 = vtrn1q_s64(c[0][10], c[0][11]);
      c3 = vtrn2q_s64(c[0][10], c[0][11]);
      vst1q_s32(C + 40, c2);
      vst1q_s32(C + 44, c3);

      c4 = vtrn1q_s64(c[1][0], c[1][1]);
      c5 = vtrn2q_s64(c[1][0], c[1][1]);
      vst1q_s32(C + LDC, c4);
      vst1q_s32(C + LDC + 4, c5);
      c6 = vtrn1q_s64(c[1][2], c[1][3]);
      c7 = vtrn2q_s64(c[1][2], c[1][3]);
      vst1q_s32(C + LDC + 8, c6);
      vst1q_s32(C + LDC + 12, c7);
      c0 = vtrn1q_s64(c[1][4], c[1][5]);
      c1 = vtrn2q_s64(c[1][4], c[1][5]);
      vst1q_s32(C + LDC + 16, c0);
      vst1q_s32(C + LDC + 20, c1);
      c2 = vtrn1q_s64(c[1][6], c[1][7]);
      c3 = vtrn2q_s64(c[1][6], c[1][7]);
      vst1q_s32(C + LDC + 24, c2);
      vst1q_s32(C + LDC + 28, c3);
      c4 = vtrn1q_s64(c[1][8], c[1][9]);
      c5 = vtrn2q_s64(c[1][8], c[1][9]);
      vst1q_s32(C + LDC + 32, c4);
      vst1q_s32(C + LDC + 36, c5);
      c6 = vtrn1q_s64(c[1][10], c[1][11]);
      c7 = vtrn2q_s64(c[1][10], c[1][11]);
      vst1q_s32(C + LDC + 40, c6);
      vst1q_s32(C + LDC + 44, c7);
      )")) + std::string(R"(
      C += 48;
    }
    for (; n + N_block_8 - 1 < N; n += N_block_8) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[2][N_block_8];
      )") +
            (with_bias ? std::string(R"(
      c[0][2] = vld1q_s32(bias_ptr + m * 4);
      c[0][0] = vzip1q_s64(c[0][2], c[0][2]);
      c[0][1] = vzip2q_s64(c[0][2], c[0][2]);
      c[0][2] = c[0][0];
      c[0][3] = c[0][1];
      c[0][4] = c[0][0];
      c[0][5] = c[0][1];
      c[0][6] = c[0][0];
      c[0][7] = c[0][1];

      c[1][2] = vld1q_s32(bias_ptr + (m + 1) * 4);
      c[1][0] = vzip1q_s64(c[1][2], c[1][2]);
      c[1][1] = vzip2q_s64(c[1][2], c[1][2]);
      c[1][2] = c[1][0];
      c[1][3] = c[1][1];
      c[1][4] = c[1][0];
      c[1][5] = c[1][1];
      c[1][6] = c[1][0];
      c[1][7] = c[1][1];
      )")
                       : std::string(R"(
      c[0][0] = vdupq_n_s32(0);
      c[0][1] = c[0][0];
      c[0][2] = c[0][0];
      c[0][3] = c[0][0];
      c[0][4] = c[0][0];
      c[0][5] = c[0][0];
      c[0][6] = c[0][0];
      c[0][7] = c[0][0];

      c[1][0] = c[0][0];
      c[1][1] = c[0][0];
      c[1][2] = c[0][0];
      c[1][3] = c[0][0];
      c[1][4] = c[0][0];
      c[1][5] = c[0][0];
      c[1][6] = c[0][0];
      c[1][7] = c[0][0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b2_3 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b4_5 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b6_7 = vld1q_s8(B_ptr);
        B_ptr += 16;

        c[0][0] = vmmlaq_s32(c[0][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[0][2] = vmmlaq_s32(c[0][2], b2_3, a0_1); //! c02 c12 c03 c13
        c[0][4] = vmmlaq_s32(c[0][4], b4_5, a0_1);
        c[0][6] = vmmlaq_s32(c[0][6], b6_7, a0_1);

        a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[0][1] = vmmlaq_s32(c[0][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[0][3] = vmmlaq_s32(c[0][3], b2_3, a2_3); //! c22 c32 c23 c33
        c[0][5] = vmmlaq_s32(c[0][5], b4_5, a2_3);
        c[0][7] = vmmlaq_s32(c[0][7], b6_7, a2_3);

        a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[1][0] = vmmlaq_s32(c[1][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[1][2] = vmmlaq_s32(c[1][2], b2_3, a0_1); //! c02 c12 c03 c13
        c[1][4] = vmmlaq_s32(c[1][4], b4_5, a0_1);
        c[1][6] = vmmlaq_s32(c[1][6], b6_7, a0_1);

        c[1][1] = vmmlaq_s32(c[1][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[1][3] = vmmlaq_s32(c[1][3], b2_3, a2_3); //! c22 c32 c23 c33
        c[1][5] = vmmlaq_s32(c[1][5], b4_5, a2_3);
        c[1][7] = vmmlaq_s32(c[1][7], b6_7, a2_3);
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}
      int32x4_t c4 = vtrn1q_s64(c[0][4], c[0][5]);
      int32x4_t c5 = vtrn2q_s64(c[0][4], c[0][5]);
      ${nonline_gen_func(c4, C + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + 20, temp_scale, dst_scale_inv)}
      int32x4_t c6 = vtrn1q_s64(c[0][6], c[0][7]);
      int32x4_t c7 = vtrn2q_s64(c[0][6], c[0][7]);
      ${nonline_gen_func(c6, C + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + 28, temp_scale, dst_scale_inv)}

      c4 = vtrn1q_s64(c[1][0], c[1][1]);
      c5 = vtrn2q_s64(c[1][0], c[1][1]);
      ${nonline_gen_func(c4, C + LDC, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + LDC + 4, temp_scale, dst_scale_inv)}
      c6 = vtrn1q_s64(c[1][2], c[1][3]);
      c7 = vtrn2q_s64(c[1][2], c[1][3]);
      ${nonline_gen_func(c6, C + LDC + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + LDC + 12, temp_scale, dst_scale_inv)}
      c0 = vtrn1q_s64(c[1][4], c[1][5]);
      c1 = vtrn2q_s64(c[1][4], c[1][5]);
      ${nonline_gen_func(c0, C + LDC + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + LDC + 20, temp_scale, dst_scale_inv)}
      c2 = vtrn1q_s64(c[1][6], c[1][7]);
      c3 = vtrn2q_s64(c[1][6], c[1][7]);
      ${nonline_gen_func(c2, C + LDC + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + LDC + 28, temp_scale, dst_scale_inv)}
    )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);
      int32x4_t c4 = vtrn1q_s64(c[0][4], c[0][5]);
      int32x4_t c5 = vtrn2q_s64(c[0][4], c[0][5]);
      vst1q_s32(C + 16, c4);
      vst1q_s32(C + 20, c5);
      int32x4_t c6 = vtrn1q_s64(c[0][6], c[0][7]);
      int32x4_t c7 = vtrn2q_s64(c[0][6], c[0][7]);
      vst1q_s32(C + 24, c6);
      vst1q_s32(C + 28, c7);

      c4 = vtrn1q_s64(c[1][0], c[1][1]);
      c5 = vtrn2q_s64(c[1][0], c[1][1]);
      vst1q_s32(C + LDC, c4);
      vst1q_s32(C + LDC + 4, c5);
      c6 = vtrn1q_s64(c[1][2], c[1][3]);
      c7 = vtrn2q_s64(c[1][2], c[1][3]);
      vst1q_s32(C + LDC + 8, c6);
      vst1q_s32(C + LDC + 12, c7);
      c0 = vtrn1q_s64(c[1][4], c[1][5]);
      c1 = vtrn2q_s64(c[1][4], c[1][5]);
      vst1q_s32(C + LDC + 16, c0);
      vst1q_s32(C + LDC + 20, c1);
      c2 = vtrn1q_s64(c[1][6], c[1][7]);
      c3 = vtrn2q_s64(c[1][6], c[1][7]);
      vst1q_s32(C + LDC + 24, c2);
      vst1q_s32(C + LDC + 28, c3);
    )")) + std::string(R"(
      C += 32;
    }
    for (; n + N_block_4 - 1 < N; n += N_block_4) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[2][N_block_4];
      )") +
            (with_bias ? std::string(R"(
      c[0][2] = vld1q_s32(bias_ptr + m * 4);
      c[0][0] = vzip1q_s64(c[0][2], c[0][2]);
      c[0][1] = vzip2q_s64(c[0][2], c[0][2]);
      c[0][2] = c[0][0];
      c[0][3] = c[0][1];

      c[1][2] = vld1q_s32(bias_ptr + (m + 1) * 4);
      c[1][0] = vzip1q_s64(c[1][2], c[1][2]);
      c[1][1] = vzip2q_s64(c[1][2], c[1][2]);
      c[1][2] = c[1][0];
      c[1][3] = c[1][1];
      )")
                       : std::string(R"(
      c[0][0] = vdupq_n_s32(0);
      c[0][1] = c[0][0];
      c[0][2] = c[0][0];
      c[0][3] = c[0][0];

      c[1][0] = c[0][0];
      c[1][1] = c[0][0];
      c[1][2] = c[0][0];
      c[1][3] = c[0][0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        int8x16_t b2_3 = vld1q_s8(B_ptr + 16);
        B_ptr += 32;

        c[0][0] = vmmlaq_s32(c[0][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[0][2] = vmmlaq_s32(c[0][2], b2_3, a0_1); //! c02 c12 c03 c13

        a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[0][1] = vmmlaq_s32(c[0][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[0][3] = vmmlaq_s32(c[0][3], b2_3, a2_3); //! c22 c32 c23 c33

        a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[1][0] = vmmlaq_s32(c[1][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[1][2] = vmmlaq_s32(c[1][2], b2_3, a0_1); //! c02 c12 c03 c13

        c[1][1] = vmmlaq_s32(c[1][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[1][3] = vmmlaq_s32(c[1][3], b2_3, a2_3); //! c22 c32 c23 c33
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}

      int32x4_t c4 = vtrn1q_s64(c[1][0], c[1][1]);
      int32x4_t c5 = vtrn2q_s64(c[1][0], c[1][1]);
      ${nonline_gen_func(c4, C + LDC, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + LDC + 4, temp_scale, dst_scale_inv)}
      int32x4_t c6 = vtrn1q_s64(c[1][2], c[1][3]);
      int32x4_t c7 = vtrn2q_s64(c[1][2], c[1][3]);
      ${nonline_gen_func(c6, C + LDC + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + LDC + 12, temp_scale, dst_scale_inv)}
      )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
      int32x4_t c1 = vtrn2q_s64(c[0][0], c[0][1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[0][2], c[0][3]);
      int32x4_t c3 = vtrn2q_s64(c[0][2], c[0][3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);

      int32x4_t c4 = vtrn1q_s64(c[1][0], c[1][1]);
      int32x4_t c5 = vtrn2q_s64(c[1][0], c[1][1]);
      vst1q_s32(C + LDC, c4);
      vst1q_s32(C + LDC + 4, c5);
      int32x4_t c6 = vtrn1q_s64(c[1][2], c[1][3]);
      int32x4_t c7 = vtrn2q_s64(c[1][2], c[1][3]);
      vst1q_s32(C + LDC + 8, c6);
      vst1q_s32(C + LDC + 12, c7);
      )")) + std::string(R"(
      C += 16;
    }
    if (n < N) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[2][N_block_4];
      )") +
            (with_bias ? std::string(R"(
      c[0][2] = vld1q_s32(bias_ptr + m * 4);
      c[0][0] = vzip1q_s64(c[0][2], c[0][2]);
      c[0][1] = vzip2q_s64(c[0][2], c[0][2]);
      c[0][2] = c[0][0];
      c[0][3] = c[0][1];

      c[1][2] = vld1q_s32(bias_ptr + (m + 1) * 4);
      c[1][0] = vzip1q_s64(c[1][2], c[1][2]);
      c[1][1] = vzip2q_s64(c[1][2], c[1][2]);
      c[1][2] = c[1][0];
      c[1][3] = c[1][1];
      )")
                       : std::string(R"(
      c[0][0] = vdupq_n_s32(0);
      c[0][1] = c[0][0];
      c[0][2] = c[0][0];
      c[0][3] = c[0][0];

      c[1][0] = c[0][0];
      c[1][1] = c[0][0];
      c[1][2] = c[0][0];
      c[1][3] = c[0][0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        int8x16_t b2_3 = vld1q_s8(B_ptr + 16);
        B_ptr += 32;

        c[0][0] = vmmlaq_s32(c[0][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[0][2] = vmmlaq_s32(c[0][2], b2_3, a0_1); //! c02 c12 c03 c13

        a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[0][1] = vmmlaq_s32(c[0][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[0][3] = vmmlaq_s32(c[0][3], b2_3, a2_3); //! c22 c32 c23 c33

        a2_3 = vld1q_s8(A_ptr);
        A_ptr += 16;

        c[1][0] = vmmlaq_s32(c[1][0], b0_1, a0_1); //! c00 c10 c01 c11
        c[1][2] = vmmlaq_s32(c[1][2], b2_3, a0_1); //! c02 c12 c03 c13

        c[1][1] = vmmlaq_s32(c[1][1], b0_1, a2_3); //! c20 c30 c21 c31
        c[1][3] = vmmlaq_s32(c[1][3], b2_3, a2_3); //! c22 c32 c23 c33
      }
      )") +
            (need_postprocess ? std::string(R"(
      switch (N - n) {
        case 3: {
          int32x4_t c0 = vtrn1q_s64(c[0][2], c[0][3]);
          ${nonline_gen_func(c0, C + 8, temp_scale, dst_scale_inv)}

          int32x4_t c4 = vtrn1q_s64(c[1][2], c[1][3]);
          ${nonline_gen_func(c4, C + LDC + 8, temp_scale, dst_scale_inv)}
          //! fall through
        }
        case 2: {
          int32x4_t c0 = vtrn2q_s64(c[0][0], c[0][1]);
          ${nonline_gen_func(c0, C + 4, temp_scale, dst_scale_inv)}

          int32x4_t c4 = vtrn2q_s64(c[1][0], c[1][1]);
          ${nonline_gen_func(c4, C + LDC + 4, temp_scale, dst_scale_inv)}
          //! fall through
        }
        case 1: {
          int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
          ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}

          int32x4_t c4 = vtrn1q_s64(c[1][0], c[1][1]);
          ${nonline_gen_func(c4, C + LDC, temp_scale, dst_scale_inv)}
          //! fall through
        }
      }
      )")
                              : std::string(R"(
      switch (N - n) {
        case 3: {
          int32x4_t c0 = vtrn1q_s64(c[0][2], c[0][3]);
          vst1q_s32(C + 8, c0);

          int32x4_t c4 = vtrn1q_s64(c[1][2], c[1][3]);
          vst1q_s32(C + LDC + 8, c4);
          //! fall through
        }
        case 2: {
          int32x4_t c0 = vtrn2q_s64(c[0][0], c[0][1]);
          vst1q_s32(C + 4, c0);

          int32x4_t c4 = vtrn2q_s64(c[1][0], c[1][1]);
          vst1q_s32(C + LDC + 4, c4);
          //! fall through
        }
        case 1: {
          int32x4_t c0 = vtrn1q_s64(c[0][0], c[0][1]);
          vst1q_s32(C, c0);

          int32x4_t c4 = vtrn1q_s64(c[1][0], c[1][1]);
          vst1q_s32(C + LDC, c4);
          //! fall through
        }
      }
      )")) + std::string(R"(
      C += 4 * (N - n);
    }
  }
  if (m < M) {
    B_ptr = B0;
    C = C0 + m * LDC;
    int n = 0;
    for (; n + N_block_12 - 1 < N; n += N_block_12) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[N_block_12];
      )") +
            (with_bias ? std::string(R"(
      c[2] = vld1q_s32(bias_ptr + m * 4);
      c[0] = vzip1q_s64(c[2], c[2]);
      c[1] = vzip2q_s64(c[2], c[2]);
      c[2] = c[0];
      c[3] = c[1];
      c[4] = c[0];
      c[5] = c[1];
      c[6] = c[0];
      c[7] = c[1];
      c[8] = c[0];
      c[9] = c[1];
      c[10] = c[0];
      c[11] = c[1];
      )")
                       : std::string(R"(
      c[0] = vdupq_n_s32(0);
      c[1] = c[0];
      c[2] = c[0];
      c[3] = c[0];
      c[4] = c[0];
      c[5] = c[0];
      c[6] = c[0];
      c[7] = c[0];
      c[8] = c[0];
      c[9] = c[0];
      c[10] = c[0];
      c[11] = c[0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 48;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b2_3 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b4_5 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b6_7 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b8_9 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b10_11 = vld1q_s8(B_ptr);
        B_ptr += 16;

        c[0] = vmmlaq_s32(c[0], b0_1, a0_1); //! c00 c10 c01 c11
        c[2] = vmmlaq_s32(c[2], b2_3, a0_1); //! c02 c12 c03 c13
        c[4] = vmmlaq_s32(c[4], b4_5, a0_1);
        c[6] = vmmlaq_s32(c[6], b6_7, a0_1);
        c[8] = vmmlaq_s32(c[8], b8_9, a0_1);
        c[10] = vmmlaq_s32(c[10], b10_11, a0_1);

        c[1] = vmmlaq_s32(c[1], b0_1, a2_3); //! c20 c30 c21 c31
        c[3] = vmmlaq_s32(c[3], b2_3, a2_3); //! c22 c32 c23 c33
        c[5] = vmmlaq_s32(c[5], b4_5, a2_3);
        c[7] = vmmlaq_s32(c[7], b6_7, a2_3);
        c[9] = vmmlaq_s32(c[9], b8_9, a2_3);
        c[11] = vmmlaq_s32(c[11], b10_11, a2_3);
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}
      int32x4_t c4 = vtrn1q_s64(c[4], c[5]);
      int32x4_t c5 = vtrn2q_s64(c[4], c[5]);
      ${nonline_gen_func(c4, C + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + 20, temp_scale, dst_scale_inv)}
      int32x4_t c6 = vtrn1q_s64(c[6], c[7]);
      int32x4_t c7 = vtrn2q_s64(c[6], c[7]);
      ${nonline_gen_func(c6, C + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + 28, temp_scale, dst_scale_inv)}
      c0 = vtrn1q_s64(c[8], c[9]);
      c1 = vtrn2q_s64(c[8], c[9]);
      ${nonline_gen_func(c0, C + 32, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 36, temp_scale, dst_scale_inv)}
      c2 = vtrn1q_s64(c[10], c[11]);
      c3 = vtrn2q_s64(c[10], c[11]);
      ${nonline_gen_func(c2, C + 40, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 44, temp_scale, dst_scale_inv)}
      )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);
      int32x4_t c4 = vtrn1q_s64(c[4], c[5]);
      int32x4_t c5 = vtrn2q_s64(c[4], c[5]);
      vst1q_s32(C + 16, c4);
      vst1q_s32(C + 20, c5);
      int32x4_t c6 = vtrn1q_s64(c[6], c[7]);
      int32x4_t c7 = vtrn2q_s64(c[6], c[7]);
      vst1q_s32(C + 24, c6);
      vst1q_s32(C + 28, c7);
      c0 = vtrn1q_s64(c[8], c[9]);
      c1 = vtrn2q_s64(c[8], c[9]);
      vst1q_s32(C + 32, c0);
      vst1q_s32(C + 36, c1);
      c2 = vtrn1q_s64(c[10], c[11]);
      c3 = vtrn2q_s64(c[10], c[11]);
      vst1q_s32(C + 40, c2);
      vst1q_s32(C + 44, c3);
      )")) + std::string(R"(
      C += 48;
    }
    for (; n + N_block_8 - 1 < N; n += N_block_8) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[N_block_8];
      )") +
            (with_bias ? std::string(R"(
      c[2] = vld1q_s32(bias_ptr + m * 4);
      c[0] = vzip1q_s64(c[2], c[2]);
      c[1] = vzip2q_s64(c[2], c[2]);
      c[2] = c[0];
      c[3] = c[1];
      c[4] = c[0];
      c[5] = c[1];
      c[6] = c[0];
      c[7] = c[1];
      )")
                       : std::string(R"(
      c[0] = vdupq_n_s32(0);
      c[1] = c[0];
      c[2] = c[0];
      c[3] = c[0];
      c[4] = c[0];
      c[5] = c[0];
      c[6] = c[0];
      c[7] = c[0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 48;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b2_3 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b4_5 = vld1q_s8(B_ptr);
        B_ptr += 16;
        int8x16_t b6_7 = vld1q_s8(B_ptr);
        B_ptr += 16;

        c[0] = vmmlaq_s32(c[0], b0_1, a0_1); //! c00 c10 c01 c11
        c[2] = vmmlaq_s32(c[2], b2_3, a0_1); //! c02 c12 c03 c13
        c[4] = vmmlaq_s32(c[4], b4_5, a0_1);
        c[6] = vmmlaq_s32(c[6], b6_7, a0_1);

        c[1] = vmmlaq_s32(c[1], b0_1, a2_3); //! c20 c30 c21 c31
        c[3] = vmmlaq_s32(c[3], b2_3, a2_3); //! c22 c32 c23 c33
        c[5] = vmmlaq_s32(c[5], b4_5, a2_3);
        c[7] = vmmlaq_s32(c[7], b6_7, a2_3);
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}
      int32x4_t c4 = vtrn1q_s64(c[4], c[5]);
      int32x4_t c5 = vtrn2q_s64(c[4], c[5]);
      ${nonline_gen_func(c4, C + 16, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c5, C + 20, temp_scale, dst_scale_inv)}
      int32x4_t c6 = vtrn1q_s64(c[6], c[7]);
      int32x4_t c7 = vtrn2q_s64(c[6], c[7]);
      ${nonline_gen_func(c6, C + 24, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c7, C + 28, temp_scale, dst_scale_inv)}
      )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);
      int32x4_t c4 = vtrn1q_s64(c[4], c[5]);
      int32x4_t c5 = vtrn2q_s64(c[4], c[5]);
      vst1q_s32(C + 16, c4);
      vst1q_s32(C + 20, c5);
      int32x4_t c6 = vtrn1q_s64(c[6], c[7]);
      int32x4_t c7 = vtrn2q_s64(c[6], c[7]);
      vst1q_s32(C + 24, c6);
      vst1q_s32(C + 28, c7);
      )")) + std::string(R"(
      C += 32;
    }
    for (; n + N_block_4 - 1 < N; n += N_block_4) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[N_block_4];
      )") +
            (with_bias ? std::string(R"(
      c[2] = vld1q_s32(bias_ptr + m * 4);
      c[0] = vzip1q_s64(c[2], c[2]);
      c[1] = vzip2q_s64(c[2], c[2]);
      c[2] = c[0];
      c[3] = c[1];
      )")
                       : std::string(R"(
      c[0] = vdupq_n_s32(0);
      c[1] = c[0];
      c[2] = c[0];
      c[3] = c[0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 48;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        int8x16_t b2_3 = vld1q_s8(B_ptr + 16);
        B_ptr += 32;

        c[0] = vmmlaq_s32(c[0], b0_1, a0_1); //! c00 c10 c01 c11
        c[2] = vmmlaq_s32(c[2], b2_3, a0_1); //! c02 c12 c03 c13
        c[1] = vmmlaq_s32(c[1], b0_1, a2_3); //! c20 c30 c21 c31
        c[3] = vmmlaq_s32(c[3], b2_3, a2_3); //! c22 c32 c23 c33
      }
      )") +
            (need_postprocess ? std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c1, C + 4, temp_scale, dst_scale_inv)}
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      ${nonline_gen_func(c2, C + 8, temp_scale, dst_scale_inv)}
      ${nonline_gen_func(c3, C + 12, temp_scale, dst_scale_inv)}
      )")
                              : std::string(R"(
      int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
      int32x4_t c1 = vtrn2q_s64(c[0], c[1]);
      vst1q_s32(C, c0);
      vst1q_s32(C + 4, c1);
      int32x4_t c2 = vtrn1q_s64(c[2], c[3]);
      int32x4_t c3 = vtrn2q_s64(c[2], c[3]);
      vst1q_s32(C + 8, c2);
      vst1q_s32(C + 12, c3);
      )")) + std::string(R"(
      C += 16;
    }
    if (n < N) {
      A_ptr = A0 + m * 4 * 4 * K;
      int32x4_t c[N_block_4];
      )") +
            (with_bias ? std::string(R"(
      c[2] = vld1q_s32(bias_ptr + m * 4);
      c[0] = vzip1q_s64(c[2], c[2]);
      c[1] = vzip2q_s64(c[2], c[2]);
      c[2] = c[0];
      c[3] = c[1];
      )")
                       : std::string(R"(
      c[0] = vdupq_n_s32(0);
      c[1] = c[0];
      c[2] = c[0];
      c[3] = c[0];
      )")) + std::string(R"(
      for (int k = 0; k < K; k += 2) {
        int8x16_t a0_1 = vld1q_s8(A_ptr);
        A_ptr += 16;
        int8x16_t a2_3 = vld1q_s8(A_ptr);
        A_ptr += 48;

        int8x16_t b0_1 = vld1q_s8(B_ptr);
        int8x16_t b2_3 = vld1q_s8(B_ptr + 16);
        B_ptr += 32;

        c[0] = vmmlaq_s32(c[0], b0_1, a0_1); //! c00 c10 c01 c11
        c[2] = vmmlaq_s32(c[2], b2_3, a0_1); //! c02 c12 c03 c13
        c[1] = vmmlaq_s32(c[1], b0_1, a2_3); //! c20 c30 c21 c31
        c[3] = vmmlaq_s32(c[3], b2_3, a2_3); //! c22 c32 c23 c33
      }
      )") +
            (need_postprocess ? std::string(R"(
      switch (N - n) {
        case 3: {
          int32x4_t c0 = vtrn1q_s64(c[2], c[3]);
          ${nonline_gen_func(c0, C + 8, temp_scale, dst_scale_inv)}
          //! fall through
        }
        case 2: {
          int32x4_t c0 = vtrn2q_s64(c[0], c[1]);
          ${nonline_gen_func(c0, C + 4, temp_scale, dst_scale_inv)}
          //! fall through
        }
        case 1: {
          int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
          ${nonline_gen_func(c0, C, temp_scale, dst_scale_inv)}
          //! fall through
        }
      }
      )")
                              : std::string(R"(
      switch (N - n) {
        case 3: {
          int32x4_t c0 = vtrn1q_s64(c[2], c[3]);
          vst1q_s32(C + 8, c0);
          //! fall through
        }
        case 2: {
          int32x4_t c0 = vtrn2q_s64(c[0], c[1]);
          vst1q_s32(C + 4, c0);
          //! fall through
        }
        case 1: {
          int32x4_t c0 = vtrn1q_s64(c[0], c[1]);
          vst1q_s32(C, c0);
          //! fall through
        }
      }
      )")) + std::string(R"(
      C += 4 * (N - n);
    }
  }
#undef N_block_12
#undef N_block_8
#undef N_block_4
    }
    )");
    return StringTemplate::StringTemplateArgs()
            .add("dst_specifier", dst_specifier)
            .add("preset_str", preset_str)
            .add("kernel_sig", sig)
            .add("nonline_gen_init", nonline_gen_init)
            .add("nonline_gen_func", nonline_gen_func)
            .render(kern_body);
}

std::string gen_common_code() {
    std::string res = R"(
#define ROUND_UP(A, B) (((A) + ((B) - 1)) / (B) * (B)) 
    )";
    return res;
}

}  // namespace

std::string MatmulInt8I8mmM8K8N12MK4Kernel::GetKernelSymbol(TContext* ctx) const {
    std::stringstream ss;
    ss << "Arm64_int8_i8mm_m8_k8_n12_mk4_gemm";
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

std::vector<KernelObj> MatmulInt8I8mmM8K8N12MK4Kernel::GetDependInternalSymbol(
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

std::string MatmulInt8I8mmM8K8N12MK4Kernel::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    writer << "#include <string.h>\n";
    writer << "#include <arm_neon.h>\n";
    writer << "#include \"utils.h\"\n";
    writer << gen_common_code();
    auto dtype = ctx->getAttrStr("dtype");
    std::string last_dtype = "si8";
    if (ctx->haveAttr("last_dtype")) {
        last_dtype = ctx->getAttrStr("last_dtype");
    }
    std::string dst_specifier = "int32_t";
    auto nonline_mode =
            ctx->haveAttr("nonlineMode") ? ctx->getAttrStr("nonlineMode") : "IDENTITY";
    if (Utils::is_quant_dtype(dtype) &&
        (nonline_mode == "RELU" || nonline_mode == "IDENTITY" ||
         nonline_mode == "H_SWISH")) {
        dst_specifier = Utils::cvt_dtype_specifier(last_dtype);
    }
    bool need_postprocess = Utils::is_quant_dtype(dtype);

    writer << gen_pack_a(GetPackASignature(ctx));
    writer << gen_pack_b(GetPackBSignature(ctx));
    writer << gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
    writer << gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));

    if (Utils::is_quant_dtype(dtype)) {
        writer << gen_kernel(
                dst_specifier, GetNakedKernelSignature(ctx), ctx, "", need_postprocess);
    } else {
        std::string preset_temp = R"(
        const int8_t* pack_a = A;
        int8_t* pack_b = workspace;
        ${packb_sym}(pack_b, B, LDB, 0, N, 0, K);
    )";
        std::string preset_str =
                StringTemplate::StringTemplateArgs()
                        .add("packa_workspace_sym", GetPackAWorkspaceSymbol(ctx))
                        .add("packb_sym", GetPackBSymbol(ctx))
                        .render(preset_temp);
        writer << gen_kernel(
                dst_specifier, GetKernelSignature(ctx), ctx, preset_str,
                need_postprocess);
    }
    return writer.str();
}

std::string MatmulInt8I8mmM8K8N12MK4Kernel::GetPackAWorkspaceBody(TContext* ctx) const {
    return gen_pack_a_workspace(GetPackAWorkspaceSignature(ctx));
}
std::string MatmulInt8I8mmM8K8N12MK4Kernel::GetPackBWorkspaceBody(TContext* ctx) const {
    return gen_pack_b_workspace(GetPackBWorkspaceSignature(ctx));
}

// vim: syntax=cpp.doxygen
