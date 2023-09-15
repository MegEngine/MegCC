#include <sstream>
#include <string>
#include "Arm/Arm64/Activation.h"
#include "Arm/Arm64/ConvKernel.h"
#include "Arm/Arm64/InternalKernel/InternalKernel.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"
#include "compiler/KernelGen/KernelGen.h"

using namespace megcc;
using namespace KernelGen;
using namespace Arm64;
using namespace ArmCommon;
namespace megcc {
namespace KernelGen {
namespace Arm64 {

namespace {
std::string pad_src() {
    return R"(
static inline void pad_src(const int8_t *src, int8_t *dst, const int IC, const int IH,
             const int IW, const int PH, const int PW) {
  const int paded_H = IH + 2 * PH;
  const int paded_W = IW + 2 * PW;
  const int paded_HW = paded_H * paded_W;
  memset(dst, 0, IC * PACKED_IC * paded_HW * sizeof(int8_t));
  for (int ic = 0; ic < IC; ic++) {
    dst += PH * paded_W * PACKED_IC;
    for (int ih = 0; ih < IH; ++ih) {
      memcpy(dst + ih * paded_W * PACKED_IC + PW * PACKED_IC, src + ih * IW * PACKED_IC,
             IW * PACKED_IC * sizeof(int8_t));
    }
    dst += (IH + PH) * paded_W * PACKED_IC;
    src += IH * IW * PACKED_IC;
  }
}
    )";
}

std::string im2col_s1() {
    return R"(
static inline void im2col(const int8_t *src, int8_t *dst, const int IC, const int OH,
            const int OW, const int FH, const int FW, const int paded_H,
            const int paded_W) {
  const int src_stride_ic = paded_H * paded_W * PACKED_IC;
  const int src_stride_h = paded_W * PACKED_IC;
  const int dst_stride_ic = OH * OW * PACKED_IC;
  const int dst_stride_h = OW * PACKED_IC;
  for (int ic = 0; ic < IC; ++ic) {
    for (int fh = 0; fh < FH; ++fh) {
      for (int fw = 0; fw < FW; ++fw) {
        const int8_t *src_base =
            src + ic * src_stride_ic + fh * src_stride_h + fw * PACKED_IC;
        for (int oh = 0; oh < OH; ++oh) {
          memcpy(dst + oh * dst_stride_h, src_base + oh * src_stride_h,
                 dst_stride_h * sizeof(int8_t));
        }
        dst += dst_stride_ic;
      }
    }
  }
}
    )";
}

std::string im2col_s2() {
    return R"(
static inline void im2col(const int8_t *src, int8_t *dst, const int IC, const int OH,
            const int OW, const int FH, const int FW, const int paded_H,
            const int paded_W) {
  const int src_stride_ic = paded_H * paded_W * PACKED_IC;
  const int src_stride_h = paded_W * PACKED_IC;
  const int dst_stride_ic = OH * OW * PACKED_IC;
  const int dst_stride_h = OW * PACKED_IC;
  for (int ic = 0; ic < IC; ++ic) {
    for (int fh = 0; fh < FH; ++fh) {
      for (int fw = 0; fw < FW; ++fw) {
        const int8_t *src_base =
            src + ic * src_stride_ic + fh * src_stride_h + fw * PACKED_IC;
        for (int oh = 0; oh < OH; ++oh) {
          const int32_t *src_ptr = (const int32_t*)(src_base + oh * 2 * src_stride_h);
          int32_t *dst_ptr = (int32_t*)(dst + oh * dst_stride_h);
          int ow = 0;
          for (; ow + 3 < OW; ow += 4) {
            int32x4x2_t d = vld2q_s32(src_ptr + ow * 2);
            vst1q_s32(dst_ptr + ow, d.val[0]);
          }
          for (; ow < OW; ++ow) {
            dst_ptr[ow] = src_ptr[ow * 2];
          }
        }
        dst += dst_stride_ic;
      }
    }
  }
}
    )";
}

std::string fuse_im2col_packB_s2(TContext* ctx) {
    const int FH = ctx->getAttrInt("kernel_h"), FW = ctx->getAttrInt("kernel_w");
    CC_ASSERT(FH == FW);
    std::string res = R"(
static void fuse_im2col_packB_s2(const int8_t *src, int8_t *dst, const int IC,
                                 const int IH, const int IW, const int OH, const int OW) {
    TINYNN_ASSERT(OW >= 12);)";
    if (FW % 2) {
        res += R"(
        int32x4x2_t d[15];
        int32_t buffer[6][24];)";
    }
    res += R"(
  int32_t *dst_ptr = (int32_t *)dst;
  const int N = OH * OW;
  int n = 0;
  for (; n + 11 < N; n += 12) {
    const int oh = n / OW, ow = n % OW;
    const int32_t *src_base =
        (const int32_t *)(src + oh * 2 * IW * PACKED_IC + ow * 2 * PACKED_IC);
    if (OW - ow >= 12) {)";
    if (FW % 2) {
        res += R"(
      int ic = 0;
      for (; ic + 1 < IC; ic += 2) {)";
        int idx = 0;
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 96);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx2}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 16);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx5}] = vld2q_s32(src_base + IW * ${row1_idx} + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 96);");
            }
        }
        int fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 96);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx2", (idx + 2) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx4", (idx + 4) % 15)
                       .add("idx5", (idx + 5) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("idx8", (idx + 8) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx2}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 16);
                        src_base += (IH * IW);
                        d[${idx3}] = vld2q_s32(src_base);
                        d[${idx4}] = vld2q_s32(src_base + 8);
                        d[${idx5}] = vld2q_s32(src_base + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
        idx = (idx + 9) % 15;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                   std::string(", src_base + ") + std::to_string(fw * 2 + 1) +
                   std::string(", 96);");
        }
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string((FH + 1) / 2 * FW * 24 + fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 96);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset",
                                (FH + 1) / 2 * FW * 24 + fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2 + 1)
                           .add("row1_idx", fh * 2 + 2)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx2}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 16);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx5}] = vld2q_s32(src_base + IW * ${row1_idx} + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 24 + fh * FW * 24 +
                               (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 2) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 96);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 24)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
        if (ic < IC) {
    )");
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 96);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx2}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 16);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx5}] = vld2q_s32(src_base + IW * ${row1_idx} + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 96);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 96);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx2", (idx + 2) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("idx8", (idx + 8) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx2}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 15);
                        src_base += (IH * IW);
                        d[${idx3}].val[0] = vdupq_n_s32(0);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx3}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[1], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);

                        dst_ptr += (${dst_offset} + 24);
                    }
                } else {
                    const int32_t* src_base_next = (const int32_t*)(src + (oh + 1) * 2 * IW * PACKED_IC);
                    const int part0 = OW - ow, part1 = 12 - part0;
                    int ic = 0;
                    for (; ic + 1 < IC; ic += 2) {
                       )");
        idx = (idx + 9) % 15;
        int buffer_idx = 0;
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part1 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx0}] + part0 * 2, src_base_next + IW * ${row0_idx} + ${src_offset}, part1 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}] + part0 * 2, src_base_next + IW * ${row1_idx}, part1 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx1}] = vld2q_s32(buffer[${buffer_idx0}] + 8);
                        d[${idx2}] = vld2q_s32(buffer[${buffer_idx0}] + 16);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx4}] = vld2q_s32(buffer[${buffer_idx1}] + 8);
                        d[${idx5}] = vld2q_s32(buffer[${buffer_idx1}] + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") +
                       std::to_string(fh * 2 + 1) + std::string(" + ") +
                       std::to_string(fw * 2 + 1) + std::string(", part1 * 2 * 4);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part0 * 2 * 4);");
            res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base_next + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part1 * 2 * 4);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx2", (idx + 2) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx4", (idx + 4) % 15)
                       .add("idx5", (idx + 5) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("idx8", (idx + 8) % 15)
                       .add("buffer_idx0", buffer_idx % 6)
                       .add("buffer_idx1", (buffer_idx + 1) % 6)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx0}] + part0 * 2, src_base_next + IW * ${row0_idx} + ${src_offset}, part1 * 2 * 4);
                        src_base += (IH * IW);
                        src_base_next += (IH * IW);
                        memcpy(buffer[${buffer_idx1}], src_base, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}] + part0 * 2, src_base_next, part1 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx1}] = vld2q_s32(buffer[${buffer_idx0}] + 8);
                        d[${idx2}] = vld2q_s32(buffer[${buffer_idx0}] + 16);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx4}] = vld2q_s32(buffer[${buffer_idx1}] + 8);
                        d[${idx5}] = vld2q_s32(buffer[${buffer_idx1}] + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
        idx = (idx + 9) % 15;
        buffer_idx = (buffer_idx + 2) % 6;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                   std::string(", src_base + ") + std::to_string(fw * 2 + 1) +
                   std::string(", part0 * 2 * 4);");
            res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                   std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                   std::string(", src_base_next + ") + std::to_string(fw * 2 + 1) +
                   std::string(", part1 * 2 * 4);");
        }
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string((FH + 1) / 2 * FW * 24 + fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string((FH + 1) / 2 * FW * 24 + fh * FW * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") +
                       std::to_string(fh * 2 + 1) + std::string(" + ") +
                       std::to_string(fw * 2) + std::string(", part1 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset",
                                (FH + 1) / 2 * FW * 24 + fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2 + 1)
                           .add("row1_idx", fh * 2 + 2)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx0}] + part0 * 2, src_base_next + IW * ${row0_idx} + ${src_offset}, part1 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}] + part0 * 2, src_base_next + IW * ${row1_idx}, part1 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx1}] = vld2q_s32(buffer[${buffer_idx0}] + 8);
                        d[${idx2}] = vld2q_s32(buffer[${buffer_idx0}] + 16);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx4}] = vld2q_s32(buffer[${buffer_idx1}] + 8);
                        d[${idx5}] = vld2q_s32(buffer[${buffer_idx1}] + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 24 + fh * FW * 24 +
                               (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 2) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 24 + fh * FW * 24 +
                               (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") +
                       std::to_string(fh * 2 + 2) + std::string(" + ") +
                       std::to_string(fw * 2 + 1) + std::string(", part1 * 2 * 4);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 24)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
            src_base_next += (IH * IW);
        }
        if (ic < IC) {
    )");
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(fh * FW * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part1 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx2", (idx + 2) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx5", (idx + 5) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("idx8", (idx + 8) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx0}] + part0 * 2, src_base_next + IW * ${row0_idx} + ${src_offset}, part1 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}] + part0 * 2, src_base_next + IW * ${row1_idx}, part1 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx1}] = vld2q_s32(buffer[${buffer_idx0}] + 8);
                        d[${idx2}] = vld2q_s32(buffer[${buffer_idx0}] + 16);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx4}] = vld2q_s32(buffer[${buffer_idx1}] + 8);
                        d[${idx5}] = vld2q_s32(buffer[${buffer_idx1}] + 16);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx5}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(fh * FW * 24 + (FW + 1) / 2 * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") +
                       std::to_string(fh * 2 + 1) + std::string(" + ") +
                       std::to_string(fw * 2 + 1) + std::string(", part1 * 2 * 4);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part0 * 2 * 4);");
            res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                   std::to_string(fh * FW * 24 + fw * 24) +
                   std::string(", src_base_next + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part1 * 2 * 4);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx2", (idx + 2) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx4", (idx + 4) % 15)
                       .add("idx5", (idx + 5) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("idx8", (idx + 8) % 15)
                       .add("buffer_idx0", buffer_idx % 6)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 24 + FW / 2 * 24)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx0}] + part0 * 2, src_base_next + IW * ${row0_idx} + ${src_offset}, part1 * 2 * 4);
                        src_base += (IH * IW);
                        src_base_next += (IH * IW);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx1}] = vld2q_s32(buffer[${buffer_idx0}] + 8);
                        d[${idx2}] = vld2q_s32(buffer[${buffer_idx0}] + 16);
                        d[${idx3}].val[0] = vdupq_n_s32(0);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx3}].val[0]);
                        d[${idx8}] = vzipq_s32(d[${idx2}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 16, d[${idx8}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 20, d[${idx8}].val[1]);

                        dst_ptr += (${dst_offset} + 24);
                    }
                }
            }
            for (; n + 7 < N; n += 8) {
                const int oh = n / OW, ow = n % OW;
                const int32_t *src_base =
                    (const int32_t *)(src + oh * 2 * IW * PACKED_IC + ow * 2 * PACKED_IC);
                TINYNN_ASSERT(OW - ow >= 8);
                int ic = 0;
                for (; ic + 1 < IC; ic += 2) {
                       )");
        idx = (idx + 9) % 15;
        buffer_idx = (buffer_idx + 2) % 6;
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 64);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 16 + FW / 2 * 16)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 16 + (FW + 1) / 2 * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 64);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 16 + fw * 16) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 64);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx4", (idx + 4) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 16 + FW / 2 * 16)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        src_base += (IH * IW);
                        d[${idx3}] = vld2q_s32(src_base);
                        d[${idx4}] = vld2q_s32(src_base + 8);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                       )");
        idx = (idx + 9) % 15;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 16 + (FW + 1) / 2 * 16 + fw * 16) +
                   std::string(", src_base + ") + std::to_string(fw * 2 + 1) +
                   std::string(", 64);");
        }
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string((FH + 1) / 2 * FW * 16 + fh * FW * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 64);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset",
                                (FH + 1) / 2 * FW * 16 + fh * FW * 16 + FW / 2 * 16)
                           .add("row0_idx", fh * 2 + 1)
                           .add("row1_idx", fh * 2 + 2)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 16 + fh * FW * 16 +
                               (FW + 1) / 2 * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 2) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 64);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 16)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
        if (ic < IC) {
    )");
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 64);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx1", (idx + 1) % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx4", (idx + 4) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("idx7", (idx + 7) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 16 + FW / 2 * 16)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 8);
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx4}] = vld2q_s32(src_base + IW * ${row1_idx} + 8);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[0], d[${idx4}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 16 + (FW + 1) / 2 * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 64);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 16 + fw * 16) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 64);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx1", (idx + 1) % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("idx7", (idx + 7) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 16 + FW / 2 * 16)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx1}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset} + 7);
                        src_base += (IH * IW);
                        d[${idx3}].val[0] = vdupq_n_s32(0);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        d[${idx7}] = vzipq_s32(d[${idx1}].val[1], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 8, d[${idx7}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 12, d[${idx7}].val[1]);

                        dst_ptr += (${dst_offset} + 16);
                    }
                }
                for (; n < N; n += 4) {
                    const int oh = n / OW, ow = n % OW;
                    const int32_t* src_base = (const int32_t*)(src + oh * 2 * IW * PACKED_IC + ow * 2 * PACKED_IC);
                    TINYNN_ASSERT(oh + 1 == OH);
                    if (OW - ow >= 4) {
                        int ic = 0;
                        for (; ic + 1 < IC; ic += 2) {
                       )");
        idx = (idx + 9) % 15;
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 32);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 32);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + fw * 8) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 32);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        src_base += (IH * IW);
                        d[${idx3}] = vld2q_s32(src_base);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
        idx = (idx + 9) % 15;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                   std::string(", src_base + ") + std::to_string(fw * 2 + 1) +
                   std::string(", 32);");
        }
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string((FH + 1) / 2 * FW * 8 + fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 32);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset",
                                (FH + 1) / 2 * FW * 8 + fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2 + 1)
                           .add("row1_idx", fh * 2 + 2)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 8 + fh * FW * 8 + (FW + 1) / 2 * 8 +
                               fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 2) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 32);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 8)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
        if (ic < IC) {
    )");
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 32);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        d[${idx3}] = vld2q_s32(src_base + IW * ${row1_idx});
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", 32);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + fw * 8) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) + std::string(", 32);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        d[${idx0}] = vld2q_s32(src_base + IW * ${row0_idx} + ${src_offset});
                        src_base += (IH * IW);
                        d[${idx3}].val[0] = vdupq_n_s32(0);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);

                        dst_ptr += (${dst_offset} + 8);
                    }
                } else {
                    const int part0 = OW - ow;
                    int ic = 0;
                    for (; ic + 1 < IC; ic += 2) {
                       )");
        idx = (idx + 9) % 15;
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + fw * 8) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part0 * 2 * 4);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("buffer_idx0", buffer_idx % 6)
                       .add("buffer_idx1", (buffer_idx + 1) % 6)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        src_base += (IH * IW);
                        memcpy(buffer[${buffer_idx1}], src_base, part0 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
        idx = (idx + 9) % 15;
        buffer_idx = (buffer_idx + 2) % 6;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                   std::string(", src_base + ") + std::to_string(fw * 2 + 1) +
                   std::string(", part0 * 2 * 4);");
        }
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string((FH + 1) / 2 * FW * 8 + fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset",
                                (FH + 1) / 2 * FW * 8 + fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2 + 1)
                           .add("row1_idx", fh * 2 + 2)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(
                               (FH + 1) / 2 * FW * 8 + fh * FW * 8 + (FW + 1) / 2 * 8 +
                               fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 2) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 8)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
        if (ic < IC) {
    )");
        for (int fh = 0; fh < FH / 2; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
            }
            res += StringTemplate::StringTemplateArgs()
                           .add("idx0", idx % 15)
                           .add("idx3", (idx + 3) % 15)
                           .add("idx6", (idx + 6) % 15)
                           .add("buffer_idx0", buffer_idx % 6)
                           .add("buffer_idx1", (buffer_idx + 1) % 6)
                           .add("src_offset", FW / 2 * 2)
                           .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                           .add("row0_idx", fh * 2)
                           .add("row1_idx", fh * 2 + 1)
                           .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        memcpy(buffer[${buffer_idx1}], src_base + IW * ${row1_idx}, part0 * 2 * 4);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx3}] = vld2q_s32(buffer[${buffer_idx1}]);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);
                       )");
            idx = (idx + 9) % 15;
            buffer_idx = (buffer_idx + 2) % 6;
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * FW * 8 + (FW + 1) / 2 * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh * 2 + 1) +
                       std::string(" + ") + std::to_string(fw * 2 + 1) +
                       std::string(", part0 * 2 * 4);");
            }
        }
        fh = FH / 2;
        for (int fw = 0; fw < FW / 2; ++fw) {
            res += std::string("memcpy(dst_ptr + ") +
                   std::to_string(fh * FW * 8 + fw * 8) +
                   std::string(", src_base + IW * ") + std::to_string(fh * 2) +
                   std::string(" + ") + std::to_string(fw * 2) +
                   std::string(", part0 * 2 * 4);");
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("idx0", idx % 15)
                       .add("idx3", (idx + 3) % 15)
                       .add("idx6", (idx + 6) % 15)
                       .add("buffer_idx0", buffer_idx % 6)
                       .add("src_offset", FW / 2 * 2)
                       .add("dst_offset", fh * FW * 8 + FW / 2 * 8)
                       .add("row0_idx", fh * 2)
                       .render(R"(
                        memcpy(buffer[${buffer_idx0}], src_base + IW * ${row0_idx} + ${src_offset}, part0 * 2 * 4);
                        src_base += (IH * IW);
                        d[${idx0}] = vld2q_s32(buffer[${buffer_idx0}]);
                        d[${idx3}].val[0] = vdupq_n_s32(0);
                        d[${idx6}] = vzipq_s32(d[${idx0}].val[0], d[${idx3}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset}, d[${idx6}].val[0]);
                        vst1q_s32(dst_ptr + ${dst_offset} + 4, d[${idx6}].val[1]);

                        dst_ptr += (${dst_offset} + 8);
                    }
                }
            }
        }
                       )");
        idx = (idx + 9) % 15;
        buffer_idx = (buffer_idx + 2) % 6;
    } else {
        res += R"(
        for (int ic = 0; ic < IC; ++ic) {)";
        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * (FW / 2) * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 96);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 12)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
        )");
        res += R"(
        } else {
            const int32_t* src_base_next = (const int32_t*)(src + (oh + 1) * 2 * IW * PACKED_IC);
            const int part0 = OW - ow, part1 = 12 - part0;
            for (int ic = 0; ic < IC; ++ic) {
                )";
        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * (FW / 2) * 24 + fw * 24) +
                       std::string(", src_base + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
                res += std::string("memcpy(dst_ptr + part0 * 2 + ") +
                       std::to_string(fh * (FW / 2) * 24 + fw * 24) +
                       std::string(", src_base_next + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part1 * 2 * 4);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 12)
                       .render(R"(
                dst_ptr += ${offset};
                src_base += (IH * IW);
                src_base_next += (IH * IW);
            }
        }
    }
    for (; n + 7 < N; n += 8) {
        const int oh = n / OW, ow = n % OW;
        const int32_t *src_base =
            (const int32_t *)(src + oh * 2 * IW * PACKED_IC + ow * 2 * PACKED_IC);
        TINYNN_ASSERT(OW - ow >= 8);
        for (int ic = 0; ic < IC; ++ic) {)");
        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * (FW / 2) * 16 + fw * 16) +
                       std::string(", src_base + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 64);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 8)
                       .render(R"(
            dst_ptr += ${offset};
            src_base += (IH * IW);
        }
    }
    for (; n < N; n += 4) {
        const int oh = n / OW, ow = n % OW;
        const int32_t *src_base =
            (const int32_t *)(src + oh * 2 * IW * 4 + ow * 2 * 4);
        TINYNN_ASSERT(oh + 1 == OH);
        if (OW - ow >= 4) {
            for (int ic = 0; ic < IC; ++ic) {
        )");
        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * (FW / 2) * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", 32);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 4)
                       .render(R"(
                dst_ptr += ${offset};
                src_base += (IH * IW);
            }
        )");
        res += R"(
        } else {
            const int part0 = OW - ow;
            for (int ic = 0; ic < IC; ++ic) {
                )";
        for (int fh = 0; fh < FH; ++fh) {
            for (int fw = 0; fw < FW / 2; ++fw) {
                res += std::string("memcpy(dst_ptr + ") +
                       std::to_string(fh * (FW / 2) * 8 + fw * 8) +
                       std::string(", src_base + IW * ") + std::to_string(fh) +
                       std::string(" + ") + std::to_string(fw * 2) +
                       std::string(", part0 * 2 * 4);");
            }
        }
        res += StringTemplate::StringTemplateArgs()
                       .add("offset", FH * FW * 4)
                       .render(R"(
                dst_ptr += ${offset};
                src_base += (IH * IW);
            }
        }
    }
}
        )");
    }
    return res;
}
}  // namespace

bool ConvBiasIm2colI8mmNCHW44::IsAvailable(TContext* ctx) const {
    bool param_value_ok =
            ctx->getAttrUInt("kernel_h") == ctx->getAttrUInt("kernel_w") &&
            ctx->getAttrUInt("stride_h") == ctx->getAttrUInt("stride_w") &&
            (ctx->getAttrUInt("stride_w") == 1 || ctx->getAttrUInt("stride_w") == 2) &&
            ctx->getAttrUInt("pad_h") == ctx->getAttrUInt("pad_w") &&
            ctx->getAttrUInt("dilate_h") == 1 && ctx->getAttrUInt("dilate_w") == 1;
    bool param_mode_ok = (ctx->getAttrStr("sparse") == "DENSE" ||
                          (ctx->getAttrStr("sparse") == "GROUP" &&
                           ctx->getAttrOprand("operand:1").shape.size() ==
                                   7 /*reject channel wise whose dimension is 6*/)) &&
                         ctx->getAttrStr("format") == "NCHW44" &&
                         ctx->getAttrStr("mode") == "CROSS_CORRELATION";
    bool noline_ok = !ctx->haveAttr("nonlineMode") ||
                     ctx->getAttrStr("nonlineMode") == "IDENTITY" ||
                     ctx->getAttrStr("nonlineMode") == "RELU" ||
                     ctx->getAttrStr("nonlineMode") == "H_SWISH";

    bool type_ok = is_qint8_conv_dtype(ctx, true);

    bool layout_ok = ctx->getAttrOprand("operand:0").shape.size() == 5 &&
                     ctx->getAttrOprand("operand:0").shape[4] == 4;
    bool bias_ok = !is_bias(ctx) || is_channel_broadcast_bias(ctx);
    return param_value_ok && param_mode_ok && type_ok && noline_ok && layout_ok &&
           bias_ok;
}

std::string ConvBiasIm2colI8mmNCHW44::GetKernelSymbol(TContext* ctx) const {
    return "Arm64_kernel_im2col_i8mm_m8n12k8_" + ConvImpl::GetKernelSymbol(ctx);
}

std::string ConvBiasIm2colI8mmNCHW44::GetInitBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const std::string group_str =
            is_group ? "const int group = in_weights->layout.dims[0];"
                     : "const int group = 1;";
    const int oc_idx = is_group ? 1 : 0;
    writer << m_inner_gemm.GetPackASignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackAWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << "#define PACKED_IC 4\n";
    writer << "#define PACKED_OC 4\n";
    writer << GenCommonRet() << " " << GetInitSignature(ctx);
    const uint32_t nr_out_weight = 1;
    const std::string common_def = StringTemplate::StringTemplateArgs()
                                           .add("group", group_str)
                                           .add("oc_idx", oc_idx)
                                           .render(R"(
        Tensor* in_weights = inputs[1];
        ${group}
        const int ymax = in_weights->layout.dims[${oc_idx}] * PACKED_OC;
        const int kmax = in_weights->layout.dims[${oc_idx} + 1] * in_weights->layout.dims[${oc_idx} + 2] * in_weights->layout.dims[${oc_idx} + 3] * PACKED_IC;
        const int ldin = kmax * PACKED_OC;
    )");
    const std::string fill_weight_attr =
            R"(
        out_weights->layout.nr_dim = 2;
        out_weights->layout.dims[0] = group;
        out_weights->layout.dims[1] = )" +
            m_inner_gemm.GetPackAWorkspaceSymbol(inner_ctx.get()) +
            R"((0, ymax, 0, kmax);
        out_weights->layout.stride[0] = out_weights->layout.dims[1];
        out_weights->layout.stride[1] = 1;
        out_weights->dtype.type_enum = TinyNN_QINT8;
        out_weights->name = in_weights->name;
        out_weights->dtype.param.scale = in_weights->dtype.param.scale;
    )";
    const std::string fill_weight_transform =
            StringTemplate::StringTemplateArgs()
                    .add("packa_sym", m_inner_gemm.GetPackASymbol(inner_ctx.get()))
                    .render(
                            R"(    
        int8_t* outptr = out_weights->ptr;
        int8_t* inptr = in_weights->ptr;
        ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
        for (int i = 1; i < group; ++i) {
            inptr += in_weights->layout.stride[0];
            outptr += out_weights->layout.stride[0];
            ${packa_sym}(outptr, inptr, ldin, 0, ymax, 0, kmax);
        }
    )");
    writer << StringTemplate::render_init_body(
            nr_out_weight, fill_weight_attr, fill_weight_transform, common_def);
    writer << "\n#undef PACKED_IC\n";
    writer << "#undef PACKED_OC\n";

    return writer.str();
}

std::string ConvBiasIm2colI8mmNCHW44::GetWorkspaceBodyCondition(
        TContext* ctx, bool jit) const {
    std::stringstream ss;
    auto inner_ctx = GetInnerCtx(ctx);
    const bool is_group = ctx->getAttrStr("sparse") == "GROUP";
    const std::string group_str = is_group
                                        ? "const int group = inputs[1]->layout.dims[0];"
                                        : "const int group = 1;";
    if (jit) {
        ss << m_inner_gemm.GetPackBWorkspaceBody(inner_ctx.get()) << ";\n";
    } else {
        ss << "extern " << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get())
           << ";\n";
    }
    ss << "#define PACKED_IC 4\n";
    ss << GenCommonRet() << " " << GetWorkspaceSignature(ctx);
    std::string workspace_temp = R"({
        TINYNN_ASSERT(workspace);
        TINYNN_ASSERT(${stride_h} == ${stride_w});
        TINYNN_ASSERT(${kernel_h} == ${kernel_w});
        ${group}
        const Layout src_layout = inputs[0]->layout;
        const size_t IC = src_layout.dims[1] / group * PACKED_IC;
        const size_t IH = src_layout.dims[2], IW = src_layout.dims[3];

        const size_t padded_IH = IH + 2 * ${pad_h};
        const size_t padded_IW = IW + 2 * ${pad_w};
        size_t pad_size = 0;
        if ((${pad_h} != 0) || (${pad_w} != 0)){
            pad_size = IC * padded_IH * padded_IW * sizeof(int8_t);
        }

        const size_t OH = (padded_IH - ${kernel_h}) / ${stride_h} + 1;
        const size_t OW = (padded_IW - ${kernel_w}) / ${stride_w} + 1;
        const size_t K = IC * ${kernel_h} * ${kernel_w}, N = OH * OW;
        size_t im2col_size = 0;
        if ((${kernel_h} != 1 && ${stride_h} == 1) || (${stride_h} == 2 && OW < 12)){
            im2col_size = K * N * sizeof(int8_t);
        }

        *workspace = pad_size + im2col_size + ${packb_workspace_sym}(0, N, 0, K);
        return TinyNN_SUCCESS;
    })";
    ss << StringTemplate::StringTemplateArgs(ctx)
                    .add("packb_workspace_sym",
                         m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                    .add("group", group_str)
                    .add_ctx_int("pad_h")
                    .add_ctx_int("pad_w")
                    .add_ctx_int("kernel_h")
                    .add_ctx_int("kernel_w")
                    .add_ctx_int("stride_h")
                    .add_ctx_int("stride_w")
                    .render(workspace_temp);
    ss << "\n#undef PACKED_IC\n";
    return ss.str();
}

std::vector<KernelObj> ConvBiasIm2colI8mmNCHW44::GetDependInternalSymbol(
        TContext* ctx) const {
    auto inner_ctx = GetInnerCtx(ctx);

    return {
            {m_inner_gemm.GetKernelSymbol(inner_ctx.get()),
             m_inner_gemm.GetKernelBody(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardBegin(inner_ctx.get()),
             m_inner_gemm.GetBodyGuardEnd(inner_ctx.get()),
             m_inner_gemm.GetDependInternalSymbol(inner_ctx.get())}};
}

std::shared_ptr<TContext> ConvBiasIm2colI8mmNCHW44::GetInnerCtx(TContext* ctx) const {
    auto inner_ctx = std::make_shared<CodeGenContext>();
    if (ctx->haveAttr("nonlineMode")) {
        inner_ctx->setAttr("nonlineMode", CCAttr(ctx->getAttrStr("nonlineMode")));
    }
    inner_ctx->setAttr("with_bias", ConvImpl::is_bias(ctx));
    inner_ctx->setAttr("transposeA", false);
    inner_ctx->setAttr("transposeB", false);
    inner_ctx->setAttr("format", "MK4");
    inner_ctx->setAttr("dtype", ctx->getAttrOprand("operand:0").dtype);
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    inner_ctx->setAttr("last_dtype", last_dtype_str);
    return inner_ctx;
}

std::string ConvBiasIm2colI8mmNCHW44::GetKernelBody(TContext* ctx) const {
    std::stringstream writer;
    auto inner_ctx = GetInnerCtx(ctx);
    writer << m_inner_gemm.GetPackBWorkspaceSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetNakedKernelSignature(inner_ctx.get()) << ";\n";
    writer << m_inner_gemm.GetPackBSignature(inner_ctx.get()) << ";\n";
    writer << "#define PACKED_IC 4\n";
    writer << "#define PACKED_OC 4\n";
    const bool need_pad = (ctx->getAttrInt("pad_h") || ctx->getAttrInt("pad_w")),
               need_im2col =
                       (ctx->getAttrInt("kernel_h") != 1 ||
                        ctx->getAttrInt("kernel_w") != 1 ||
                        ctx->getAttrInt("stride_h") != 1 ||
                        ctx->getAttrInt("stride_w") != 1);
    if (need_pad) {
        writer << pad_src();
    }
    if (need_im2col) {
        if (ctx->getAttrInt("stride_h") == 1 && ctx->getAttrInt("stride_w") == 1) {
            writer << im2col_s1();
        } else {
            CC_ASSERT(
                    ctx->getAttrInt("stride_h") == 2 &&
                    ctx->getAttrInt("stride_w") == 2);
            writer << "#include <arm_neon.h>\n";
            writer << im2col_s2();
            writer << fuse_im2col_packB_s2(ctx);
        }
    }
    writer << GenCommonRet() << " " << GetKernelSignature(ctx);
    std::string bias_ptr_str = is_bias(ctx) ? "inputs[2]->ptr;" : "0;";
    auto last_dtype = Utils::get_last_operand(ctx).dtype;
    auto last_dtype_str = SymbolHelper::gen_valid_dtype(last_dtype);
    std::string dst_specifier = Utils::cvt_dtype_specifier(last_dtype_str);
    std::string temp_body = R"({
    int8_t* input_data = inputs[0]->ptr;
    ${dst_specifier}* output_data = outputs[0]->ptr;

    Layout in_layout = inputs[0]->layout;
    Layout weight_layout = inputs[1]->layout;
    const int group = weight_layout.dims[0];
    Layout out_layout = outputs[0]->layout;
    const int in_n = in_layout.dims[0];
    const int in_c = in_layout.dims[1] / group * in_layout.dims[4];
    const int in_h = in_layout.dims[2];
    const int in_w = in_layout.dims[3];
    const float src_scale = inputs[0]->dtype.param.scale;
    const float flt_scale = inputs[1]->dtype.param.scale;
    const float dst_scale = outputs[0]->dtype.param.scale;
    const float temp_scale = src_scale * flt_scale;
    const float dst_scale_inv = 1.f / dst_scale;
    const float scale = src_scale * flt_scale * dst_scale_inv;

    const int out_c = out_layout.dims[1] / group * out_layout.dims[4];
    const int out_h = out_layout.dims[2];
    const int out_w = out_layout.dims[3];
    const size_t N = out_h * out_w, M = out_c, K = in_c * ${kernel_h} * ${kernel_w};

    const int LDC = out_h * out_w * PACKED_OC;
    const int LDB = out_h * out_w * PACKED_OC;

    const size_t padded_ih = in_h + 2 * ${pad_h}, padded_iw = in_w + 2 * ${pad_w};
    size_t pad_size = 0, im2col_size = 0;
    if ((${pad_h} != 0) || (${pad_w} != 0)) {
        pad_size = in_c * padded_ih * padded_iw * sizeof(int8_t);
    }
    if ((${kernel_h} != 1 && ${stride_h} == 1) || (${stride_h} == 2 && out_w < 12)){
        im2col_size = K * N * sizeof(int8_t);
    }
    void *pad_ws = workspace->ptr;
    void *im2col_ws = pad_ws + pad_size;
    void *packb_ws = im2col_ws + im2col_size;
    const int pad_h = ${pad_h}, pad_w = ${pad_w}, kernel_h = ${kernel_h}, kernel_w = ${kernel_w};)";
    if (ctx->getAttrInt("stride_h") == 1) {
        temp_body += R"(
    for (int n_idx = 0; n_idx < in_n; ++n_idx) {
        int32_t* bias_data = ${bias_ptr_str};
        int8_t* weight_data = inputs[1]->ptr;
        for (int g = 0; g < group; ++g) {
            ${exec_pad}
            ${exec_im2col}
            ${packb_sym}(packb_ws, im2col_ws, LDB, 0, N, 0, K);
            ${naked_kern_sym}(weight_data, packb_ws, output_data, LDC, M, N, K, bias_data, NULL, scale, temp_scale, dst_scale_inv);
            weight_data += weight_layout.stride[0];
            bias_data += out_c;
            input_data += in_c * in_h * in_w;
            output_data += out_c * out_h * out_w;
        }
    }
    return TinyNN_SUCCESS;
})";
    } else {
        CC_ASSERT(ctx->getAttrInt("stride_h") == 2);
        temp_body += R"(
    //! Because of the implementation of function `fuse_im2col_packB_s2` assumes that the 12 elements span at most two lines.
    if (out_w < 12) {
        for (int n_idx = 0; n_idx < in_n; ++n_idx) {
            int32_t* bias_data = ${bias_ptr_str};
            int8_t* weight_data = inputs[1]->ptr;
            for (int g = 0; g < group; ++g) {
                ${exec_pad}
                ${exec_im2col}
                ${packb_sym}(packb_ws, im2col_ws, LDB, 0, N, 0, K);
                ${naked_kern_sym}(weight_data, packb_ws, output_data, LDC, M, N, K, bias_data, NULL, scale, temp_scale, dst_scale_inv);
                weight_data += weight_layout.stride[0];
                bias_data += out_c;
                input_data += in_c * in_h * in_w;
                output_data += out_c * out_h * out_w;
            }
        }
    } else {
        for (int n_idx = 0; n_idx < in_n; ++n_idx) {
            int32_t* bias_data = ${bias_ptr_str};
            int8_t* weight_data = inputs[1]->ptr;
            for (int g = 0; g < group; ++g) {
                ${exec_pad}
                fuse_im2col_packB_s2(pad_ws, packb_ws, in_c / PACKED_IC, padded_ih, padded_iw, out_h, out_w);
                ${naked_kern_sym}(weight_data, packb_ws, output_data, LDC, M, N, K, bias_data, NULL, scale, temp_scale, dst_scale_inv);
                weight_data += weight_layout.stride[0];
                bias_data += out_c;
                input_data += in_c * in_h * in_w;
                output_data += out_c * out_h * out_w;
            }
        }
    }
    return TinyNN_SUCCESS;
})";
    }
    std::string exec_pad =
            (need_pad ? std::string(R"(
            pad_src(input_data, pad_ws, in_c / PACKED_IC, in_h, in_w, pad_h, pad_w);)")
                      : std::string(R"(
            pad_ws = input_data;)"));
    std::string exec_im2col =
            (need_im2col ? std::string(R"(
            im2col(pad_ws, im2col_ws, in_c / PACKED_IC, out_h, out_w, kernel_h, kernel_w,
            padded_ih, padded_iw);)")
                         : std::string(R"( im2col_ws = pad_ws;)"));

    writer << StringTemplate::StringTemplateArgs(ctx)
                      .add("bias_ptr_str", bias_ptr_str)
                      .add("packb_size_sym",
                           m_inner_gemm.GetPackBWorkspaceSymbol(inner_ctx.get()))
                      .add("packb_sym", m_inner_gemm.GetPackBSymbol(inner_ctx.get()))
                      .add("naked_kern_sym",
                           m_inner_gemm.GetNakedKernelSymbol(inner_ctx.get()))
                      .add("dst_specifier", dst_specifier)
                      .add_ctx_int("pad_h")
                      .add_ctx_int("pad_w")
                      .add_ctx_int("kernel_h")
                      .add_ctx_int("kernel_w")
                      .add_ctx_int("stride_h")
                      .add_ctx_int("stride_w")
                      .add("exec_pad", exec_pad)
                      .add("exec_im2col", exec_im2col)
                      .render(temp_body);
    writer << "\n#undef PACKED_IC\n";
    writer << "#undef PACKED_OC\n";
    return writer.str();
}

}  // namespace Arm64
}  // namespace KernelGen
}  // namespace megcc

// vim: syntax=cpp.doxygen
