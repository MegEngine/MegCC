/**
 * \file
 * compiler/lib/KernelGen/GeneralIntrinsic/GuassianBlur.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

#include <algorithm>
#include <sstream>

#include "GaussianBlur.h"
#include "Utils/StringTemplate.h"
#include "Utils/SymbolHelper.h"
#include "Utils/Utils.h"

using namespace megcc;
using namespace KernelGen;
using namespace GeneralIntrinsic;

bool GaussianBlurKernel::IsCVAvailable(TContext* context) const {
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    bool dtype_ok = Utils::is_int_dtype(src_dtype, 8);
    std::string mode = context->getAttrStr("border_mode");
    bool bmode_ok =
            (mode == "CONSTANT" || mode == "REFLECT" || mode == "REFLECT_101" ||
             mode == "REPLICATE");
    return dtype_ok && bmode_ok;
}

//! kernel gen
std::string GaussianBlurKernel::GetCVKernelSubSymbol(TContext* context) const {
    std::stringstream ss;
    auto src_dtype = context->getAttrOprand("operand:0").dtype;
    std::string bmode = context->getAttrStr("border_mode");
    std::transform(bmode.begin(), bmode.end(), bmode.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    ss << "tinycv_guassian_blur_" << bmode << "_" << src_dtype;
    return ss.str();
}

std::string GaussianBlurKernel::GetCVKernelSignature(TContext* context) const {
    return GetCVKernelSymbol(context) +
           "(const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w, "
           "double sigma1, double sigma2)";
}

namespace {
std::string gen_border_interpolate(const std::string& bmode) {
    std::string body_temp = R"(
            static inline int border_interpolate(int p, const int len){
                if ((unsigned)p >= (unsigned)len){
                    ${core_temp}
                }
                return p;
            }
        )";
    std::string core_temp;
    if (bmode == "REFLECT") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1;
                else
                    p = len - 1 - (p - len);
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REFLECT_101") {
        core_temp = R"(
            if (len == 1)
                return 0;
            do {
                if (p < 0)
                    p = -p - 1 + 1;
                else
                    p = len - 1 - (p - len) - 1;
            } while ((unsigned)p >= (unsigned)len);
        )";
    } else if (bmode == "REPLICATE") {
        core_temp = R"(
            p = p < 0 ? 0 : len - 1;
        )";
    } else if (bmode == "CONSTANT") {
        core_temp = R"(
            p = -1;
        )";
    } else if (bmode == "WRAP") {
        core_temp = R"(
            if (p < 0)
                p -= ((p - len + 1) / len) * len;
            
            while (p >= len) {
                p -= len;
            }
        )";
    } else {
        CC_ABORT << "no support bmode " << bmode << "\n";
    }
    return StringTemplate::StringTemplateArgs()
            .add("core_temp", core_temp)
            .render(body_temp);
}

std::string gen_create_gaussian_kernels() {
    std::string temp_str = R"(
        static inline void getGaussianKernel(size_t n, double sigma, TinyMat* kernel) {
#define SMALL_GAUSSIAN_SIZE  7
            static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
                    {1.f},
                    {0.25f, 0.5f, 0.25f},
                    {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
                    {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f, 0.03125f}};

            const float* fixed_kernel = n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0
                                            ? small_gaussian_tab[n >> 1]
                                            : 0;
#undef SMALL_GAUSSIAN_SIZE

            float* c = (float*)(kernel->data);

            double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
            double scale2X = -0.5 / (sigmaX * sigmaX);
            double sum = 0;

            int i;
            for (i = 0; i < (int)n; i++) {
                double x = i - (n - 1) * 0.5;
                double t = fixed_kernel ? (double)fixed_kernel[i] : exp(scale2X * x * x);
                c[i] = (float)t;
                sum += c[i];
            }

            sum = 1. / sum;
            for (i = 0; i < (int)n; i++)
                c[i] = (float)(c[i] * sum);
        }

        static inline void calRowsAndCols(int* rows, int* cols, double sigma1, double sigma2){
            if (sigma2 <= 0)
                sigma2 = sigma1;

            if (*cols <= 0 && sigma1 > 0) {
                double num = sigma1 * 4 * 2 + 1;
                num = (int)(num + (num >= 0 ? 0.5 : -0.5));
                *cols = ((int)num) | 1;
            }
            if (*rows <= 0 && sigma2 > 0) {
                double num = sigma2 * 4 * 2 + 1;
                num = (int)(num + (num >= 0 ? 0.5 : -0.5));
                *rows = ((int)num) | 1;
            }

            TINYNN_ASSERT(
                    (*cols > 0 && *cols % 2 == 1 && *rows > 0 &&
                    *rows % 2 == 1));
        }

        static inline void createGaussianKernels(
                TinyMat* kx, TinyMat* ky, int rows, int cols, double sigma1, double sigma2) {
            if (sigma2 <= 0)
                sigma2 = sigma1;

            sigma1 = sigma1 > 0. ? sigma1 : 0.;
            sigma2 = sigma2 > 0. ? sigma2 : 0.;

            getGaussianKernel(cols, sigma1, kx);
            if (rows == cols && sigma1 - sigma2 < 1e-7 && sigma1 - sigma2 > -1e-7)
                memcpy(ky->data, kx->data, ky->cols * sizeof(float));
            else
                getGaussianKernel(rows, sigma2, ky);
        }
    )";
    return temp_str;
}

std::string gen_kern_func(const std::string& bmode) {
    std::string helper = R"(
        typedef struct RowFilter{
            TinyMat* m_kernel;
            int m_anchor, ksize;
            void(*m_filter)(struct RowFilter* self, const uint8_t* src, uint8_t* dst, int width, int cn);
        } RowFilter;

        typedef struct ColFilter{
            TinyMat* m_kernel;
            int m_anchor, ksize;
            void(*m_filter)(struct ColFilter *self, const uint8_t** src, uint8_t* dst, int dststep, int count, int width);
            uint8_t(*cast_op)(int);
        } ColFilter;

        static void symmRowSmallFilter(RowFilter* self, const uint8_t* src, uint8_t* dst, int width, int cn) {
            int ksize2 = self->ksize / 2, ksize2n = ksize2 * cn;
            const int* kx = (int*)self->m_kernel->data + ksize2;
            int* D = (int*)dst;
            int i = 0, j, k;

            //! The center
            const uint8_t* S = (uint8_t*)src + i + ksize2n;
            width *= cn;

            if (self->ksize == 1 && kx[0] == 1) {
                for (; i <= width - 2; i += 2) {
                    int s0 = S[i], s1 = S[i + 1];
                    D[i] = s0;
                    D[i + 1] = s1;
                }
                S += i;
            } else if (self->ksize == 3) {
                int k0 = kx[0], k1 = kx[1];
                for (; i <= width - 2; i += 2, S += 2) {
                    int s0 = S[0] * k0 + (S[-cn] + S[cn]) * k1,
                    s1 = S[1] * k0 + (S[1 - cn] + S[1 + cn]) * k1;
                    D[i] = s0;
                    D[i + 1] = s1;
                }
            } else if (self->ksize == 5) {
                int k0 = kx[0], k1 = kx[1], k2 = kx[2];
                for (; i <= width - 2; i += 2, S += 2) {
                    int s0 = S[0] * k0 + (S[-cn] + S[cn]) * k1 +
                            (S[-cn * 2] + S[cn * 2]) * k2;
                    int s1 = S[1] * k0 + (S[1 - cn] + S[1 + cn]) * k1 +
                            (S[1 - cn * 2] + S[1 + cn * 2]) * k2;
                    D[i] = s0;
                    D[i + 1] = s1;
                }
            }

            for (; i < width; i++, S++) {
                int s0 = kx[0] * S[0];
                for (k = 1, j = cn; k <= ksize2; k++, j += cn)
                    s0 += kx[k] * (S[j] + S[-j]);
                D[i] = s0;
            }
        }

        static void rowFilter(RowFilter* self, const uint8_t* src, uint8_t* dst, int width, int cn){
            const int *kx = (int*)self->m_kernel->data;
            const uint8_t *S;
            int *D = (int*)dst;
            int i = 0, k;
            width *= cn;

            for(; i + 3 < width; i += 4){
                S = src + i;
                int s0 = kx[0] * S[0], s1 = kx[0] * S[1], s2 = kx[0] * S[2], s3 = kx[0] * S[3];
                for (k = 1; k < self->ksize; ++k) {
                    S += cn;
                    s0 += kx[k] * S[0];
                    s1 += kx[k] * S[1];
                    s2 += kx[k] * S[2];
                    s3 += kx[k] * S[3];
                }
                D[i] = s0, D[i + 1] = s1, D[i + 2] = s2, D[i + 3] = s3;
            }

            for(; i < width; ++i){
                S = src + i;
                int s0 = kx[0] * S[0];
                for (k = 1; k < self->ksize; ++k) {
                    S += cn;
                    s0 += kx[k] * S[0];
                }
                D[i] = s0;
            }
        }

        static void symmColumnSmallFilter(ColFilter *self, const uint8_t** src, uint8_t* dst, int dststep, int count, int width) {
            int ksize2 = self->ksize / 2;
            const int *ky = (int*)self->m_kernel->data + ksize2;
            int i;
            int f0 = ky[0], f1 = ky[1];
            src += ksize2;

            for (; count > 0; count--, dst += dststep, src++) {
                uint8_t* D = (uint8_t*)dst;
                i = 0;
                if (count == 0)
                    break;
                const int* S0 = (const int*)src[-1];
                const int* S1 = (const int*)src[0];
                const int* S2 = (const int*)src[1];

                {
                    for (; i <= width - 4; i += 4) {
                        int s0 = (S0[i] + S2[i]) * f1 + S1[i] * f0;
                        int s1 = (S0[i + 1] + S2[i + 1]) * f1 + S1[i + 1] * f0;
                        D[i] = self->cast_op(s0);
                        D[i + 1] = self->cast_op(s1);

                        s0 = (S0[i + 2] + S2[i + 2]) * f1 + S1[i + 2] * f0;
                        s1 = (S0[i + 3] + S2[i + 3]) * f1 + S1[i + 3] * f0;
                        D[i + 2] = self->cast_op(s0);
                        D[i + 3] = self->cast_op(s1);
                    }
                    for (; i < width; i++) {
                        int s0 = (S0[i] + S2[i]) * f1 + S1[i] * f0;
                        D[i] = self->cast_op(s0);
                    }
                }
            }
        }

        static void columnFilter(ColFilter *self, const uint8_t** src, uint8_t* dst, int dststep, int count, int width) {
            const int* ky = (int*)self->m_kernel->data;
            int i = 0, k;
            {
                for (; count > 0; count--, dst += dststep, src++) {
                    uint8_t* D = (uint8_t*)dst;
                    i = 0;
                    for (; i <= width - 4; i += 4) {
                        int f = ky[0];
                        const int* S = (const int*)src[0] + i;
                        int s0 = f * S[0], s1 = f * S[1], s2 = f * S[2], s3 = f * S[3];

                        for (k = 1; k < self->ksize; k++) {
                            S = (const int*)src[k] + i;
                            f = ky[k];
                            s0 += f * S[0];
                            s1 += f * S[1];
                            s2 += f * S[2];
                            s3 += f * S[3];
                        }

                        D[i] = self->cast_op(s0);
                        D[i + 1] = self->cast_op(s1);
                        D[i + 2] = self->cast_op(s2);
                        D[i + 3] = self->cast_op(s3);
                    }
                    for (; i < width; i++) {
                        int s0 = 0;
                        for (k = 0; k < self->ksize; k++) {
                            s0 += ky[k] * ((const int*)src[k])[i];
                        }
                        D[i] = self->cast_op(s0);
                    }
                }
            }
        }

        static uint8_t castOp(int x){
            int delta = (1 << 15);
            int res = ((x + delta) >> 16);
            return (uint8_t)((unsigned)res <= 255 ? res : (res < 0 ? 0 : 255));
        }

        typedef struct FilterEngine {
            RowFilter* m_row_filter;
            ColFilter* m_column_filter;
            size_t m_ch;
            
            //! the size of the kernel
            size_t m_ksize_row, m_ksize_col;

            //! the center of kernel, e.g GuassianBlur m_anchor is (kernel_row/2,
            //! kernel_column/2)
            size_t m_anchor_x, m_anchor_y;

            //! the whole size.
            size_t m_whole_h, m_whole_w;
            //! store the border value, if sizeof(src_type) >= 4,
            int* m_border_table;
            //! nr of border value
            int m_border_elem_size;

            //! the step of the buffer data.
            int m_buf_step;

            //! store the total row if the border is BORDER_CONSTANT, the size is
            //! image_width + kernel_width - 1, which include the row and the border.
            uint8_t* m_const_border_row;
            //! store the total row if the border is not BORDER_CONSTANT
            uint8_t* m_src_row;

            //! store the kernel_height rows data.
            uint8_t* m_ring_buf;

            //! the border left width, equal to m_anchor.x
            int m_left_width;
            //! equal to m_ksize.width() - m_left_width - 1
            int m_right_width;

            void(*ctor)(struct FilterEngine* self, RowFilter* row_filter, ColFilter* column_filter, const size_t ch, const size_t cols, const size_t rows);
            int (*proceed)(
                struct FilterEngine* self, const uint8_t* src, int srcstep, int count, uint8_t* dst, int dststep);
            void (*dtor)(struct FilterEngine* self);
        }FilterEngine;

#define VEC_ALIGN 16
        static inline size_t align_size(size_t sz, int n){
            TINYNN_ASSERT(((n & (n-1)) == 0));
            return (sz + n - 1) & -n;
        }

        static inline uint8_t* align_ptr(uint8_t* ptr, int n) {
            return (uint8_t*)(((size_t)ptr + n - 1) & -n);
        }

        static void filterEngineCtor(FilterEngine* self, RowFilter* row_filter, ColFilter* column_filter, const size_t ch, const size_t cols, const size_t rows){
            self->m_row_filter = row_filter;
            self->m_column_filter = column_filter;
            self->m_ch = ch;

            self->m_ksize_col = self->m_row_filter->ksize;
            self->m_ksize_row = self->m_column_filter->ksize;
            self->m_anchor_x = self->m_row_filter->m_anchor;
            self->m_anchor_y = self->m_column_filter->m_anchor;
            self->m_buf_step = 0;
            
            self->m_whole_w = cols;
            self->m_whole_h = rows;

            int element_size = (int)sizeof(uint8_t) * self->m_ch;
            int buf_elem_size = (int)sizeof(int) * self->m_ch;

            int cn = self->m_ch;
            self->m_src_row = (uint8_t*)tinynn_malloc(element_size * (self->m_whole_w + self->m_ksize_col - 1));
            
            ${init_const_border_row}

            self->m_buf_step = buf_elem_size *
                        (int)align_size(self->m_whole_w + self->m_ksize_col - 1, VEC_ALIGN);
            self->m_ring_buf = (uint8_t*)tinynn_malloc(self->m_buf_step * self->m_ksize_row + VEC_ALIGN);
            self->m_left_width = self->m_anchor_x;
            self->m_right_width = self->m_ksize_col - self->m_anchor_x - 1;

            int src_elem_size = (int)sizeof(uint8_t) * self->m_ch;
            self->m_border_elem_size = src_elem_size;
            ${init_non_const_border_table}
        }

        static int filterEngineProceed(
                FilterEngine* self, const uint8_t* src, int srcstep, int count, uint8_t* dst, int dststep) {
            int src_elem_size = (int)(sizeof(uint8_t) * self->m_ch);
            int dy = 0, i = 0;

            int row_count = 0;
            int start_y = 0;
            uint8_t **buf_rows = (uint8_t**)tinynn_malloc(sizeof(uint8_t*) * self->m_ksize_row);
            for (;; dst += dststep * i, dy += i) {
                int dcount = self->m_ksize_row - self->m_anchor_y - start_y - row_count;
                dcount = dcount > 0 ? dcount : 1;
                dcount = dcount < count ? dcount : count;
                count -= dcount;
                for (; dcount-- > 0; src += srcstep) {
                    int bi = (start_y + row_count) % self->m_ksize_row;
                    uint8_t* brow = align_ptr(self->m_ring_buf, VEC_ALIGN) + bi * self->m_buf_step;
                    uint8_t* row = self->m_src_row;

                    if (++row_count > (int)self->m_ksize_row) {
                        --row_count;
                        ++start_y;
                    }

                    memcpy(row + self->m_left_width * src_elem_size, src,
                        self->m_whole_w * src_elem_size);

                    ${set_non_const_border_value_by_border_table}

                    self->m_row_filter->m_filter(self->m_row_filter, row, brow, self->m_whole_w, self->m_ch);
                }

                int max_i = 
                        self->m_ksize_row < self->m_whole_h - dy + (self->m_ksize_row - 1) ? 
                        self->m_ksize_row : self->m_whole_h - dy + (self->m_ksize_row - 1);
                for (i = 0; i < max_i; i++) {
                    int src_y = border_interpolate(
                            dy + i - self->m_anchor_y, self->m_whole_h);
                    if (src_y < 0) {
                        TINYNN_ASSERT(self->m_const_border_row);
                        buf_rows[i] = align_ptr(self->m_const_border_row, VEC_ALIGN);
                    } else {
                        TINYNN_ASSERT((src_y >= start_y));
                        if (src_y >= start_y + row_count) {
                            break;
                        }
                        int bi = src_y % self->m_ksize_row;
                        buf_rows[i] = align_ptr(self->m_ring_buf, VEC_ALIGN) + bi * self->m_buf_step;
                    }
                }
                if (i < (int)(self->m_ksize_row)) {
                    break;
                }
                i -= self->m_ksize_row - 1;
                self->m_column_filter->m_filter(
                        self->m_column_filter, (const uint8_t**)buf_rows, dst, dststep, i,
                        self->m_whole_w * self->m_ch);
            }

            tinynn_free(buf_rows);

            return dy;
        }
#undef VEC_ALIGN
        static void filterEngineDtor(FilterEngine* self){
            tinynn_free(self->m_src_row);
            tinynn_free(self->m_ring_buf);
            if(self->m_const_border_row){
                tinynn_free(self->m_const_border_row);
            }
            if(self->m_border_table){
                tinynn_free(self->m_border_table);
            }
        }
        )";
    int is_constant = bmode == "CONSTANT";
    std::string init_const_border_row =
            (is_constant ? R"(
            memset(self->m_src_row, 0, element_size * (self->m_whole_w + self->m_ksize_col - 1));
            self->m_const_border_row = (uint8_t*)tinynn_malloc(
                    buf_elem_size *
                    (self->m_whole_w + self->m_ksize_col - 1 + VEC_ALIGN));
            memset(self->m_const_border_row, 0, buf_elem_size *
                    (self->m_whole_w + self->m_ksize_col - 1 + VEC_ALIGN));
    )"
                         : "self->m_const_border_row = NULL;");
    std::string init_non_const_border_table = (!is_constant)
                                                    ? R"(
            int border_length = (int)(self->m_ksize_col - 1) > (int)1 ? (int)(self->m_ksize_col - 1) : (int)1;
            self->m_border_table = (int*)tinynn_malloc(sizeof(int) * border_length * self->m_border_elem_size);
            //! calc the index of the border value, we will not calc it when
            //! process border each time
            if (self->m_left_width > 0 || self->m_right_width > 0) {
                for (int i = 0; i < self->m_left_width; i++) {
                    int p0 = border_interpolate(
                                    i - self->m_left_width, self->m_whole_w) *
                            self->m_border_elem_size;
                    for (int j = 0; j < self->m_border_elem_size; j++)
                        self->m_border_table[i * self->m_border_elem_size + j] = p0 + j;
                }

                for (int i = 0; i < self->m_right_width; i++) {
                    int p0 = border_interpolate(
                                    self->m_whole_w + i, self->m_whole_w) *
                            self->m_border_elem_size;
                    for (int j = 0; j < self->m_border_elem_size; j++)
                        self->m_border_table[(i + self->m_left_width) * self->m_border_elem_size + j] =
                                p0 + j;
                }
            }
    )"
                                                    : "self->m_border_table = NULL;";
    std::string set_non_const_border_value_by_border_table =
            (!is_constant ? R"(
                    TINYNN_ASSERT(self->m_border_table);
                    if (self->m_left_width > 0 || self->m_right_width > 0) {
                        for (int i = 0; i < self->m_left_width * src_elem_size; i++)
                            row[i] = src[self->m_border_table[i]];
                        for (int i = 0; i < self->m_right_width * src_elem_size; i++)
                            row[i + (self->m_whole_w + self->m_left_width) * src_elem_size] =
                                    src[self->m_border_table[i + self->m_left_width * src_elem_size]];
                    }
    )"
                          : "");
    return StringTemplate::StringTemplateArgs()
            .add("init_const_border_row", init_const_border_row)
            .add("init_non_const_border_table", init_non_const_border_table)
            .add("set_non_const_border_value_by_border_table",
                 set_non_const_border_value_by_border_table)
            .render(helper);
}
}  // namespace

std::string GaussianBlurKernel::GetCVKernelBody(TContext* context) const {
    auto kernel_sig = GetCVKernelSignature(context);
    std::string bmode = context->getAttrStr("border_mode");
    std::stringstream writer;
    writer << R"(
        #include <math.h>
        #include <stdlib.h>
        #include <string.h>
        #include "gi_int.h"
        #include "tinycv_c.h"
        #include "utils.h"
    )";
    writer << gen_border_interpolate(bmode);
    writer << gen_create_gaussian_kernels();
    writer << gen_kern_func(bmode);

    std::string body_temp = R"(
        void ${kernel_sig} {
            int row = kernel_h, col = kernel_w;
            calRowsAndCols(&row, &col, sigma1, sigma2);
            TinyMat kernel_row = {1, row, 1, NULL};
            TinyMat kernel_col = {1, col, 1, NULL};
            kernel_row.data = (float*)tinynn_malloc(sizeof(float) * row);
            kernel_col.data = (float*)tinynn_malloc(sizeof(float) * col);
            createGaussianKernels(&kernel_col, &kernel_row, row, col, sigma1, sigma2);

            const uint8_t bits = 8;
            for(size_t i = 0; i < kernel_row.cols; ++i){
                ((int*)(kernel_row.data))[i] = (int)(((float*)(kernel_row.data))[i] * (1 << bits));
            }
            for(size_t i = 0; i < kernel_col.cols; ++i){
                ((int*)(kernel_col.data))[i] = (int)(((float*)(kernel_col.data))[i] * (1 << bits));
            }

            RowFilter rf;
            rf.m_kernel = &kernel_col;
            rf.m_anchor = col / 2;
            rf.ksize = col;
            if (rf.ksize <= 5) {
                rf.m_filter = symmRowSmallFilter;
            } else {
                rf.m_filter = rowFilter;
            }

            ColFilter cf;
            cf.m_kernel = &kernel_row;
            cf.m_anchor = row / 2;
            cf.ksize = row;
            if (cf.ksize == 3) {
                cf.m_filter = symmColumnSmallFilter;
            } else {
                cf.m_filter = columnFilter;
            }
            cf.cast_op = castOp;

            FilterEngine fe;
            fe.ctor = filterEngineCtor;
            fe.proceed = filterEngineProceed;
            fe.dtor = filterEngineDtor;

            fe.ctor(&fe, &rf, &cf, src->channels, src->cols, src->rows);
            uint8_t *src_ptr = (uint8_t*)src->data;
            uint8_t *dst_ptr = (uint8_t*)dst->data;
            fe.proceed(&fe, src_ptr, src->cols * src->channels, fe.m_whole_h, dst_ptr, dst->cols * dst->channels);
            fe.dtor(&fe);

            tinynn_free(kernel_row.data);
            tinynn_free(kernel_col.data);
        }
    )";

    writer << StringTemplate::StringTemplateArgs()
                      .add("kernel_sig", kernel_sig)
                      .render(body_temp);
    return writer.str();
}

// vim: syntax=cpp.doxygen
