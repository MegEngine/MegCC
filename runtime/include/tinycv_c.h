#ifndef TINYCV_C_H_
#define TINYCV_C_H_

#include "stdbool.h"
#include "stddef.h"
#include "stdint.h"
#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief the simple Matrix description
 */
typedef struct TinyMat {
    size_t rows;
    size_t cols;
    size_t channels;
    void* data;
} TinyMat;

/**
 * \fn tinycv_transpose_ui8
 * \brief Transpose image.
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 */
void tinycv_transpose_ui8(const TinyMat* src, const TinyMat* dst);

/**
 * \fn tinycv_flip_ui8
 * \brief Flips an image around vertical and/or horizontal axes.
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 * \param[in] vertical Specifying whether the image should be flipped
 * vertically.
 * \param[in] horizontal Specifying whether the image should be flipped
 * horizontally.
 *
 * \warning \c vertical and \c horizontal can be set at the same time.
 */
void tinycv_flip_ui8(
        const TinyMat* src, const TinyMat* dst, bool vertical, bool horizontal);

void tinycv_cvt_rgb2bgr_ui8(const TinyMat* src, const TinyMat* dst);

void tinycv_cvt_yuv2bgr_nv21_ui8(const TinyMat* src, const TinyMat* dst);

void tinycv_cvt_rgb2yuv_ui8(const TinyMat* src, const TinyMat* dst);

void tinycv_cvt_rgb2gray_ui8(const TinyMat* src, const TinyMat* dst);

/**
 * \fn tinycv_resize_linear_ui8
 * \brief Resize an image, The Interpolation Mode is linear
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 * \param[in] imode The Interpolation Mode megcv::InterpolationMode
 */
void tinycv_resize_linear_ui8(const TinyMat* src, const TinyMat* dst);

void tinycv_resize_linear_f32(const TinyMat* src, const TinyMat* dst);

/**
 * \fn tinycv_rotate_ui8
 * \brief Rotate image 90 degree, clockwise indicate the direction.
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 * \param[in] clockwise The rotate direction.
 *
 */
void tinycv_rotate_ui8(const TinyMat* src, const TinyMat* dst, bool clockwise);

/**
 * \fn tinycv_warp_affine_replicate_linear_ui8
 * \brief Applies an affine transformation to an image.
 *  boarder type replicate  `aaaaaa|abcdefgh|hhhhhhh`
 *  interpolationMode linear
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 * \param[in] trans 2X3 transformation matrix
 *
 * \warning Ensure the size of trans is 6
 */
void tinycv_warp_affine_replicate_linear_ui8(
        const TinyMat* src, const TinyMat* dst, const double* trans);

/**
 * \fn tinycv_warp_affine_constant_linear_ui8
 * \brief Applies an affine transformation to an image.
 *  boarder type constant  `vvvvvv|abcdefgh|vvvvvvv`
 *  interpolationMode linear
 * \param[in] src Input mat ptr.
 * \param[out] dst Output mat ptr.
 * \param[in] trans 2X3 transformation matrix
 * \param[in] uint8_t constant val
 *
 * \warning Ensure the size of trans is 6
 */
void tinycv_warp_affine_constant_linear_ui8(
        const TinyMat* src, const TinyMat* dst, const double* trans, uint8_t board);

/**
 * \fn tinycv_roi_copy_ui8
 * \brief Copy ROI region from src to dst
 *
 * Copy [row_from, row_to) x [col_from, col_to) rectangle region from src
 * to dst. src and dst must have the same data type and the same nr. of
 * channels.
 *
 * \warning dst must have size (row_to - row_from) x (col_to - col_from)
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Ouput mat ptr.
 * \param[in] row_from The ROI row start.
 * \param[in] row_to The ROI row end.
 * \param[in] col_from The ROI col start.
 * \param[in] col_to The ROI col end.
 */
void tinycv_roi_copy_ui8(
        const TinyMat* src, const TinyMat* dst, size_t row_from, size_t row_to,
        size_t col_from, size_t col_to);

/**
 * \fn tinycv_gaussian_blur_constant_ui8
 * \brief Gaussian blur on src and store to dst
 *
 *  boarder type constant  `000000|abcdefgh|000000`
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Ouput mat ptr.
 * \param[in] kernel_h Height of gaussian blur kernel.
 * \param[in] kernel_w Width of gaussian blur kernel.
 * \param[in] sigma_x The variance of the Gaussian function in the x-direction(width),
 * used to calculate the kernel.
 * \param[in] sigma_y The variance of the Gaussian function in the y-direction(height),
 * used to calculate the kernel.
 */
void tinycv_gaussian_blur_constant_ui8(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

void tinycv_gaussian_blur_constant_f32(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

/**
 * \fn tinycv_gaussian_blur_reflect_ui8
 * \brief Gaussian blur on src and store to dst
 *
 *  boarder type reflect  `hgfedcba|abcdefgh|hgfedcba`
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Ouput mat ptr.
 * \param[in] kernel_h Height of gaussian blur kernel.
 * \param[in] kernel_w Width of gaussian blur kernel.
 * \param[in] sigma_x The variance of the Gaussian function in the x-direction(width),
 * used to calculate the kernel.
 * \param[in] sigma_y The variance of the Gaussian function in the y-direction(height),
 * used to calculate the kernel.
 */
void tinycv_gaussian_blur_reflect_ui8(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

void tinycv_gaussian_blur_reflect_f32(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

/**
 * \fn tinycv_gaussian_blur_reflect_101_ui8
 * \brief Gaussian blur on src and store to dst
 *
 *  boarder type reflect_101  `hgfedcb|abcdefgh|gfedcba`
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Ouput mat ptr.
 * \param[in] kernel_h Height of gaussian blur kernel.
 * \param[in] kernel_w Width of gaussian blur kernel.
 * \param[in] sigma_x The variance of the Gaussian function in the x-direction(width),
 * used to calculate the kernel.
 * \param[in] sigma_y The variance of the Gaussian function in the y-direction(height),
 * used to calculate the kernel.
 */
void tinycv_gaussian_blur_reflect_101_ui8(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

void tinycv_gaussian_blur_reflect_101_f32(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

/**
 * \fn tinycv_gaussian_blur_replicate_ui8
 * \brief Gaussian blur on src and store to dst
 *
 *  boarder type replicate  `aaaaaaa|abcdefgh|hhhhhhhh`
 *
 * \param[in] src Input mat ptr.
 * \param[out] dst Ouput mat ptr.
 * \param[in] kernel_h Height of gaussian blur kernel.
 * \param[in] kernel_w Width of gaussian blur kernel.
 * \param[in] sigma_x The variance of the Gaussian function in the x-direction(width),
 * used to calculate the kernel.
 * \param[in] sigma_y The variance of the Gaussian function in the y-direction(height),
 * used to calculate the kernel.
 */
void tinycv_gaussian_blur_replicate_ui8(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

void tinycv_gaussian_blur_replicate_f32(
        const TinyMat* src, const TinyMat* dst, int kernel_h, int kernel_w,
        double sigma_x, double sigma_y);

#ifdef __cplusplus
}
#endif
#endif
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
