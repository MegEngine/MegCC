/**
 * \file
 * compiler/script/tool/cv_remap_table.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */

//! this file is used to gen
//! compiler/lib/KernelGen/Arm/ArmCommon/InternalKernel/InternalKernel.h
#include <limits.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <string.h>
static inline int min(int x, int y) {
    return x < y ? x : y;
}
static inline int max(int x, int y) {
    return x > y ? x : y;
}
static inline int div_ceil(int x, int r) {
    return (x + r - 1) / r;
}

#define SATURATE_CAST_SHORT(X) (short)min(max((int)(X), SHRT_MIN), SHRT_MAX)
#define INTER_BITS 5
#define AB_BITS 10
static const int AB_SCALE = 1 << AB_BITS;
static const int INTER_TAB_SIZE = (1 << INTER_BITS);
static const int INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE;
static const int BLOCK_SZ = 64;
static const int INTER_REMAP_COEF_BITS = 15;
static const int INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS;

static short global_table[INTER_TAB_SIZE * INTER_TAB_SIZE * 2 * 2];
static bool init_table = false;
static inline void interpolate_linear(float x, float* coeffs) {
    coeffs[0] = 1.f - x;
    coeffs[1] = x;
}
static inline void init_inter_tab_1d(float* tab, int tabsz) {
    float scale = 1.f / tabsz;
    for (int i = 0; i < tabsz; ++i, tab += 2)
        interpolate_linear(i * scale, tab);
}

static short* get_table() {
    const int ksize = 2;
    if (init_table) {
        return &global_table[0];
    } else {
        short* itab = global_table;
        float _tab[8 * INTER_TAB_SIZE];
        int i, j, k1, k2;
        init_inter_tab_1d(_tab, INTER_TAB_SIZE);
        for (i = 0; i < INTER_TAB_SIZE; ++i) {
            for (j = 0; j < INTER_TAB_SIZE; ++j, itab += ksize * ksize) {
                int isum = 0;
                for (k1 = 0; k1 < ksize; ++k1) {
                    float vy = _tab[i * ksize + k1];
                    for (k2 = 0; k2 < ksize; ++k2) {
                        float v = vy * _tab[j * ksize + k2];
                        isum += itab[k1 * ksize + k2] =
                                SATURATE_CAST_SHORT(v * INTER_REMAP_COEF_SCALE);
                    }
                }
                if (isum != INTER_REMAP_COEF_SCALE) {
                    int diff = isum - INTER_REMAP_COEF_SCALE;
                    int ksize2 = ksize / 2, Mk1 = ksize2, Mk2 = ksize2;
                    int mk1 = ksize2, mk2 = ksize2;
                    for (k1 = ksize2; k1 < ksize2 + 2; ++k1)
                        for (k2 = ksize2; k2 < ksize2 + 2; ++k2) {
                            if (itab[k1 * ksize + k2] <
                                itab[mk1 * ksize + mk2]) {
                                mk1 = k1;
                                mk2 = k2;
                            } else if (itab[k1 * ksize + k2] >
                                       itab[Mk1 * ksize + Mk2]) {
                                Mk1 = k1;
                                Mk2 = k2;
                            }
                        }
                    if (diff < 0)
                        itab[Mk1 * ksize + Mk2] =
                                (short)(itab[Mk1 * ksize + Mk2] - diff);
                    else
                        itab[mk1 * ksize + mk2] =
                                (short)(itab[mk1 * ksize + mk2] - diff);
                }
            }
        }
        init_table = true;
        return &global_table[0];
    }
}
int main() {
    short* table = get_table();
    for (int i = 0; i < INTER_TAB_SIZE * INTER_TAB_SIZE; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%d, ", *table);
            ++table;
        }
        printf("\n");
    }
}