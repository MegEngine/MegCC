/**
 * \file
 * compiler/test/kernel/common/dnn_proxy_deduce.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once

#include "megbrain/common.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace test {

template <typename Opr, size_t Arity, bool can_deduce_layout>
struct DeduceLayoutProxy;

template <typename Opr, size_t Arity>
struct DeduceLayoutProxy<Opr, Arity, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 2, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        mgb_assert(layouts.size() == 2);
        opr->deduce_layout(layouts[0], layouts[1]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 3, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        mgb_assert(layouts.size() == 3);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 4, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        mgb_assert(layouts.size() == 4);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 5, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        mgb_assert(layouts.size() == 5);
        opr->deduce_layout(layouts[0], layouts[1], layouts[2], layouts[3], layouts[4]);
    }
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 5, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 6, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 7, false> {
    static void deduce_layout(Opr*, TensorLayoutArray&) {}
};

template <typename Opr>
struct DeduceLayoutProxy<Opr, 8, true> {
    static void deduce_layout(Opr* opr, TensorLayoutArray& layouts) {
        mgb_assert(layouts.size() == 8);
        opr->deduce_layout(
                layouts[0], layouts[1], layouts[2], layouts[3], layouts[4], layouts[5],
                layouts[6], layouts[7]);
    }
};

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
