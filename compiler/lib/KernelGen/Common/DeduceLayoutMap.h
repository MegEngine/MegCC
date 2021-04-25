/**
 * \file
 * compiler/lib/KernelGen/Common/DeduceLayoutMap.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <string>
#include "compiler/KernelGen/KernelGen.h"
namespace megcc {
namespace KernelGen {
struct DeduceLayoutMap {
    DeduceLayoutMap();
    std::unordered_map<KernelPack::KernType, std::shared_ptr<DeduceFunc>> map;
};

}  // namespace KernelGen
}  // namespace megcc