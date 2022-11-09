/**
 * \file compiler/include/compiler/Target/Hako/hako_parse.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <vector>
namespace megcc {
std::vector<uint8_t> parse_hako(const std::vector<uint8_t>& model_buffer,
                                int version = 2);
}  // namespace megcc
