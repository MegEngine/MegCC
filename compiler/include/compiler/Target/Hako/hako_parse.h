/**
 * \file compiler/include/compiler/Target/Hako/hako_parse.h
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#pragma once
#include <cstdint>
#include <vector>
namespace megcc {
enum class EncryptionType { NAIVE = 0, SFRC4, RC4, NONE };
std::pair<std::vector<uint8_t>, EncryptionType> parse_model(
        const std::vector<uint8_t>& model_buffer);
}  // namespace megcc
