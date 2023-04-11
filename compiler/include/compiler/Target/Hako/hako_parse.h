#pragma once
#include <cstdint>
#include <vector>
namespace megcc {
enum class EncryptionType { NAIVE = 0, SFRC4, RC4, NONE };
std::pair<std::vector<uint8_t>, EncryptionType> parse_model(
        const std::vector<uint8_t>& model_buffer);
}  // namespace megcc
