#pragma once
#include <stdio.h>
#include <cstdint>
#include <vector>
namespace megcc {
enum class EncryptionType { NAIVE = 0, SFRC4, RC4, NONE };
struct DecryptedModel {
    // the header added by `build.py` when encrypt using `hako`
    std::vector<uint8_t> hako_header;
    std::vector<uint8_t> model;
    EncryptionType enc_type;
    DecryptedModel() = default;
    DecryptedModel(
            const std::vector<uint8_t>& hako_header, const std::vector<uint8_t>& model,
            EncryptionType enc_type)
            : hako_header(hako_header), model(model), enc_type(enc_type) {}
};
DecryptedModel parse_model(std::vector<uint8_t> model_buffer);
}  // namespace megcc
