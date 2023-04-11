#include "compiler/Target/Hako/hako_parse.h"
#include <string.h>
#include <fstream>
#include <memory>
#include "compiler/Common/Logger.h"
#include "rc4/rc4_cryption_base.h"
#include "rc4_cryption.h"
using namespace megcc;

std::vector<uint8_t> find_prefix(
        const std::vector<uint8_t>& src,
        const std::pair<std::string, int>& prefix_offset) {
    const auto& prefix = prefix_offset.first;
    const int offset = prefix_offset.second;
    CC_ASSERT(src.size() > prefix.size());
    CC_ASSERT(prefix.size() > 0);
    std::vector<uint8_t> res;
    for (size_t i = 0; i < src.size() - prefix.size(); ++i) {
        if (src[i] == prefix[0]) {
            bool match = true;
            for (size_t j = 0; j < prefix.size(); j++) {
                if (src[i + j] != prefix[j]) {
                    match = false;
                    break;
                }
            }
            if (match) {
                res.resize(src.size() - i - offset);
                memcpy(res.data(), ((const uint8_t*)src.data()) + i + offset,
                       src.size() - i - offset);
                return res;
            }
        }
    }
    return res;
}

template <class T>
std::vector<uint8_t> decrypt_model(const std::vector<uint8_t>& model_buffer) {
    auto&& result = T::decrypt_model(
            model_buffer.data(), model_buffer.size(), T::get_decrypt_key());
    std::vector<std::pair<std::string, int>> valid_magic{
            {"mgb0001", 0}, {"mgb0000a", 0}, {"MGBC", 0}, {"MGBS", 0}, {"mgv2", -8}};
    std::vector<uint8_t> mdl_result;
    if (result.size()) {
        for (auto& prefix : valid_magic) {
            mdl_result = find_prefix(result, prefix);
            if (mdl_result.size() > 0) {
                break;
            }
        }
    }
    return mdl_result;
}

std::pair<std::vector<uint8_t>, EncryptionType> megcc::parse_model(
        const std::vector<uint8_t>& model_buffer) {
    LOG_DEBUG << "Now supports non-encrypted models and models encrypted with NAIVE, "
                 "RC4 and SimpleFastRC4 algorithms.\n";
    LOG_DEBUG << "Try to parse non-encrypted model...\n";
    auto mdl_result = decrypt_model<FakeEncrypt>(model_buffer);
    if (mdl_result.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {mdl_result, EncryptionType::NONE};
    }

    LOG_DEBUG << "Parse non-encrypted model failed. Try to parse model using "
                 "`NaiveEncrypt`...\n";
    mdl_result = decrypt_model<NaiveEncrypt>(model_buffer);
    if (mdl_result.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {mdl_result, EncryptionType::NAIVE};
    }

    LOG_DEBUG << "Parse model using `NaiveEncrypt` failed. Try to parse model using "
                 "`SimpleFastRC4`...\n";
    mdl_result = decrypt_model<SimpleFastRC4>(model_buffer);
    if (mdl_result.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {mdl_result, EncryptionType::SFRC4};
    }

    LOG_DEBUG << "Parse model using `SimpleFastRC4` failed. Try to parse model using "
                 "`RC4`...\n";
    mdl_result = decrypt_model<RC4>(model_buffer);
    if (mdl_result.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {mdl_result, EncryptionType::RC4};
    }

    CC_ASSERT(mdl_result.size() > 0) << "can not parse model\n";
    return {};
}