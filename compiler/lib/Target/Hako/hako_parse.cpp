#include "compiler/Target/Hako/hako_parse.h"
#include <string.h>
#include <memory>
#include <unordered_set>
#include "compiler/Common/Logger.h"
#include "rc4/rc4_cryption_base.h"
#include "rc4_cryption.h"
using namespace megcc;

namespace {
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> try_split_head_and_mdl(
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
                int head_len = static_cast<int>(i) + offset;
                CC_ASSERT(head_len >= 0);
                std::vector<uint8_t> head(src.begin(), src.begin() + head_len);
                res.assign(src.begin() + head.size(), src.end());
                return {head, res};
            }
        }
    }
    return {{}, res};
}

template <class T>
std::pair<std::vector<uint8_t>, std::vector<uint8_t>> decrypt_model(
        const std::vector<uint8_t>& model_buffer) {
    auto&& result = T::decrypt_model(
            model_buffer.data(), model_buffer.size(), T::get_decrypt_key());
    std::vector<std::pair<std::string, int>> valid_magic{
            {"mgb0001", 0}, {"mgb0000a", 0}, {"MGBC", 0}, {"MGBS", 0}, {"mgv2", -8}};
    std::pair<std::vector<uint8_t>, std::vector<uint8_t>> head_and_mdl;
    if (result.size()) {
        for (auto& prefix : valid_magic) {
            head_and_mdl = try_split_head_and_mdl(result, prefix);
            if (head_and_mdl.second.size() > 0) {
                break;
            }
        }
    }
    return head_and_mdl;
}
}  // namespace

DecryptedModel megcc::parse_model(std::vector<uint8_t> model_buffer) {
    std::unordered_set<std::string> valid_anc_magic{
            "anc00001", "anc00002", "mgf00001", "mgf00002"};
    constexpr int magic_len = 8;
    std::string magic(
            reinterpret_cast<const char*>(model_buffer.data()) + sizeof(int32_t),
            magic_len);
    if (valid_anc_magic.find(magic) != valid_anc_magic.end()) {
        //! ancbase(megface) model format accroding to `ancbase code`:
        //! |xxxx(4 bytes)|magic(8 bytes)|model_offset(int, 4 bytes)|model_len(int, 4
        //! bytes)|...|emod|
        //! BUT, actually, offset(emod) = model_len. In other words, the offset of
        //! `emod` model is `model_len`.
        int len = *reinterpret_cast<const int*>(
                model_buffer.data() + sizeof(int32_t) + magic_len + sizeof(int));
        model_buffer.assign(model_buffer.begin() + len, model_buffer.end());
    }
    LOG_DEBUG << "Now supports non-encrypted models and models encrypted with NAIVE, "
                 "RC4 and SimpleFastRC4 algorithms.\n";
    LOG_DEBUG << "Try to parse non-encrypted model...\n";
    auto head_and_mdl = decrypt_model<FakeEncrypt>(model_buffer);
    if (head_and_mdl.second.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {head_and_mdl.first, head_and_mdl.second, EncryptionType::NONE};
    }

    LOG_DEBUG << "Parse non-encrypted model failed. Try to parse model using "
                 "`NaiveEncrypt`...\n";
    head_and_mdl = decrypt_model<NaiveEncrypt>(model_buffer);
    if (head_and_mdl.second.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {head_and_mdl.first, head_and_mdl.second, EncryptionType::NAIVE};
    }

    LOG_DEBUG << "Parse model using `NaiveEncrypt` failed. Try to parse model using "
                 "`SimpleFastRC4`...\n";
    head_and_mdl = decrypt_model<SimpleFastRC4>(model_buffer);
    if (head_and_mdl.second.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {head_and_mdl.first, head_and_mdl.second, EncryptionType::SFRC4};
    }

    LOG_DEBUG << "Parse model using `SimpleFastRC4` failed. Try to parse model using "
                 "`RC4`...\n";
    head_and_mdl = decrypt_model<RC4>(model_buffer);
    if (head_and_mdl.second.size()) {
        LOG_DEBUG << "Parse model successfully.\n";
        return {head_and_mdl.first, head_and_mdl.second, EncryptionType::RC4};
    }

    CC_ASSERT(0) << "can not parse model\n";
    return {};
}