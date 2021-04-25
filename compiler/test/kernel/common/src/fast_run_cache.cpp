/**
 * \file
 * compiler/test/kernel/common/src/fast_run_cache.cpp
 *
 * This file is part of MegCC, a deep learning compiler developed by Megvii.
 *
 * \copyright Copyright (c) 2021-2022 Megvii Inc. All rights reserved.
 */
#include "megbrain/common.h"
#include "test/kernel/common/fast_run_cache.h"

using namespace megdnn;
using namespace test;

FastRunCache::SearchItemStorage::SearchItemStorage(
        const Algorithm::SearchItem& item) {
    Algorithm::serialize_write_pod(item.opr_type, data_hold);
    for (auto&& layout : item.layouts) {
        data_hold += layout.serialize();
    }
    data_hold += item.param;
}

Algorithm::Info::Desc FastRunCache::get(const Algorithm::SearchItem& key) {
    SearchItemStorage key_storage(key);
    key_storage.init_hash();

    auto iter = m_cache.find(key_storage);
    if (iter == m_cache.end()) {
        return {};
    }
    return iter->second;
}

void FastRunCache::put(const Algorithm::SearchItem& key,
                       const Algorithm::Info::Desc& val) {
    SearchItemStorage key_storage(key);
    key_storage.init_hash();
    mgb_assert(m_cache.find(key_storage) == m_cache.end());
    m_cache[std::move(key_storage)] = val;
}

// vim: syntax=cpp.doxygen
